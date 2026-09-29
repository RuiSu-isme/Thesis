"""
Correlate gaze data with saliency maps using NSS (Normalized Scanpath Saliency).

WHAT THIS DOES
---------------
For each frame:
  1. Loads the saliency map image (grayscale) for that frame
  2. Normalizes it to zero mean, unit standard deviation (this is what
     makes it "Normalized" - it accounts for maps that are overall
     brighter/dimmer or more/less contrasty)
  3. Looks up the normalized saliency value exactly at the gaze (x, y)
     coordinate for that frame
  4. That value is the NSS score for the frame

INTERPRETING THE RESULT
------------------------
  NSS > 0   -> gaze fell on a brighter-than-average (more salient) region
  NSS = 0   -> gaze fell on an average-brightness region (no better than chance)
  NSS < 0   -> gaze fell on a darker-than-average (less salient) region
  Higher average NSS across frames = the saliency map predicts real
  gaze behavior better.

EXPECTED INPUT FORMAT
-----------------------
A CSV/Excel file with (at minimum) these three columns:
  frame, x, y
  1,     512, 340
  2,     515, 338
  ...
If your columns are named differently, pass --frame_col / --x_col / --y_col
to tell the script what to use.

COORDINATE FORMAT
-------------------
By default, x/y are assumed to be NORMALIZED to [0, 1], where (0, 0) is the
top-left corner and (1, 1) is the bottom-right corner of the frame (so
(0.5, 0.5) is the exact center). This is the default because it's what the
gaze-tracking data for this project uses. Each row's normalized (x, y) is
converted to pixel coordinates using that specific frame's saliency map
dimensions (width, height), so it stays correct even if maps happen to be
different sizes.

If your gaze data is already in absolute pixel coordinates instead, pass
--coords pixel to skip the normalization step.

Saliency maps are expected to be image files named with the frame number,
e.g. frame_0001.png, frame_0002.png, ... (this naming can be changed with
--pattern, using Python's format syntax with {frame:04d} as the placeholder).

USAGE
------
    python nss_score.py --gaze_file gaze.csv --saliency_dir ./predictions/vsod/MyData/myvideo

Optional:
    --frame_col frame --x_col x --y_col y   (rename if your columns differ)
    --pattern "frame_{frame:04d}.png"       (change filename pattern)
    --output nss_scores.csv                 (where to save per-frame results)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import cv2


def load_gaze_data(gaze_file, frame_col, x_col, y_col):
    if gaze_file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(gaze_file)
    else:
        df = pd.read_csv(gaze_file)

    missing = [c for c in (frame_col, x_col, y_col) if c not in df.columns]
    if missing:
        print(f"ERROR: could not find column(s) {missing} in {gaze_file}")
        print(f"Columns found in the file: {list(df.columns)}")
        sys.exit(1)

    return df


def compute_nss(saliency_map, x, y, coords="normalized"):
    """Normalize the map (zero mean, unit std) and sample the value at (x, y).

    coords="normalized": x, y are in [0, 1], (0,0)=top-left, (1,1)=bottom-right,
        converted to pixel coordinates using this map's own width/height.
    coords="pixel": x, y are already absolute pixel coordinates.
    """
    saliency_map = saliency_map.astype(np.float64)
    h, w = saliency_map.shape

    std = saliency_map.std()
    if std == 0:
        # Flat map (e.g. no detections at all this frame) - NSS is undefined,
        # we treat it as 0 (no information either way)
        return 0.0

    normalized = (saliency_map - saliency_map.mean()) / std

    if coords == "normalized":
        px = x * w
        py = y * h
    else:
        px = x
        py = y

    ix, iy = int(round(px)), int(round(py))

    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return None  # gaze point fell outside the frame

    return float(normalized[iy, ix])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze_file", required=True, help="CSV or Excel file with gaze data")
    parser.add_argument("--saliency_dir", required=True, help="Folder containing per-frame saliency map images")
    parser.add_argument("--frame_col", default="frame")
    parser.add_argument("--x_col", default="x")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--pattern", default="frame_{frame:04d}.png",
                         help="Filename pattern for saliency maps, using {frame} as placeholder")
    parser.add_argument("--coords", choices=["normalized", "pixel"], default="normalized",
                         help="'normalized' (default): x,y in [0,1], (0,0)=top-left, (1,1)=bottom-right. "
                              "'pixel': x,y are already absolute pixel coordinates.")
    parser.add_argument("--output", default="nss_scores.csv")
    args = parser.parse_args()

    gaze_df = load_gaze_data(args.gaze_file, args.frame_col, args.x_col, args.y_col)

    results = []
    skipped_missing_file = 0
    skipped_out_of_bounds = 0
    skipped_nan = 0

    print(f"Processing {len(gaze_df)} gaze rows... (coords mode: {args.coords})")

    for _, row in gaze_df.iterrows():
        frame_num = int(row[args.frame_col])
        x = row[args.x_col]
        y = row[args.y_col]

        if pd.isna(x) or pd.isna(y):
            skipped_nan += 1
            continue

        filename = args.pattern.format(frame=frame_num)
        filepath = os.path.join(args.saliency_dir, filename)

        if not os.path.exists(filepath):
            skipped_missing_file += 1
            continue

        saliency_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if saliency_map is None:
            skipped_missing_file += 1
            continue

        nss = compute_nss(saliency_map, x, y, coords=args.coords)
        if nss is None:
            skipped_out_of_bounds += 1
            continue

        results.append({"frame": frame_num, "x": x, "y": y, "nss": nss})

    if not results:
        print("No NSS scores could be computed. Check your file paths and column names.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    print(f"\nDone. Computed NSS for {len(results_df)} frames.")
    if skipped_missing_file:
        print(f"  Skipped {skipped_missing_file} rows (saliency map file not found)")
    if skipped_out_of_bounds:
        print(f"  Skipped {skipped_out_of_bounds} rows (gaze point outside frame bounds)")
    if skipped_nan:
        print(f"  Skipped {skipped_nan} rows (missing/NaN x or y value)")

    print(f"\nMean NSS: {results_df['nss'].mean():.4f}")
    print(f"Std NSS:  {results_df['nss'].std():.4f}")
    print(f"Min/Max:  {results_df['nss'].min():.4f} / {results_df['nss'].max():.4f}")
    print(f"\nPer-frame results saved to: {args.output}")


if __name__ == "__main__":
    main()
