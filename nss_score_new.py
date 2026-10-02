"""
Correlate gaze data with saliency maps using NSS (Normalized Scanpath Saliency).

WHAT THIS DOES
--------------
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
A CSV/Excel file with (at minimum) a frame column and an x/y pair.
These layouts are detected automatically:
  frame, x, y                            generic / older exports
  videoFrame, gaze_x, gaze_y             resample_gaze.py output
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

BATCH MODE
----------
--gaze_dir scores every gaze file in a folder in a single run. Two ways to
say which saliency maps belong to those files:

  --saliency_dir DIR    one map folder for the whole batch - the normal case,
                        e.g. all 55 participants of blockTowerR video
  --saliency_root DIR   map folder picked per gaze file from the task name in
                        its file name
                        (1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv
                         -> DIR\\bananaQantH), for batches spanning tasks;
                        add --recursive to walk subfolders

Every participant of one video shares the same map folder, so the batch reads
each saliency map ONCE and looks up all participants for that frame, instead
of re-reading the same 1080p PNG once per participant. A 55-file batch costs
roughly what scoring a single file costs.

OUTPUT
------
  <output_dir>\\<group>\\<gaze file name>.csv        per-frame NSS (frame,x,y,nss)
  <output_dir>\\<group>\\<group>_nss_summary.csv    one row per gaze file
"<group>" is the saliency folder name, or the task name parsed from each gaze
file name when --saliency_root is used; --group overrides it.

USAGE
------
Single file (as before):
    python nss_score_new.py --gaze_file gaze.csv --saliency_dir predictions\\myvideo

Batch - all gaze files of one task:
    python nss_score_new.py --gaze_dir Resample_Output\\blockTowerR \\
                            --saliency_dir saliency_frames\\blockTowerR

Batch - many tasks at once:
    python nss_score_new.py --gaze_dir Resample_Output --recursive \\
                            --saliency_root saliency_frames

Optional:
    --output_dir nss_Output            where results are written
    --group NAME                       force the output subfolder name
    --limit 3                          only the first 3 gaze files (quick check)
    --frame_col frame --x_col x --y_col y   override auto-detected columns
    --pattern "frame_{frame:04d}.png"  saliency map file name pattern
    --coords normalized|pixel          how to read x and y
    --output FILE                      (single file only) exact output path
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


# Gaze tables use one of these column layouts; the first complete match wins
# unless the matching --frame_col / --x_col / --y_col are given explicitly.
COLUMN_PRESETS = (
    ("frame", "x", "y"),                    # generic / older exports
    ("videoFrame", "gaze_x", "gaze_y"),     # resample_gaze.py output
)

GAZE_EXTENSIONS = (".csv", ".xlsx", ".xls")


def load_gaze_data(gaze_file):
    gaze_file = str(gaze_file)
    if gaze_file.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(gaze_file)
    return pd.read_csv(gaze_file)


def resolve_columns(df, frame_col, x_col, y_col):
    """Work out which columns hold the frame number and the gaze position.

    Returns (frame, x, y) column names, or None when the table does not have
    a usable combination.
    """
    if frame_col and x_col and y_col:
        if all(c in df.columns for c in (frame_col, x_col, y_col)):
            return frame_col, x_col, y_col
        return None

    for preset in COLUMN_PRESETS:
        candidate = (
            frame_col or preset[0],
            x_col or preset[1],
            y_col or preset[2],
        )
        if all(c in df.columns for c in candidate):
            return candidate

    return None


def build_frame_entries(df, cols):
    """Group the gaze table into {frame: [(row, x, y), ...]}.

    The row number is kept so every output keeps the row order of its input
    file. Rows without a usable frame number or without both coordinates are
    counted and skipped, exactly like the single-file version skipped them.
    """
    frame_col, x_col, y_col = cols

    frames = pd.to_numeric(df[frame_col], errors="coerce")
    xs = pd.to_numeric(df[x_col], errors="coerce")
    ys = pd.to_numeric(df[y_col], errors="coerce")

    entries = defaultdict(list)
    skipped_bad_frame = 0
    skipped_nan = 0

    for row_number, (frame, x, y) in enumerate(zip(frames, xs, ys)):
        if pd.isna(frame):
            skipped_bad_frame += 1
            continue
        if pd.isna(x) or pd.isna(y):
            skipped_nan += 1
            continue
        entries[int(frame)].append((row_number, float(x), float(y)))

    return entries, skipped_bad_frame, skipped_nan


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


def task_name_from_file(path):
    """'1268_bl_2025-04-28_bananaQantH.mp4_gaze.csv' -> 'bananaQantH'."""
    stem = Path(path).stem
    for suffix in (".mp4_gaze", "_gaze"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def collect_gaze_files(args):
    """All gaze tables the run should score, in a stable order."""
    if args.gaze_file:
        return [Path(args.gaze_file)]

    root = Path(args.gaze_dir)
    pattern = "**/*" if args.recursive else "*"
    files = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in GAZE_EXTENSIONS
    ]

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    return files


def saliency_dir_for(path, args):
    """Which map folder belongs to this gaze file, and the output group name."""
    if args.saliency_root:
        task = task_name_from_file(path)
        folder = os.path.join(args.saliency_root, task)
        if not os.path.isdir(folder):
            return None, task, f"no saliency folder '{task}' under {args.saliency_root}"
        return folder, task, None

    group = args.group or os.path.basename(os.path.normpath(args.saliency_dir))
    return args.saliency_dir, group, None


def existing_names(directory):
    """Names present in a map folder, or None when the pattern needs a path."""
    try:
        return set(os.listdir(directory))
    except OSError:
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-frame NSS for one gaze file or a whole folder of them."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gaze_file", help="Single CSV/Excel file with gaze data")
    source.add_argument("--gaze_dir", help="Folder with gaze files to process in one run")

    maps = parser.add_mutually_exclusive_group(required=True)
    maps.add_argument("--saliency_dir", help="Folder with the per-frame saliency maps")
    maps.add_argument("--saliency_root",
                      help="Folder with one saliency map subfolder per task; the "
                           "task is read from each gaze file name")

    parser.add_argument("--recursive", action="store_true",
                        help="With --gaze_dir: also look in subfolders")
    parser.add_argument("--limit", type=int, default=0,
                        help="With --gaze_dir: stop after this many gaze files (0 = all)")
    parser.add_argument("--output_dir", default="nss_Output",
                        help="Where per-file results and summaries are written")
    parser.add_argument("--group", default=None,
                        help="Force the output subfolder name (default: task name)")
    parser.add_argument("--output", default=None,
                        help="Single file only: exact output path for the per-frame CSV")

    parser.add_argument("--frame_col", default=None)
    parser.add_argument("--x_col", default=None)
    parser.add_argument("--y_col", default=None)
    parser.add_argument("--pattern", default="frame_{frame:04d}.png",
                        help="Filename pattern for saliency maps, using {frame} as placeholder")
    parser.add_argument("--coords", choices=["normalized", "pixel"], default="normalized",
                        help="'normalized' (default): x,y in [0,1], (0,0)=top-left, (1,1)=bottom-right. "
                             "'pixel': x,y are already absolute pixel coordinates.")
    args = parser.parse_args()

    if args.gaze_dir and not os.path.isdir(args.gaze_dir):
        print(f"ERROR: gaze folder does not exist: {args.gaze_dir}")
        sys.exit(1)
    if args.gaze_file and not os.path.isfile(args.gaze_file):
        print(f"ERROR: gaze file does not exist: {args.gaze_file}")
        sys.exit(1)
    if args.saliency_dir and not os.path.isdir(args.saliency_dir):
        print(f"ERROR: saliency folder does not exist: {args.saliency_dir}")
        sys.exit(1)
    if args.saliency_root and not os.path.isdir(args.saliency_root):
        print(f"ERROR: saliency root does not exist: {args.saliency_root}")
        sys.exit(1)
    if args.output and args.gaze_dir:
        print("ERROR: --output writes a single file; use --output_dir with --gaze_dir")
        sys.exit(1)

    gaze_files = collect_gaze_files(args)
    if not gaze_files:
        print(f"ERROR: no gaze files found in {args.gaze_dir}")
        sys.exit(1)

    print("================================")
    print("NSS SCORING")
    print("================================")
    print(f"Gaze files : {len(gaze_files)}")
    print(f"Saliency   : {args.saliency_dir or (args.saliency_root + ' (per task)')}")
    print(f"Coords     : {args.coords}")
    print()

    # ----------------------------------------------------------------
    # 1. Read every gaze table and group its rows by frame
    # ----------------------------------------------------------------

    states = []

    for path in gaze_files:
        state = {
            "path": str(path),
            "name": path.name,
            "sal_dir": None,
            "group": None,
            "entries": {},
            "results": {},
            "rows_scored": 0,
            "skipped_missing_file": 0,
            "skipped_out_of_bounds": 0,
            "skipped_nan": 0,
            "skipped_bad_frame": 0,
            "columns": None,
            "error": None,
        }
        states.append(state)

        sal_dir, group, error = saliency_dir_for(path, args)
        state["group"] = group
        if error:
            state["error"] = error
            continue
        state["sal_dir"] = sal_dir

        try:
            df = load_gaze_data(path)
        except Exception as exc:
            state["error"] = f"could not read the file: {exc}"
            continue

        cols = resolve_columns(df, args.frame_col, args.x_col, args.y_col)
        if cols is None:
            state["error"] = (
                "no frame/x/y columns (looked for "
                + " | ".join(",".join(p) for p in COLUMN_PRESETS)
                + f"; columns found: {list(df.columns)})"
            )
            continue
        state["columns"] = cols

        entries, skipped_bad_frame, skipped_nan = build_frame_entries(df, cols)
        state["entries"] = entries
        state["skipped_bad_frame"] = skipped_bad_frame
        state["skipped_nan"] = skipped_nan

    usable = [s for s in states if s["sal_dir"]]

    if not usable:
        print("Nothing to score:")
        for state in states:
            print(f"  {state['name']}: {state['error']}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Score frame by frame, one pass per saliency folder
    #    Each map image is read once and reused for every gaze file.
    # ----------------------------------------------------------------

    buckets = defaultdict(lambda: defaultdict(list))
    for index, state in enumerate(states):
        if not state["sal_dir"]:
            continue
        for frame, rows in state["entries"].items():
            for row_number, x, y in rows:
                buckets[state["sal_dir"]][frame].append((index, row_number, x, y))

    for sal_dir in sorted(buckets):
        frames = buckets[sal_dir]
        names = existing_names(sal_dir) if os.sep not in args.pattern else None
        total = len(frames)

        print(f"Saliency folder: {sal_dir}")
        print(f"  frames to score: {total}")

        for done, frame in enumerate(sorted(frames), start=1):
            filename = args.pattern.format(frame=frame)
            filepath = os.path.join(sal_dir, filename)

            found = (filename in names) if names is not None else os.path.exists(filepath)
            if not found:
                for index, _, _, _ in frames[frame]:
                    states[index]["skipped_missing_file"] += 1
                continue

            saliency_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if saliency_map is None:
                for index, _, _, _ in frames[frame]:
                    states[index]["skipped_missing_file"] += 1
                continue

            for index, row_number, x, y in frames[frame]:
                nss = compute_nss(saliency_map, x, y, coords=args.coords)
                if nss is None:
                    states[index]["skipped_out_of_bounds"] += 1
                    continue
                states[index]["results"][row_number] = (frame, x, y, nss)
                states[index]["rows_scored"] += 1

            if done % 500 == 0 or done == total:
                print(f"    {done}/{total} frames scored")

        print()

    # ----------------------------------------------------------------
    # 3. Write one per-frame CSV per gaze file, plus a summary per task
    # ----------------------------------------------------------------

    def frame_rows(state):
        return [
            {"frame": frame, "x": x, "y": y, "nss": nss}
            for row_number, (frame, x, y, nss) in sorted(state["results"].items())
        ]

    # Single file with an explicit destination: behave like the old script.
    if args.output:
        state = states[0]
        if not state["results"]:
            print(f"No NSS scores could be computed for {state['name']}.")
            if state["error"]:
                print(f"  {state['error']}")
            else:
                print(f"  {state['skipped_missing_file']} rows had no saliency map file")
            sys.exit(1)

        results_df = pd.DataFrame(frame_rows(state), columns=["frame", "x", "y", "nss"])
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        results_df.to_csv(args.output, index=False)

        print(f"\nDone. Computed NSS for {len(results_df)} frames.")
        if state["skipped_missing_file"]:
            print(f"  Skipped {state['skipped_missing_file']} rows (saliency map file not found)")
        if state["skipped_out_of_bounds"]:
            print(f"  Skipped {state['skipped_out_of_bounds']} rows (gaze point outside frame bounds)")
        if state["skipped_nan"]:
            print(f"  Skipped {state['skipped_nan']} rows (missing/NaN x or y value)")
        if state["skipped_bad_frame"]:
            print(f"  Skipped {state['skipped_bad_frame']} rows (missing/NaN frame number)")

        print(f"\nMean NSS: {results_df['nss'].mean():.4f}")
        print(f"Std NSS:  {results_df['nss'].std():.4f}")
        print(f"Min/Max:  {results_df['nss'].min():.4f} / {results_df['nss'].max():.4f}")
        print(f"\nPer-frame results saved to: {args.output}")
        return

    summaries = defaultdict(list)
    files_with_scores = 0

    for state in states:
        if not state["results"]:
            continue

        results_df = pd.DataFrame(frame_rows(state), columns=["frame", "x", "y", "nss"])

        group_dir = os.path.join(args.output_dir, state["group"]) if state["group"] else args.output_dir
        os.makedirs(group_dir, exist_ok=True)
        out_path = os.path.join(group_dir, state["name"])

        results_df.to_csv(out_path, index=False)
        files_with_scores += 1

        summaries[state["group"] or ""].append({
            "gaze_file": state["name"],
            "task": state["group"] or "",
            "rows_in_file": len(state["results"]) + state["skipped_missing_file"]
                            + state["skipped_out_of_bounds"] + state["skipped_nan"]
                            + state["skipped_bad_frame"],
            "rows_scored": len(results_df),
            "mean_nss": results_df["nss"].mean(),
            "std_nss": results_df["nss"].std(),
            "min_nss": results_df["nss"].min(),
            "max_nss": results_df["nss"].max(),
            "skipped_no_map": state["skipped_missing_file"],
            "skipped_out_of_bounds": state["skipped_out_of_bounds"],
            "skipped_missing_xy": state["skipped_nan"],
            "skipped_bad_frame": state["skipped_bad_frame"],
            "status": "ok",
        })

    summary_paths = {}
    for group, rows in summaries.items():
        group_dir = os.path.join(args.output_dir, group) if group else args.output_dir
        os.makedirs(group_dir, exist_ok=True)
        name = f"{group}_nss_summary.csv" if group else "nss_summary.csv"
        summary_path = os.path.join(group_dir, name)
        summary_df = pd.DataFrame(rows).sort_values("gaze_file")
        summary_df.to_csv(summary_path, index=False)
        summary_paths[group] = summary_path

    # ----------------------------------------------------------------
    # 4. Report
    # ----------------------------------------------------------------

    print("================================")
    print("RESULTS")
    print("================================")

    for group, rows in summaries.items():
        label = group or "(no group)"
        per_file_means = [r["mean_nss"] for r in rows]
        scored = sum(r["rows_scored"] for r in rows)
        print(f"\n{label}: {len(rows)} files scored, {scored} gaze frames")
        print(f"  mean of per-file mean NSS: {np.mean(per_file_means):.4f}")
        print(f"  spread of per-file means : "
              f"{np.min(per_file_means):.4f} .. {np.max(per_file_means):.4f}")
        print(f"  saved to: {os.path.join(args.output_dir, group) if group else args.output_dir}")
        print(f"  summary : {summary_paths.get(group)}")

    failed = [s for s in states if s["error"]]
    if failed:
        print(f"\nSkipped {len(failed)} file(s):")
        for state in failed:
            print(f"  {state['name']}: {state['error']}")

    no_scores = [s for s in states if not s["error"] and not s["results"]]
    if no_scores:
        print(f"\nNo NSS could be computed for {len(no_scores)} file(s) "
              f"(no matching saliency map for their frames):")
        for state in no_scores[:10]:
            print(f"  {state['name']}: {state['skipped_missing_file']} frames without a map")
        if len(no_scores) > 10:
            print(f"  ... and {len(no_scores) - 10} more")

    total_oob = sum(s["skipped_out_of_bounds"] for s in states)
    total_nan = sum(s["skipped_nan"] for s in states)
    if total_oob:
        print(f"\nSkipped {total_oob} rows in total (gaze point outside frame bounds)")
    if total_nan:
        print(f"Skipped {total_nan} rows in total (missing/NaN x or y value)")

    print()
    if files_with_scores == 0:
        print("No NSS scores could be computed. Check your file paths and column names.")
        sys.exit(1)

    print(f"Done. {files_with_scores} per-frame CSV file(s) written, "
          f"{len(gaze_files)} gaze file(s) considered.")


if __name__ == "__main__":
    main()
