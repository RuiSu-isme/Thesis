import pandas as pd
import numpy as np
import sys
from pathlib import Path


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

if len(sys.argv) != 3:
    print('Usage: python resample_gaze.py "input_folder" "output_folder"')
    sys.exit(1)

INPUT_FOLDER = Path(sys.argv[1])
OUTPUT_FOLDER = Path(sys.argv[2])


# ============================================================
# SETTINGS
# ============================================================

VIDEO_FPS = 60

# Maximum time difference between a video frame and
# the nearest valid eye-tracking sample.
MAX_TIME_DIFF = 1 / 120


# ============================================================
# CHECK INPUT / CREATE OUTPUT FOLDER
# ============================================================

if not INPUT_FOLDER.exists():
    print("ERROR: Input folder does not exist:")
    print(INPUT_FOLDER)
    sys.exit(1)

if not INPUT_FOLDER.is_dir():
    print("ERROR: Input path is not a folder:")
    print(INPUT_FOLDER)
    sys.exit(1)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND ALL CSV FILES
# ============================================================

csv_files = sorted(INPUT_FOLDER.glob("*.csv"))

if len(csv_files) == 0:
    print("No CSV files found in:")
    print(INPUT_FOLDER)
    sys.exit(1)

print("================================")
print("BATCH GAZE RESAMPLING")
print("================================")
print("Input folder:")
print(INPUT_FOLDER)
print()
print("Output folder:")
print(OUTPUT_FOLDER)
print()
print("CSV files found:", len(csv_files))
print()


# ============================================================
# PROCESS EACH FILE
# ============================================================

successful = 0
failed = 0

for file_number, input_file in enumerate(csv_files, start=1):

    print("--------------------------------")
    print(f"[{file_number}/{len(csv_files)}] Processing:")
    print(input_file.name)

    try:

        # ====================================================
        # 1. LOAD DATA
        # ====================================================

        df = pd.read_csv(input_file)

        print("Original rows:", len(df))


        # ====================================================
        # 2. VIDEO TIME
        # ====================================================

        df["videoTime"] = pd.to_numeric(
            df["videoTime"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["videoTime"]
        ).copy()


        # ====================================================
        # 3. GAZE COLUMNS
        # ====================================================

        for column in [
            "eyeXmf",
            "eyeYmf",
            "nudged_eyeX",
            "nudged_eyeY"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        # ====================================================
        # 4. SELECT EFFECTIVE GAZE
        # ====================================================

        calib = (
            df["calib_winner"]
            .astype(str)
            .str.lower()
        )

        df["gaze_x"] = np.where(
            calib == "raw",
            df["eyeXmf"],
            np.where(
                calib == "nudged",
                df["nudged_eyeX"],
                np.nan
            )
        )

        df["gaze_y"] = np.where(
            calib == "raw",
            df["eyeYmf"],
            np.where(
                calib == "nudged",
                df["nudged_eyeY"],
                np.nan
            )
        )


        # ====================================================
        # 5. SORT BY TIME
        # ====================================================

        df = df.sort_values(
            "videoTime"
        ).reset_index(drop=True)


        # ====================================================
        # 6. KEEP VALID GAZE
        # ====================================================

        valid_gaze = df.dropna(
            subset=["gaze_x", "gaze_y"]
        ).copy()

        if len(valid_gaze) == 0:
            raise ValueError(
                "No valid gaze observations found."
            )

        print("Valid gaze observations:", len(valid_gaze))


        # ====================================================
        # 7. CREATE 60-FPS VIDEO TIMELINE
        # ====================================================

        video_end = df["videoTime"].max()

        frame_interval = 1 / VIDEO_FPS

        n_frames = (
            int(np.floor(
                video_end / frame_interval
            )) + 1
        )

        video_times = (
            np.arange(n_frames)
            * frame_interval
        )


        # ====================================================
        # 8. CREATE VIDEO FRAME DATAFRAME
        # ====================================================

        video_df = pd.DataFrame({

            "videoFrame":
                np.arange(
                    1,
                    n_frames + 1
                ),

            "videoTime":
                video_times
        })


        # ====================================================
        # 9. MATCH NEAREST GAZE SAMPLE
        # ====================================================

        valid_gaze = valid_gaze[
            [
                "videoTime",
                "gaze_x",
                "gaze_y",
                "calib_winner"
            ]
        ].sort_values(
            "videoTime"
        )

        matched = pd.merge_asof(
            video_df,
            valid_gaze,
            on="videoTime",
            direction="nearest",
            tolerance=MAX_TIME_DIFF
        )


        # ====================================================
        # 10. IDENTIFY MISSING GAZE
        # ====================================================

        matched["gaze_missing"] = (
            matched["gaze_x"].isna()
            |
            matched["gaze_y"].isna()
        )

        missing_frames = (
            matched["gaze_missing"].sum()
        )

        total_frames = len(matched)

        missing_percentage = (
            missing_frames
            / total_frames
            * 100
        )


        # ====================================================
        # 11. SAVE WITH ORIGINAL FILENAME
        # ====================================================

        output_file = (
            OUTPUT_FOLDER
            / input_file.name
        )

        matched.to_csv(
            output_file,
            index=False
        )


        # ====================================================
        # 12. PRINT SUMMARY
        # ====================================================

        print(
            "Video frames:",
            total_frames
        )

        print(
            "Missing gaze:",
            missing_frames,
            f"({missing_percentage:.2f}%)"
        )

        print("Saved to:")
        print(output_file)

        successful += 1


    except Exception as e:

        failed += 1

        print("ERROR processing this file:")
        print(e)

        print("Skipping this file and continuing...")
        print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("================================")
print("BATCH PROCESSING COMPLETE")
print("================================")

print("Total files:", len(csv_files))
print("Successfully processed:", successful)
print("Failed:", failed)

print()
print("Output folder:")
print(OUTPUT_FOLDER)