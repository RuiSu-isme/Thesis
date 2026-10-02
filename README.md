# Thesis
# Command for resampling 

cd to/your/folder

python resample_gaze.py "\yourgazedata" "\outputfile"

# Command for NSS Score

I have updated the code
Now it can process all the gaze data files in one folder
The output data will be saved to nss_Output, and a folder will be created for the specific video

Command code
python nss_score_new.py --gaze_dir "path/to/your/gazedata folder" --saliency_dir "saliency_frames\e.g. blockTowerR"
