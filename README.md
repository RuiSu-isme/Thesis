# Thesis
Command for resampling 
cd to/your/folder
python resample_gaze.py "\yourgazedata" "\outputfile"

Command for NSS Score
cd to/your/folder
python nss_score_new.py --gaze_file "\your_gazefile.csv"  --saliency_dir ./your_saliency_frames --frame_col your_frame_column_name  --x_col x_col_name --y_col y_col_name --output ./nss_Output/youroutput.csv
