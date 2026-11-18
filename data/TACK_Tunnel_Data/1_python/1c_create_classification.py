# Document info
__author__ = 'Andreas Sjolander, Gemini'
__version__ = ['1.0'] 
__version_date__ = '2025-11-25' 
__maintainer__ = 'Andreas Sjolander'
__github__ = 'andreassjolander'
__email__ = 'asjola@kth.se'

"""
1c_create_classification.py
this scripts reads the csv files that contain information about images with and
withou cracks. Based on this, three classification datasets are created in the
folder "3_classification", i.e. TA, TB and TC. Each folder contains the 
subfolder "crack" and "no_crack"
"""

##################################
# IMPORT PACKAGES
##################################
import os
import shutil
import pandas as pd
import sys

##################################
# SPECIFY WORKING PATHS
##################################
# 1. Get the root directory (assuming script is running from the root)
project_root = os.getcwd()

# 2. Define Input folders
# The folder where your CSV files are located
input_csv_folder = os.path.join(project_root, "../2_model_input")
# The folder where your images are currently stored
source_img_folder = os.path.join(project_root, "../3_img")
# 3. Define Output folder
output_base_folder = os.path.join(project_root, "../3_classification")

##################################
# MAIN EXECUTION
##################################

def sort_classification_data():
    print(f"--- Starting Classification Sorting ---")
    print(f"Root Directory: {project_root}")
    print(f"Source Images : {source_img_folder}")
    print(f"Input CSVs    : {input_csv_folder}")
    
    # Datasets to process
    datasets = ["TA", "TB", "TC"]

    for dataset in datasets:
        print(f"\nProcessing Dataset: {dataset}...")
        
        # Construct CSV path
        csv_file = f"{dataset}_dataset_labels.csv"
        csv_path = os.path.join(input_csv_folder, csv_file)

        # Check if CSV exists
        if not os.path.exists(csv_path):
            print(f"  [WARNING] CSV not found: {csv_path}. Skipping.")
            continue

        # Read the CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [Error] Could not read CSV: {e}")
            continue

        # Counters for feedback
        count_crack = 0
        count_no_crack = 0
        count_missing = 0

        # Iterate through each row in the CSV
        for index, row in df.iterrows():
            # 1. Extract the filename
            # The CSV contains "../3 img/filename.png". We only want "filename.png".
            raw_path = str(row['filename'])
            filename = os.path.basename(raw_path) 
            
            # 2. Get the label
            label = str(row['label']).strip().lower() # e.g., "crack" or "no_crack"

            # 3. Define Source Path
            # We look for the file in the local "3 img" folder
            src_path = os.path.join(source_img_folder, filename)

            # 4. Define Destination Path
            # Structure: 3 Classification / TA / crack / filename.png
            dest_dir = os.path.join(output_base_folder, dataset, label)
            dest_path = os.path.join(dest_dir, filename)

            # 5. Copy the file
            if os.path.exists(src_path):
                # Create destination folder if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)
                
                shutil.copy2(src_path, dest_path)
                
                if "no_crack" in label:
                    count_no_crack += 1
                else:
                    count_crack += 1
            else:
                # If file is missing, print a warning (limit to first 5 to avoid spamming console)
                if count_missing < 5:
                    print(f"  [Missing] Could not find image: {src_path}")
                count_missing += 1

        print(f"  Summary for {dataset}:")
        print(f"    - Cracks copied   : {count_crack}")
        print(f"    - No Cracks copied: {count_no_crack}")
        if count_missing > 0:
            print(f"    - Missing images  : {count_missing} (Check filenames or source folder)")

    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    sort_classification_data()