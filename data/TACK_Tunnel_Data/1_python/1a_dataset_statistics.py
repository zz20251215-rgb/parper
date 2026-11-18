# -*- coding: utf-8 -*-
# Document info
__author__ = 'Andreas Sjölander, Gemini'
__version__ = ['1.0']
__version_date__ = '2025-11-25'
__maintainer__ = 'Andreas Sjölander'
__email__ = 'asjola@kth.se'

"""
1a_dataset_statistics.py
This is used to compute statistics of the dataset. 
1. Calculates Total Pixel Area (Resolution * Image Count).
2. Calculates "No Defect" (Background) pixel counts.
3. Calculates Pixel Percentages for all categories.
4. Maintains the TA, TB, TC split.
"""

import os
import numpy as np
from PIL import Image
import csv
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURATION ---
CLASS_MAP = {
    'Crack': 40,
    'Water': 160,
    'Leaching': 200
}

SUB_DATASETS = ['TA', 'TB', 'TC']

def init_stats_structure():
    return {
        'img_count': 0, 
        'total_pixel_area': 0, # NEW: Tracks total surface area (H*W)
        'class_counts': {k: 0 for k in CLASS_MAP.keys()},
        'no_defect_img_count': 0,
        'pixel_counts': {k: 0 for k in CLASS_MAP.keys()}
    }

def calculate_dataset_statistics_complete():
    # --- 1. Setup Paths ---
    script_location = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_location)
    
    mask_folder = os.path.join(root_dir, '3_mask')
    stats_folder = os.path.join(root_dir, '2_statistics')

    os.makedirs(stats_folder, exist_ok=True)

    if not os.path.exists(mask_folder):
        print(f"CRITICAL ERROR: Mask folder not found at: {mask_folder}")
        return

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(valid_exts)]
    
    if not mask_files:
        print("No mask images found.")
        return

    print(f"Found {len(mask_files)} masks. Calculating Pixel Distributions...")
    print("-" * 30)

    # --- 2. Initialize Data Structure ---
    all_stats = {'Total': init_stats_structure()}
    for ds in SUB_DATASETS:
        all_stats[ds] = init_stats_structure()

    errors = 0

    # --- 3. Process Images ---
    try:
        iterator = tqdm(mask_files, desc="Processing", unit="img")
    except ImportError:
        iterator = mask_files
        print("Processing... (install 'tqdm' for a progress bar)")

    for filename in iterator:
        file_path = os.path.join(mask_folder, filename)
        
        # Identify Sub-Dataset
        current_sub_ds = None
        for prefix in SUB_DATASETS:
            if filename.startswith(prefix):
                current_sub_ds = prefix
                break
        
        try:
            with Image.open(file_path) as img:
                # Ensure grayscale
                mask_arr = np.array(img.convert('L'))

            # Get image stats
            img_area = mask_arr.size # Total pixels in this image (H*W)
            unique_vals, counts = np.unique(mask_arr, return_counts=True)
            img_pixel_data = dict(zip(unique_vals, counts))

            # --- UPDATE STATISTICS ---
            targets = ['Total']
            if current_sub_ds:
                targets.append(current_sub_ds)

            for target in targets:
                stats = all_stats[target]
                stats['img_count'] += 1
                stats['total_pixel_area'] += img_area # Add this image's area to total

                has_defect = False
                
                for class_name, pixel_val in CLASS_MAP.items():
                    if pixel_val in img_pixel_data:
                        # Image Count
                        stats['class_counts'][class_name] += 1
                        # Pixel Count
                        stats['pixel_counts'][class_name] += img_pixel_data[pixel_val]
                        has_defect = True
                
                if not has_defect:
                    stats['no_defect_img_count'] += 1

        except Exception as e:
            errors += 1
            # print(f"Error: {e}") 

    # --- 4. Generate Report Paths ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = os.path.join(stats_folder, f'Dataset_Statistics_Full_{timestamp}.txt')
    csv_path = os.path.join(stats_folder, f'Dataset_Statistics_Full_{timestamp}.csv')

    # --- 5. Write TXT Report ---
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(f"DATASET STATISTICS REPORT (FULL)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================\n\n")

        report_order = ['Total'] + sorted(SUB_DATASETS)

        for ds_name in report_order:
            data = all_stats[ds_name]
            total_imgs = data['img_count']
            total_pixels = data['total_pixel_area']

            f.write(f"--- DATASET: {ds_name} ---\n")
            f.write(f"Total Images: {total_imgs}\n")
            f.write(f"Total Pixels: {total_pixels:,}\n")
            
            if total_imgs > 0:
                # A. IMAGE DISTRIBUTION
                f.write(f"  [Image Distribution]\n")
                pct_no = (data['no_defect_img_count'] / total_imgs) * 100
                f.write(f"    No Defect Images: {data['no_defect_img_count']} ({pct_no:.2f}%)\n")
                
                for c_name in CLASS_MAP.keys():
                    count = data['class_counts'][c_name]
                    pct = (count / total_imgs) * 100
                    f.write(f"    {c_name:<16}: {count} ({pct:.2f}%)\n")
                
                # B. PIXEL DISTRIBUTION
                # Calculate Defect Pixels sum
                total_defect_pixels = sum(data['pixel_counts'].values())
                # Calculate No Defect (Background) Pixels
                no_defect_pixels = total_pixels - total_defect_pixels
                
                f.write(f"  [Pixel Distribution]\n")
                
                # Write No Defect Pixels
                nd_pct = (no_defect_pixels / total_pixels) * 100
                f.write(f"    No Defect/Bg    : {no_defect_pixels:,} px ({nd_pct:.4f}%)\n")

                for c_name in CLASS_MAP.keys():
                    px = data['pixel_counts'][c_name]
                    px_pct = (px / total_pixels) * 100
                    f.write(f"    {c_name:<16}: {px:,} px ({px_pct:.4f}%)\n")
            else:
                f.write("  (No images found)\n")
            
            f.write("\n")

        if errors > 0:
            f.write(f"NOTE: {errors} files skipped due to errors.\n")

    # --- 6. Write CSV Report ---
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Dataset', 'Metric Type', 'Class', 'Count', 'Percentage'])
        
        for ds_name in report_order:
            data = all_stats[ds_name]
            total_imgs = data['img_count']
            total_pixels = data['total_pixel_area']
            
            if total_imgs == 0:
                continue

            # 1. Image Stats
            # No Defect
            pct = (data['no_defect_img_count'] / total_imgs) * 100
            writer.writerow([ds_name, 'Image Count', 'No Defect', data['no_defect_img_count'], f"{pct:.2f}%"])
            
            # Defects
            for c_name in CLASS_MAP.keys():
                count = data['class_counts'][c_name]
                pct = (count / total_imgs) * 100
                writer.writerow([ds_name, 'Image Count', c_name, count, f"{pct:.2f}%"])

            # 2. Pixel Stats
            # Calculate No Defect Pixels
            total_defect_pixels = sum(data['pixel_counts'].values())
            no_defect_pixels = total_pixels - total_defect_pixels
            
            # Write No Defect
            nd_pct = (no_defect_pixels / total_pixels) * 100
            writer.writerow([ds_name, 'Pixel Count', 'No Defect / Background', no_defect_pixels, f"{nd_pct:.5f}%"])

            # Write Defects
            for c_name in CLASS_MAP.keys():
                px = data['pixel_counts'][c_name]
                px_pct = (px / total_pixels) * 100
                writer.writerow([ds_name, 'Pixel Count', c_name, px, f"{px_pct:.5f}%"])
            
            writer.writerow([]) # Spacer

    # --- 7. Final Console Output ---
    print("\n" + "="*30)
    print("CALCULATION COMPLETE")
    print(f"Total Images: {all_stats['Total']['img_count']}")
    print(f"Total Pixels: {all_stats['Total']['total_pixel_area']:,}")
    print("-" * 30)
    print(f"Reports saved to: {stats_folder}")

if __name__ == "__main__":
    try:
        import PIL
    except ImportError:
        print("Please install Pillow: pip install Pillow")
        exit()
        
    calculate_dataset_statistics_complete()