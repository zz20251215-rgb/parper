# -*- coding: utf-8 -*-
# Document info
__author__ = 'Andreas Sjölander, Gemini'
__version__ = ['1.0'] 
__version_date__ = '2025-12-01'
__maintainer__ = 'Andreas Sjölander'
__email__ = 'asjola@kth.se'

"""
1b_histogram_plot.py
This script reads the segmented masks and plots histograms of the defect size 
distribution. It generates:
1. Individual plots for all datasets and individual plots for the tunnels 
TA, TB, TC.
2. A combined subplot figure comparing TA, TB, and TC.

The user chose which defect that should be plotted.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. SELECT DEFECT TO PLOT
# Options: 'Crack', 'Water', 'Leaching'
DEFECT_TO_PLOT = 'Crack' 

# 2. CLASS DEFINITIONS (Pixel Values)
CLASS_MAP = {
    'Crack': 40,
    'Water': 160,
    'Leaching': 200
}

# ------------------------------------------
# 3. FONT CONFIGURATION (Global)
# ------------------------------------------
FONT_PARAMS = {
    'suptitle': 18,  # Main title for the combined subplot figure
    'title':    16,  # Title of individual plots
    'label':    14,  # X and Y axis labels (e.g. "Frequency")
    'ticks':    14,  # Numbers on the axes
    'legend':   14,  # Legend text size
}

# ------------------------------------------
# 4. SETTINGS: INDIVIDUAL PLOTS (One file per tunnel)
# ------------------------------------------
INDIV_X_AXIS_MAX = 15000   # Max pixel area on X-axis
INDIV_Y_AXIS_MAX = 50      # Max frequency on Y-axis
INDIV_BIN_SIZE = 250       # Bin width for single plots

# ------------------------------------------
# 5. SETTINGS: SUBPLOT FIGURE (TA, TB, TC combined)
# ------------------------------------------
SUBPLOT_X_AXIS_MAX = 15000 # Max pixel area on X-axis
SUBPLOT_Y_AXIS_MAX = 70    # Max frequency on Y-axis
SUBPLOT_BIN_SIZE = 400     # Bin width for the comparison plot

# ==========================================
#               MAIN SCRIPT
# ==========================================

def run_histogram_analysis():
    # --- 1. Setup Paths ---
    script_location = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_location)
    
    mask_folder = os.path.join(root_dir, '3_mask')
    output_dir = os.path.join(root_dir, '2_statistics')
    
    # Create output sub-folder for plots to keep it tidy
    plot_output_dir = os.path.join(output_dir, 'Plots')
    os.makedirs(plot_output_dir, exist_ok=True)

    # Get target pixel value
    if DEFECT_TO_PLOT not in CLASS_MAP:
        print(f"Error: {DEFECT_TO_PLOT} is not in CLASS_MAP. Choose: {list(CLASS_MAP.keys())}")
        return
    
    target_value = CLASS_MAP[DEFECT_TO_PLOT]
    print(f"--- Configuration ---")
    print(f"Target Defect: {DEFECT_TO_PLOT} (Pixel Value: {target_value})")
    print(f"Source:        {mask_folder}")
    print(f"Output:        {plot_output_dir}")
    print("-" * 30)

    # --- 2. Data Collection ---
    data_buckets = {
        'Total': [],
        'TA': [],
        'TB': [],
        'TC': []
    }

    # Get files
    valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    files = []
    for ext in valid_exts:
        files.extend(glob(os.path.join(mask_folder, ext)))
    
    if not files:
        print("No mask files found.")
        return

    print("Reading masks and extracting defect sizes...")
    
    for filepath in tqdm(files, unit="mask"):
        # Read image using OpenCV (Grayscale)
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue

        # Count pixels matching the target value
        defect_pixels = np.sum(mask == target_value)

        if defect_pixels > 0:
            filename = os.path.basename(filepath)
            
            # Add to Total
            data_buckets['Total'].append(defect_pixels)
            
            # Add to Sub-dataset buckets
            if filename.startswith('TA'):
                data_buckets['TA'].append(defect_pixels)
            elif filename.startswith('TB'):
                data_buckets['TB'].append(defect_pixels)
            elif filename.startswith('TC'):
                data_buckets['TC'].append(defect_pixels)

    print("-" * 30)
    
    # --- 3. Plotting Loop (Individual) ---
    print("Generating Individual Plots...")
    for dataset_name, values in data_buckets.items():
        if not values:
            print(f"Skipping {dataset_name}: No defects found.")
            continue
            
        plot_single_histogram(
            data_values=values,
            dataset_name=dataset_name,
            defect_type=DEFECT_TO_PLOT,
            output_dir=plot_output_dir,
            x_max=INDIV_X_AXIS_MAX,
            y_max=INDIV_Y_AXIS_MAX,
            bin_size=INDIV_BIN_SIZE
        )

    # --- 4. Plotting Subplots (Combined) ---
    print("Generating Comparison Subplots...")
    plot_comparison_figure(
        data_buckets=data_buckets,
        defect_type=DEFECT_TO_PLOT,
        output_dir=plot_output_dir,
        x_max=SUBPLOT_X_AXIS_MAX,
        y_max=SUBPLOT_Y_AXIS_MAX,
        bin_size=SUBPLOT_BIN_SIZE
    )

    print("\nProcessing Complete.")

def plot_single_histogram(data_values, dataset_name, defect_type, output_dir, x_max, y_max, bin_size):
    """
    Generates and saves a single histogram.
    """
    # Statistics
    mean_val = np.mean(data_values)
    median_val = np.median(data_values)
    max_val = np.max(data_values)
    
    plt.figure(figsize=(8, 6))

    # --- Bin Calculation ---
    upper_limit = x_max if x_max else max_val
    bins = np.arange(0, upper_limit + bin_size, bin_size)

    # --- Plotting ---
    plt.hist(data_values, bins=bins, color='#1f77b4', edgecolor='black', alpha=0.7)

    # Lines for Mean/Median
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.0f}')

    # Labels & Fonts
    plt.title(f'{defect_type} Size Distribution: {dataset_name}', 
              fontsize=FONT_PARAMS['title'], fontweight='bold')
    plt.xlabel(f'Defect Area (Pixels)', fontsize=FONT_PARAMS['label'])
    plt.ylabel('Frequency (Count)', fontsize=FONT_PARAMS['label'])
    
    plt.xticks(fontsize=FONT_PARAMS['ticks'])
    plt.yticks(fontsize=FONT_PARAMS['ticks'])
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.legend(fontsize=FONT_PARAMS['legend'])

    # Axis Limits
    if x_max:
        plt.xlim(0, x_max)
    if y_max:
        plt.ylim(0, y_max)

    plt.tight_layout()

    # --- Save ---
    filename = f"Hist_{defect_type}_{dataset_name}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    # Save Stats Text
    txt_filename = f"Stats_{defect_type}_{dataset_name}.txt"
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write(f"Dataset: {dataset_name}\nDefect: {defect_type}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMax: {max_val}\n")

def plot_comparison_figure(data_buckets, defect_type, output_dir, x_max, y_max, bin_size):
    """
    Generates a 1x3 subplot figure comparing TA, TB, and TC.
    """
    tunnels = ['TA', 'TB', 'TC']
    
    # Setup Figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Use 'suptitle' from FONT_PARAMS
    fig.suptitle(f'{defect_type} Distribution Comparison (Bin Size: {bin_size}px)', 
                 fontsize=FONT_PARAMS['suptitle'], fontweight='bold')

    for ax, tunnel in zip(axes, tunnels):
        data = data_buckets.get(tunnel, [])
        
        # Handle empty data
        if not data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, 
                    fontsize=FONT_PARAMS['label'])
            ax.set_title(f"Tunnel {tunnel}", fontsize=FONT_PARAMS['title'])
            continue

        # Stats
        mean_val = np.mean(data)
        median_val = np.median(data)
        max_val_local = np.max(data)

        # Bins
        upper_limit = x_max if x_max else max_val_local
        bins = np.arange(0, upper_limit + bin_size, bin_size)

        # Plot
        ax.hist(data, bins=bins, color='Steelblue', edgecolor='black', alpha=0.7)
        
        # Lines
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
        ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.0f}')

        # Formatting with Fonts
        ax.set_title(f"Tunnel {tunnel} (n={len(data)})", fontsize=FONT_PARAMS['title'])
        ax.set_xlabel('Defect Area (Pixels)', fontsize=FONT_PARAMS['label'])
        
        # Adjust Ticks
        ax.tick_params(axis='both', which='major', labelsize=FONT_PARAMS['ticks'])
        
        ax.grid(axis='y', alpha=0.5, linestyle='--')
        ax.legend(fontsize=FONT_PARAMS['legend'], loc='upper right')

        # Axis Limits
        if x_max:
            ax.set_xlim(0, x_max)
        if y_max:
            ax.set_ylim(0, y_max)

    # Set Y-label only on the first plot
    axes[0].set_ylabel('Frequency (Count)', fontsize=FONT_PARAMS['label'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make space for suptitle
    
    # Save
    filename = f"Hist_Comparison_{defect_type}_TA_TB_TC.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {filename}")

if __name__ == "__main__":
    run_histogram_analysis()