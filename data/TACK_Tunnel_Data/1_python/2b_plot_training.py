# -*- coding: utf-8 -*-
# Document info
__author__ = 'Andreas Sjölander, Gemini'
__version__ = ['1.0']
__version_date__ = '2025-11-25'
__maintainer__ = 'Andreas Sjölander'
__email__ = 'asjola@kth.se'

"""
2b_plot_training.py
This script reads the csv output from training and creates a plot of training 
and validation loss in one plot and IoU and F1-score in a second plot. 
User only needs to change "TRAINING_DATA" to correct training set.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 🎨 CONFIGURATION & AESTHETICS SECTION
# ==========================================
# Change these variables to customize the look of your plots

# --- File Settings ---
TRAINING_DATA = 'TA' # Options: 'TA', 'TB', 'TC', etc.

# Dynamic path setup
script_location = os.path.dirname(os.path.abspath(__file__))
# Assuming the script is inside a subfolder, go up one level to root
root_dir = os.path.dirname(script_location) 

# Input Folder
file_folder = os.path.join(root_dir, '5_model_output', TRAINING_DATA, 'Training')
FILE_PATH = os.path.join(file_folder, 'training_log.csv')

# Output Folder for Plots
PLOT_OUTPUT_DIR = os.path.join(root_dir, '5_model_output', 'Plots')

# --- General Plot Settings ---
FIG_SIZE = (6, 6)      # Width, Height in inches
DPI = 100               # Resolution
USE_GRID = True         # Show grid lines?
GRID_STYLE = '--'       # Grid line style
GRID_ALPHA = 0.5        # Grid transparency

# --- Font Sizes ---
FONT_TITLE = 16
FONT_AXIS_LABEL = 14
FONT_LEGEND = 12
FONT_TICKS = 12

# --- Colors & Line Styles ---
# You can use color names ('red', 'blue') or Hex codes ('#FF5733')
LINE_WIDTH = 2.5

# Plot 1: Loss Configuration
COLOR_TRAIN_LOSS = '#1f77b4'  
COLOR_VALID_LOSS = '#ff7f0e'  
TITLE_LOSS = f"Training vs Validation Loss ({TRAINING_DATA})"
Y_LABEL_LOSS = "Loss Value"

# Plot 2: Metrics Configuration
COLOR_IOU = '#2ca02c'         
COLOR_F1 = '#d62728'          
TITLE_METRICS = f"IoU and F1 Score over Epochs ({TRAINING_DATA})"
Y_LABEL_METRICS = "Score"

# ==========================================
# 🚀 MAIN SCRIPT LOGIC
# ==========================================

def plot_training_results():
    # 1. Check if input file exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: The file '{FILE_PATH}' was not found.")
        print(f"Constructed path: {FILE_PATH}")
        print("Please check the 'TRAINING_DATA' variable or folder structure.")
        return

    # 2. Create Output Directory if it doesn't exist
    if not os.path.exists(PLOT_OUTPUT_DIR):
        try:
            os.makedirs(PLOT_OUTPUT_DIR)
            print(f"Created output directory: {PLOT_OUTPUT_DIR}")
        except OSError as e:
            print(f"Error creating directory {PLOT_OUTPUT_DIR}: {e}")
            return

    # 3. Read the CSV file
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Successfully loaded data for {TRAINING_DATA}. Found {len(df)} epochs.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Apply global font sizes using rcParams
    plt.rcParams.update({
        'font.size': FONT_TICKS,
        'axes.titlesize': FONT_TITLE,
        'axes.labelsize': FONT_AXIS_LABEL,
        'legend.fontsize': FONT_LEGEND,
        'xtick.labelsize': FONT_TICKS,
        'ytick.labelsize': FONT_TICKS
    })

    # -------------------------------------------------------
    # PLOT 1: Training and Validation Loss
    # -------------------------------------------------------
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    plt.plot(df['epoch'], df['train_loss'], 
             label='Training Loss', 
             color=COLOR_TRAIN_LOSS, 
             linewidth=LINE_WIDTH)
    
    plt.plot(df['epoch'], df['valid_loss'], 
             label='Validation Loss', 
             color=COLOR_VALID_LOSS, 
             linewidth=LINE_WIDTH, 
             linestyle='--') 

    plt.title(TITLE_LOSS, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel(Y_LABEL_LOSS)
    plt.legend()
    
    if USE_GRID:
        plt.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    
    plt.tight_layout()
    
    # Save Plot 1
    save_name_loss = f"{TRAINING_DATA}_loss_plot.png"
    save_path_loss = os.path.join(PLOT_OUTPUT_DIR, save_name_loss)
    plt.savefig(save_path_loss)
    print(f"Saved Loss plot to: {save_path_loss}")

    # -------------------------------------------------------
    # PLOT 2: IoU and F1 Score
    # -------------------------------------------------------
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    plt.plot(df['epoch'], df['iou_crack'], 
             label='IoU (Crack)', 
             color=COLOR_IOU, 
             linewidth=LINE_WIDTH)
    
    plt.plot(df['epoch'], df['f1_score_crack'], 
             label='F1 Score (Crack)', 
             color=COLOR_F1, 
             linewidth=LINE_WIDTH)

    plt.title(TITLE_METRICS, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel(Y_LABEL_METRICS)
    plt.legend()

    if USE_GRID:
        plt.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)

    plt.tight_layout()

    # Save Plot 2
    save_name_metrics = f"{TRAINING_DATA}_metrics_plot.png"
    save_path_metrics = os.path.join(PLOT_OUTPUT_DIR, save_name_metrics)
    plt.savefig(save_path_metrics)
    print(f"Saved Metrics plot to: {save_path_metrics}")

    # Show the plots
    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
    plot_training_results()