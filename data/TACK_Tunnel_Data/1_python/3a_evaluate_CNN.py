# -*- coding: utf-8 -*-
# Document info
__author__ = 'Andreas Sjölander, Gemini'
__version__ = ['1.0']
__version_date__ = '2025-11-25'
__maintainer__ = 'Andreas Sjölander'
__email__ = 'asjola@kth.se'

"""
3_evaluate_CNN.py
This script loads a pre-trained model and evaluate its performance on a list 
of datasets. The output is a .txt file with metrics. Naming of the file is based on
the SESSION_NAME and metrics for each eavluation is added in the txt file in
sequence, i.e. the metrics for all evaluation using the same model is stored in
the same file.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from fastai.vision.all import *
from fastai.losses import CrossEntropyLossFlat
from datetime import datetime  

# --- CONFIGURATION ---
SESSION_NAME = "TA+TC" 
TEST_CSVS  = ['TB_train.csv', 'TB_val.csv'] 

BASE_DIR = os.getcwd()
DATA_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../')) 
CSV_SOURCE_DIR = os.path.join(DATA_ROOT_DIR, '2_model_input/') 
ORIGINAL_MASK_DIR = os.path.join(DATA_ROOT_DIR, '3_mask')
SANITIZED_MASK_DIR = os.path.join(DATA_ROOT_DIR, '3_masks_sanitized')

OUTPUT_ROOT = os.path.join(DATA_ROOT_DIR, '5_model_output')
SESSION_DIR = os.path.join(OUTPUT_ROOT, SESSION_NAME)
TRAIN_MODEL_DIR = os.path.join(SESSION_DIR, 'Training', 'Models')
MODEL_WEIGHTS_PATH = os.path.join(TRAIN_MODEL_DIR, 'best_model.pth')

TEST_DIR = os.path.join(OUTPUT_ROOT, 'Testing')


# --- MODEL SETTINGS ---
ORIGINAL_CLASS_PIXEL_VALUE = 40 
SANITIZED_VALUE = 1               
MODEL_ARCH = resnet34
BATCH_SIZE = 8
CRACK_CLASS_WEIGHT = 20.0 

# --- DEFINITIONS (REQUIRED FOR LOADING) ---
def get_expected_mask_basename(image_basename):
    parts = image_basename.rsplit('_', 1)
    if len(parts) == 2:
        base_name, tile_id = parts
        return f"{base_name}_fuse_{tile_id}_1band"
    return image_basename

def _get_stats(inp, targ, class_idx=1, smooth=1e-6):
    pred = inp.argmax(dim=1)
    targ = targ.squeeze(1)
    tp = ((pred == class_idx) & (targ == class_idx)).sum().float()
    fp = ((pred == class_idx) & (targ != class_idx)).sum().float()
    fn = ((pred != class_idx) & (targ == class_idx)).sum().float()
    tn = ((pred != class_idx) & (targ != class_idx)).sum().float()
    return tp, fp, fn, tn, smooth

def iou_crack(inp, targ):
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fp + fn + smooth)

def dice_score_crack(inp, targ):
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

def recall_crack(inp, targ):
    tp, _, fn, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fn + smooth)

def precision_crack(inp, targ):
    tp, fp, _, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fp + smooth)

def f1_score_crack(inp, targ):
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return 2 * (precision * recall) / (precision + recall + smooth)

class WeightedCombinedLoss(nn.Module):
    def __init__(self, crack_weight=CRACK_CLASS_WEIGHT, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight, self.ce_weight = dice_weight, ce_weight
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor([1.0, crack_weight]).to(device)
        self.ce = CrossEntropyLossFlat(axis=1, weight=class_weights)
        self.dice = DiceLoss(axis=1) 
    def forward(self, inp, targ):
        ce_loss = self.ce(inp, targ.long())
        dice_loss = self.dice(inp, targ) 
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)

# --- DATA HELPERS ---
def sanitize_dataframe(df):
    os.makedirs(SANITIZED_MASK_DIR, exist_ok=True)
    new_mask_paths = []
    image_abs_paths = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sanitizing"):
        try:
            rel_path = row['filename']
            abs_img_path = os.path.normpath(os.path.join(BASE_DIR, rel_path))
            img_basename = os.path.splitext(os.path.basename(abs_img_path))[0]
            mask_basename_no_ext = get_expected_mask_basename(img_basename)
            mask_filename = f"{mask_basename_no_ext}.png"
            raw_mask_path = os.path.join(ORIGINAL_MASK_DIR, mask_filename)
            clean_mask_path = os.path.join(SANITIZED_MASK_DIR, mask_filename)

            if os.path.exists(clean_mask_path):
                image_abs_paths.append(abs_img_path); new_mask_paths.append(clean_mask_path); valid_indices.append(idx)
                continue
            if os.path.exists(raw_mask_path):
                target_class = row.get('target', 0)
                mask_arr = np.array(Image.open(raw_mask_path))
                if target_class == 1:
                    new_mask = np.zeros_like(mask_arr, dtype=np.uint8)
                    new_mask[mask_arr == ORIGINAL_CLASS_PIXEL_VALUE] = SANITIZED_VALUE
                    Image.fromarray(new_mask).save(clean_mask_path)
                else:
                    Image.fromarray(np.zeros_like(mask_arr, dtype=np.uint8)).save(clean_mask_path)
                image_abs_paths.append(abs_img_path); new_mask_paths.append(clean_mask_path); valid_indices.append(idx)
        except: pass
    clean_df = df.iloc[valid_indices].copy()
    clean_df['image_abs_path'] = image_abs_paths
    clean_df['mask_path_sanitized'] = new_mask_paths
    return clean_df

def combine_csvs(csv_list):
    dfs = []
    for f in csv_list:
        path = os.path.join(CSV_SOURCE_DIR, f)
        if os.path.exists(path): dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_metric_label(m):
    if hasattr(m, 'name'): return m.name
    if hasattr(m, 'func') and hasattr(m.func, '__name__'): return m.func.__name__
    return str(m)

# --- MAIN ---
def run():
    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"--- 🧪 Evaluation Session: {SESSION_NAME} ---")

    # 1. Data
    df_test = sanitize_dataframe(combine_csvs(TEST_CSVS))
    if len(df_test) == 0: return print("❌ No test data found.")
    
    # 2. Setup Learner
    codes = np.array(['background', 'crack'])
    dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                       get_x=ColReader('image_abs_path'), get_y=ColReader('mask_path_sanitized'),
                       batch_tfms=[Normalize.from_stats(*imagenet_stats)])
    dls = dblock.dataloaders(df_test, bs=BATCH_SIZE, num_workers=0) # Windows Fix

    print("🔄 Reconstructing Model...")
    learn = unet_learner(dls, MODEL_ARCH, loss_func=WeightedCombinedLoss(),
                         metrics=[dice_score_crack, iou_crack, recall_crack, precision_crack, f1_score_crack],
                         model_dir=TRAIN_MODEL_DIR)
    
    # 3. Load & Eval
    print(f"📂 Loading: {MODEL_WEIGHTS_PATH}")
    learn.load('best_model')
    
    print("📉 Running Validation...")
    results = learn.validate(dl=dls.test_dl(df_test, with_labels=True))
    
    metric_labels = ['valid_loss'] + [get_metric_label(m) for m in learn.metrics]
    print("\n📊 RESULTS:")
    
    output_path = os.path.join(TEST_DIR, SESSION_NAME+'_testing_score.txt')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'a') as f:
        # Header for this specific run
        f.write(f"\n{'='*40}\n")
        f.write(f"Date: {current_time}\n")
        f.write(f"Model Name: {SESSION_NAME}\n")
        f.write(f"Test CSVs: {', '.join(TEST_CSVS)}\n")
        f.write(f"{'-'*40}\n")
        
        for name, val in zip(metric_labels, results):
            print(f"{name:<25}: {val:.6f}")
            f.write(f"{name:<25}: {val:.6f}\n")
            
    print(f"📝 Results appended to: {output_path}")


if __name__ == "__main__":
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    run()