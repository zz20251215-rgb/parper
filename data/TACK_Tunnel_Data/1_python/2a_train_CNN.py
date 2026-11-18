# -*- coding: utf-8 -*-
"""
2_train_CNN.py

This script trains a UNet segmentaiton model for a single detection class. 
The user defines the "Session_Name" which is the output folder for the saved 
model, plots and metrics. 
The user use the Global Configuration to adjust parameters. This includdes: 
 - A weight factor is included for imbalanced datasets.
- Data used for Traning, Evaluation and Testing is based on csv files.
 - The script creates masks used for the fastai packaage which use 1 for defect
 and 0 for background. The user defines the pixel value for the class they want 
to train the model for. 
 - Model training parameters are easily adjusted.
 - Output includes plots of top 5 best and worst predictions of cracks and 
 txt files with a summary of the metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback, CSVLogger
from fastai.losses import CrossEntropyLossFlat

##################################
# 1. GLOBAL CONFIGURATION
##################################

# --- 🏷️ SESSION SETTINGS ---
SESSION_NAME = "TA+TB+TC_TEST" 

# --- ⚖️ CLASS IMBALANCE SETTINGS ---
CRACK_CLASS_WEIGHT = 20.0 

# --- 📁 CSV LOCATION & SELECTION ---
TRAIN_CSVS = ['TA_train.csv','TB_train.csv','TC_train.csv']
VAL_CSVS   = ['TA_val.csv','TB_val.csv','TC_val.csv']
TEST_CSVS  = ['TA_test.csv','TB_test.csv','TC_test.csv'] 

# --- Directory Settings ---
BASE_DIR = os.getcwd()
DATA_ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../')) 
CSV_SOURCE_DIR = os.path.join(DATA_ROOT_DIR, '2_model_input/') 

ORIGINAL_MASK_DIR = os.path.join(DATA_ROOT_DIR, '3_mask')
SANITIZED_MASK_DIR = os.path.join(DATA_ROOT_DIR, '3_masks_sanitized')

OUTPUT_ROOT = os.path.join(DATA_ROOT_DIR, '5_model_output')
SESSION_DIR = os.path.join(OUTPUT_ROOT, SESSION_NAME)

TRAIN_DIR = os.path.join(SESSION_DIR, 'Training')
TRAIN_MODEL_DIR = os.path.join(TRAIN_DIR, 'Models')
TRAIN_PLOT_DIR = os.path.join(TRAIN_DIR, 'Plots')

TEST_DIR = os.path.join(SESSION_DIR, 'Testing')
TEST_PLOT_DIR = os.path.join(TEST_DIR, 'Plots')

# --- Sanitization Settings ---
ORIGINAL_CLASS_PIXEL_VALUE = 40 
SANITIZED_VALUE = 1               

# --- Model Training Settings ---
MODEL_ARCH = resnet34
IMG_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 2
LEARNING_RATE = 4e-4
WD = 1e-2

##################################
# 2. HELPER & METRIC FUNCTIONS
##################################

def check_system_resources():
    print("\n--- 🖥️ System Resource Check ---")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {device_name}")
    else:
        print("❌ GPU NOT Detected. Training will be slow.")
        
def enforce_dedicated_vram(fraction=0.95):
    """
    Configures PyTorch to strictly use only a specific fraction of Dedicated VRAM.
    If the model attempts to exceed this, it will throw an OOM error immediately
    rather than spilling into slow system RAM (Shared Memory).
    """
    if torch.cuda.is_available():
        # Empty cache to start fresh
        torch.cuda.empty_cache()
        
        # Enforce the limit. 
        # We use 0.95 (95%) to leave a tiny bit of room for the OS display overhead 
        # so the driver doesn't panic and swap to RAM.
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            print(f"🔒 STRICT MODE: GPU memory capped at {fraction*100}%.")
            print("   -> Script will CRASH if this limit is exceeded (preventing slow shared memory usage).")
        except RuntimeError as e:
            print(f"⚠️ Could not set memory fraction: {e}")
    else:
        print("⚠️ GPU not available. Running on CPU (slow).")

def get_expected_mask_basename(image_basename):
    parts = image_basename.rsplit('_', 1)
    if len(parts) == 2:
        base_name, tile_id = parts
        return f"{base_name}_fuse_{tile_id}_1band"
    return image_basename

# --- Metrics ---
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

# --- Weighted Loss ---
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

##################################
# 3. DATA PROCESSING FUNCTIONS
##################################

def sanitize_dataframe(df, desc="Sanitizing"):
    os.makedirs(SANITIZED_MASK_DIR, exist_ok=True)
    new_mask_paths = []
    image_abs_paths = []
    valid_indices = []
    
    found_count = 0
    missing_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            rel_path = row['filename']
            abs_img_path = os.path.normpath(os.path.join(BASE_DIR, rel_path))
            img_basename = os.path.splitext(os.path.basename(abs_img_path))[0]
            
            mask_basename_no_ext = get_expected_mask_basename(img_basename)
            mask_filename = f"{mask_basename_no_ext}.png"
            
            raw_mask_path = os.path.join(ORIGINAL_MASK_DIR, mask_filename)
            clean_mask_path = os.path.join(SANITIZED_MASK_DIR, mask_filename)

            if os.path.exists(clean_mask_path):
                image_abs_paths.append(abs_img_path)
                new_mask_paths.append(clean_mask_path)
                valid_indices.append(idx)
                found_count += 1
                continue

            if not os.path.exists(raw_mask_path):
                if missing_count < 3:
                    print(f"⚠️ Raw mask not found for: {os.path.basename(abs_img_path)}")
                missing_count += 1
                continue
                
            target_class = row.get('target', 0) 
            
            mask_arr = np.array(Image.open(raw_mask_path))
            if target_class == 1:
                new_mask = np.zeros_like(mask_arr, dtype=np.uint8)
                new_mask[mask_arr == ORIGINAL_CLASS_PIXEL_VALUE] = SANITIZED_VALUE
                Image.fromarray(new_mask).save(clean_mask_path)
            else:
                blank_mask = np.zeros_like(mask_arr, dtype=np.uint8)
                Image.fromarray(blank_mask).save(clean_mask_path)
            
            image_abs_paths.append(abs_img_path)
            new_mask_paths.append(clean_mask_path)
            valid_indices.append(idx)
            found_count += 1
            
        except Exception as e:
            print(f"Error on {os.path.basename(abs_img_path)}: {e}")

    clean_df = df.iloc[valid_indices].copy()
    clean_df['image_abs_path'] = image_abs_paths
    clean_df['mask_path_sanitized'] = new_mask_paths
    return clean_df

def combine_csvs(csv_list, is_valid_flag=False):
    dfs = []
    for f in csv_list:
        path = os.path.join(CSV_SOURCE_DIR, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"❌ Warning: CSV file not found: {path}")
    if not dfs: return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined['is_valid'] = is_valid_flag
    return combined

##################################
# 4. REPORTING & VISUALIZATION
##################################

def visualize_sanity_check(df, save_dir, n_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    crack_df = df[df['target'] == 1]
    if len(crack_df) == 0: return

    n = min(n_samples, len(crack_df))
    samples = crack_df.sample(n=n)
    
    fig, axs = plt.subplots(n, 3, figsize=(15, 5*n))
    if n == 1: axs = np.expand_dims(axs, 0)
    
    row_idx = 0
    for _, row in samples.iterrows():
        img = Image.open(row['image_abs_path'])
        mask = np.array(Image.open(row['mask_path_sanitized']))
        masked_overlay = np.ma.masked_where(mask == 0, mask)

        axs[row_idx, 0].imshow(img, cmap='gray')
        axs[row_idx, 0].set_title("Image")
        axs[row_idx, 0].axis('off')
        
        axs[row_idx, 1].imshow(mask, cmap='gray')
        axs[row_idx, 1].set_title("Sanitized Mask")
        axs[row_idx, 1].axis('off')
        
        axs[row_idx, 2].imshow(img, cmap='gray')
        axs[row_idx, 2].imshow(masked_overlay, cmap='autumn', alpha=0.6)
        axs[row_idx, 2].set_title("Overlay")
        axs[row_idx, 2].axis('off')
        row_idx += 1
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sanity_check_preview.png'))
    plt.close()

# --- NEW: Generate Comprehensive Training Report ---
def generate_training_report(log_path, output_dir):
    """
    Reads the CSV log, finds best epoch, writes a summary TXT file.
    Returns: Dictionary of the best validation metrics found.
    """
    if not os.path.exists(log_path):
        print("⚠️ No training log found to generate report.")
        return {}

    df = pd.read_csv(log_path)
    
    # Identify the best epoch based on IoU
    # (FastAI columns: epoch, train_loss, valid_loss, [metrics...])
    # We assume 'iou_crack' is in the columns
    
    if 'iou_crack' in df.columns:
        best_row = df.loc[df['iou_crack'].idxmax()]
        best_epoch = int(best_row['epoch'])
    else:
        # Fallback if iou_crack missing, use last epoch
        best_row = df.iloc[-1]
        best_epoch = int(best_row['epoch'])

    # Prepare stats for return
    best_metrics = best_row.to_dict()

    txt_path = os.path.join(output_dir, 'training_report.txt')
    
    with open(txt_path, 'w') as f:
        f.write("TRAINING SESSION SUMMARY\n")
        f.write("========================\n")
        f.write(f"Total Epochs Run: {len(df)}\n")
        f.write(f"Best Model Saved at Epoch: {best_epoch}\n")
        f.write("Note: 'valid_loss' and metrics below are evaluated on the VALIDATION dataset.\n\n")
        
        f.write(f"BEST VALIDATION METRICS (Epoch {best_epoch}):\n")
        f.write("-----------------------------------\n")
        for k, v in best_metrics.items():
            # Format numbers nicely
            val_str = f"{v:.6f}" if isinstance(v, (int, float)) else str(v)
            f.write(f"{k:<20}: {val_str}\n")
            
        f.write("\n\nFULL TRAINING HISTORY\n")
        f.write("=====================\n")
        # Write the full dataframe table
        f.write(df.to_string(index=False))
    
    print(f"✅ Training report with Best Epoch Summary saved to {txt_path}")
    return best_metrics

# --- Generate Comparison Testing Report ---
def generate_testing_report(test_metrics, val_metrics, output_dir):
    """
    Writes test metrics AND compares them to the best validation metrics.
    """
    txt_path = os.path.join(output_dir, 'testing_report.txt')
    
    with open(txt_path, 'w') as f:
        f.write("TESTING & COMPARISON REPORT\n")
        f.write("===========================\n\n")
        
        # Header for Table
        # Columns: Metric Name | Best Validation (Train Phase) | Testing Result
        f.write(f"{'METRIC':<25} | {'BEST VALIDATION':<18} | {'TESTING RESULT':<18}\n")
        f.write("-" * 65 + "\n")
        
        for k, test_val in test_metrics.items():
            # Try to find matching key in validation stats
            # Note: FastAI training log might call it 'iou_crack' 
            # while validate() output might call it 'iou_crack' (should match)
            
            val_val = val_metrics.get(k, "N/A")
            
            # Formatting
            if isinstance(test_val, float): test_str = f"{test_val:.6f}"
            else: test_str = str(test_val)
            
            if isinstance(val_val, float): val_str = f"{val_val:.6f}"
            else: val_str = str(val_val)
            
            f.write(f"{k:<25} | {val_str:<18} | {test_str:<18}\n")
            
    print(f"✅ Testing report (with Training comparison) saved to {txt_path}")

def save_best_worst_predictions(learn, dl, save_dir, k=5):
    print(f"\n--- 📸 Generating Best/Worst {k} Predictions (CRACKS ONLY) ---")
    os.makedirs(save_dir, exist_ok=True)
    
    preds, targs = learn.get_preds(dl=dl)
    pred_masks = preds.argmax(dim=1)
    
    results = []
    for i in range(len(pred_masks)):
        p = pred_masks[i]
        t = targs[i]
        inter = ((p==1) & (t==1)).sum().item()
        union = ((p==1) | (t==1)).sum().item()
        if union == 0: iou_val = 1.0 
        else: iou_val = inter / (union + 1e-6)
        has_crack_in_gt = (t == 1).sum().item() > 0
        results.append({'idx': i, 'iou': iou_val, 'has_crack': has_crack_in_gt})
        
    res_df = pd.DataFrame(results)
    crack_candidates = res_df[res_df['has_crack'] == True].copy()
    
    if len(crack_candidates) == 0:
        print("⚠️ No images with cracks found. Skipping plots.")
        return

    print(f"    -> Found {len(crack_candidates)} images with cracks.")
    sorted_df = crack_candidates.sort_values(by='iou', ascending=True)
    worst_df = sorted_df.head(k)
    best_df = sorted_df.tail(k).iloc[::-1] 

    def plot_batch(df_rows, label_type):
        for rank, (_, row_data) in enumerate(df_rows.iterrows()):
            idx = int(row_data['idx'])
            row_item = dl.dataset.items.iloc[idx]
            img = Image.open(row_item['image_abs_path'])
            gt_mask = targs[idx]
            pred_mask = pred_masks[idx]
            iou_val = row_data['iou']
            
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img, cmap='gray'); ax[0].set_title(f"Input Image"); ax[0].axis('off')
            ax[1].imshow(gt_mask.cpu(), cmap='gray'); ax[1].set_title("Ground Truth"); ax[1].axis('off')
            ax[2].imshow(pred_mask.cpu(), cmap='gray'); ax[2].set_title(f"Pred (IoU: {iou_val:.4f})"); ax[2].axis('off')
            
            filename = os.path.basename(row_item['image_abs_path'])
            plt.suptitle(f"{label_type} #{rank+1} - {filename}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{label_type}_{rank+1}_iou_{iou_val:.2f}.png"))
            plt.close()

    plot_batch(best_df, "BEST_CRACK")
    plot_batch(worst_df, "WORST_CRACK")
    print(f"✅ Plots saved to {save_dir}")

##################################
# 5. MAIN PIPELINE
##################################

def get_metric_name(metric):
    if hasattr(metric, '__name__'): return metric.__name__
    if hasattr(metric, 'func'): return metric.func.__name__
    return str(metric)

def run_pipeline():
    check_system_resources()
    
    for d in [TRAIN_DIR, TRAIN_MODEL_DIR, TRAIN_PLOT_DIR, TEST_DIR, TEST_PLOT_DIR]:
        os.makedirs(d, exist_ok=True)
    
    print(f"--- Session: {SESSION_NAME} ---")

    # ==========================
    # PHASE 1: PREPARE TRAINING DATA
    # ==========================
    print("\n--- 🔄 Loading Training Data ---")
    df_train = combine_csvs(TRAIN_CSVS, is_valid_flag=False)
    df_val = combine_csvs(VAL_CSVS, is_valid_flag=True)
    
    if len(df_train) == 0: raise ValueError("No training data found.")
    full_df = pd.concat([df_train, df_val], ignore_index=True)
    
    df_ready = sanitize_dataframe(full_df, desc="Sanitizing Train/Val")
    visualize_sanity_check(df_ready, TRAIN_PLOT_DIR)

    codes = np.array(['background', 'crack'])
    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes)),
        get_x=ColReader('image_abs_path'),
        get_y=ColReader('mask_path_sanitized'),
        splitter=ColSplitter('is_valid'),
        batch_tfms=[
            *aug_transforms(flip_vert=True, max_rotate=15.0, max_zoom=1.1, max_lighting=0.2),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    dls = dblock.dataloaders(
        df_ready, 
        bs=BATCH_SIZE, 
        num_workers=0,      
        pin_memory=True     
                             
                             )

    # ==========================
    # PHASE 2: TRAINING
    # ==========================
    print(f"\n--- 🏋️ Starting Training ({NUM_EPOCHS} epochs) ---")
    history_log_path = os.path.join(TRAIN_DIR, 'training_log.csv')
    
    learn = unet_learner(
        dls, MODEL_ARCH, 
        loss_func=WeightedCombinedLoss(crack_weight=CRACK_CLASS_WEIGHT),
        metrics=[dice_score_crack, iou_crack, recall_crack, precision_crack, f1_score_crack],
        wd=WD,
        model_dir=TRAIN_MODEL_DIR
    ).to_fp16()
    
    callbacks = [
        SaveModelCallback(monitor='iou_crack', comp=np.greater, fname='best_model'),
        CSVLogger(fname=history_log_path) 
    ]
    
    learn.fit_one_cycle(NUM_EPOCHS, slice(LEARNING_RATE), cbs=callbacks)
    
    # --- UPDATED: Generate Enhanced Training Report ---
    # This now captures the best metrics to pass to Phase 3
    best_val_metrics = generate_training_report(history_log_path, TRAIN_DIR)
    
    print("\n--- 🔍 Evaluating Validation Set ---")
    save_best_worst_predictions(learn, dls.valid, TRAIN_PLOT_DIR, k=5)

    # ==========================
    # PHASE 3: TESTING
    # ==========================
    print("\n--- 🧪 Starting Testing Phase ---")
    
    df_test_raw = combine_csvs(TEST_CSVS)
    if len(df_test_raw) > 0:
        df_test_ready = sanitize_dataframe(df_test_raw, desc="Sanitizing Test Set")
        
        if len(df_test_ready) > 0:
            test_dl = learn.dls.test_dl(df_test_ready, with_labels=True)
            
            print("Loading best model for testing...")
            learn.load('best_model')
            
            results = learn.validate(dl=test_dl)
            metric_names = ['valid_loss'] + [get_metric_name(m) for m in learn.metrics]
            # FastAI calls validation loss 'valid_loss' in validate(), but we can just label it Loss
            
            test_metrics_dict = dict(zip(metric_names, results))
            
            print("\nTest Results:")
            for k, v in test_metrics_dict.items():
                print(f"  {k}: {v:.6f}")
            
            # --- UPDATED: Generate Comparison Report ---
            generate_testing_report(test_metrics_dict, best_val_metrics, TEST_DIR)
            
            save_best_worst_predictions(learn, test_dl, TEST_PLOT_DIR, k=5)
            
        else:
            print("⚠️ Test dataframe empty after sanitization.")
    else:
        print("⚠️ No test CSVs found or they are empty.")

    print(f"\n✅ Pipeline Complete. Output saved to: {SESSION_DIR}")

if __name__ == "__main__":
    # 1. Enforce the Lock
    enforce_dedicated_vram(fraction=0.90) # Set to 90% to be safe, or 0.95 to push limits
    
    # 2. Run Pipeline with Error Catching
    try:
        run_pipeline()
        
    except torch.cuda.OutOfMemoryError:
        # This is the specific error we want to catch
        print("\n" + "="*60)
        print("🛑 CRITICAL ERROR: OUT OF DEDICATED GPU MEMORY")
        print("="*60)
        print("The script was stopped because it filled up the Dedicated VRAM.")
        print("We prevented it from using Shared Memory (RAM) to maintain performance.")
        print("SUGGESTIONS:")
        print("1. Decrease BATCH_SIZE (currently set to {})".format(BATCH_SIZE))
        print("2. Decrease IMG_SIZE (currently set to {})".format(IMG_SIZE))
        print("3. Use a smaller model architecture (e.g., resnet18)")
        sys.exit(1)
        
    except Exception as e:
        # Catch other standard errors
        print(f"\n❌ An unexpected error occurred: {e}")
        # print(traceback.format_exc()) # Uncomment if you want full traceback