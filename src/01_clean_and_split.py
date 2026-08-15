"""
==============================================================================
Script: 01_data_cleaning_and_split.py
Author: Moutaz Marei Algharyani
Description:
    Loads generated DeepMIMO hybrid dataset, splits it into 80% Train / 20% Test
    stratified by BS_ID, applies Global Z-Score Normalization using Training statistics,
    and exports clean CSVs WITH XYZ coordinates for spatial visualization.
==============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# DYNAMIC PATH SETUP (PORTABLE ACROSS ALL MACHINES)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure data folder exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "train_Dataset_Hybrid_8Ant.csv"
TRAIN_OUTPUT_FILE = DATA_DIR / "train_Dataset_Hybrid.csv"
TEST_OUTPUT_FILE  = DATA_DIR / "test_Dataset_Hybrid.csv"

# ==============================================================================
# BLOCK 2: LOAD DATA
# ==============================================================================
print("Loading dataset...")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"[ERROR] Input dataset not found at: {INPUT_FILE}\nPlease run 00_generate_deepmimo_dataset.py first.")

df_all = pd.read_csv(INPUT_FILE)

# Identify channel columns (Inputs)
H_cols = [c for c in df_all.columns if c.startswith('H_BS')]

# ==============================================================================
# BLOCK 3: STRATIFIED SPLIT (80% Train / 20% Test)
# ==============================================================================
print("Splitting data (Stratified by BS_ID)...")
df_train, df_test = train_test_split(
    df_all, test_size=0.20, random_state=42, stratify=df_all['BS_ID']
)

df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

# ==============================================================================
# BLOCK 4: GLOBAL NORMALIZATION (Z-SCORE)
# ==============================================================================
print("Applying global normalization to channel columns...")
H_train = df_train[H_cols].values.astype(np.float32)
H_test  = df_test[H_cols].values.astype(np.float32)

# Calculate mean and std strictly from the TRAINING set to prevent data leakage
global_mean = np.mean(H_train)
global_std  = np.std(H_train) + 1e-10

# Scale both Train and Test sets
df_train[H_cols] = (H_train - global_mean) / global_std
df_test[H_cols]  = (H_test - global_mean) / global_std

# ==============================================================================
# BLOCK 5: EXPORT DATA (RETAINING X, Y, Z FOR VISUALIZATION)
# ==============================================================================
print(f"Train columns: {df_train.shape[1]}")
print(f"Test columns: {df_test.shape[1]}")

print("Saving datasets...")
df_train.to_csv(TRAIN_OUTPUT_FILE, index=False)
df_test.to_csv(TEST_OUTPUT_FILE, index=False)

print(f"Done! Train: {len(df_train)} rows | Test: {len(df_test)} rows")
print(f"Files saved successfully in: {DATA_DIR}")
print("Files are 100% ready for spatial visualization and model training!")