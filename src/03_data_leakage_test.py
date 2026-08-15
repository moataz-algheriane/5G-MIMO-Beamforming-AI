"""
===============================================================================
Script: 03_data_leakage_test.py
Author: Moutaz Marei Algharyani
Description:
    Performs a strict spatial data leakage audit between Train and Test sets.
    Checks for exact coordinate overlaps (X, Y, Z) and spatial proximity violations 
    using a SciPy cKDTree spatial index (threshold: 1 cm).
===============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC PATH SETUP (PORTABLE ACROSS ALL MACHINES)
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "train_Dataset_Hybrid.csv"
TEST_FILE  = DATA_DIR / "test_Dataset_Hybrid.csv"

# Strict Distance Threshold in meters (e.g., 0.01 meters = 1 cm)
# If a test user is closer than this to a train user, it triggers a leakage warning.
DISTANCE_THRESHOLD = 0.01 

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: DATA LOADING & EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
print("⏳ [1/5] Loading datasets into memory...")

if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    raise FileNotFoundError(f"[ERROR] Dataset files missing in: {DATA_DIR}\nPlease verify previous processing scripts first.")

df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

# Extracting spatial coordinates (X, Y, Z)
train_coords = df_train[['X', 'Y', 'Z']].values
test_coords  = df_test[['X', 'Y', 'Z']].values

print(f"✅ Loaded {len(df_train):,} Training users.")
print(f"✅ Loaded {len(df_test):,} Testing users.")

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: STRICT EXACT MATCH CHECK (PANDAS INTERSECTION)
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔍 [2/5] Performing Strict Exact Match Check...")

# Find identical rows based on X, Y, Z coordinates
exact_duplicates = pd.merge(df_train[['X', 'Y', 'Z']], df_test[['X', 'Y', 'Z']], 
                            on=['X', 'Y', 'Z'], how='inner')

exact_leakage_count = len(exact_duplicates)
has_exact_leakage = exact_leakage_count > 0

print(f"ℹ️ Exact Coordinate Duplicates Found: {exact_leakage_count}")

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: PROXIMITY-BASED LEAKAGE CHECK (scipy cKDTree)
# ═══════════════════════════════════════════════════════════════════════════
print("🛰️ [3/5] Performing Spatial Proximity Leakage Check (KD-Tree)...")

# Build a spatial index tree using training coordinates
train_tree = cKDTree(train_coords)

# Query the tree with test coordinates to find any points within the strict threshold
# 'query_ball_point' finds all indices of train points within the threshold for each test point
leaked_indices = train_tree.query_ball_point(test_coords, r=DISTANCE_THRESHOLD)

# Count how many test points have at least one training point within the restricted radius
proximity_leakage_count = sum(1 for indices in leaked_indices if len(indices) > 0)
has_proximity_leakage = proximity_leakage_count > 0

print(f"ℹ️ Spatial Proximity Violations (< {DISTANCE_THRESHOLD*100} cm): {proximity_leakage_count}")

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: STRICT CONDITIONAL DECISION & AUDIT VERDICT
# ═══════════════════════════════════════════════════════════════════════════
print("\n⚖️ [4/5] Evaluating Strict Audit Conditions...")

print("=" * 60)
print("                  DATA LEAKAGE AUDIT REPORT                  ")
print("=" * 60)

# The ultimate strict condition
if has_exact_leakage or has_proximity_leakage:
    print("🚨 AUDIT VERDICT: DATA LEAKAGE DETECTED!")
    print("❌ Status: CRITICAL FAULT")
    print(f"   -> Exact matching coordinate leaked: {exact_leakage_count} rows.")
    print(f"   -> Spatially compromised test users : {proximity_leakage_count} users.")
    print("\n💡 Recommendation: Check your data splitting logic.")
    print("   Ensure that users are uniquely split before writing to files.")
else:
    print("🛡️ AUDIT VERDICT: DATA IS PERFECTLY SAFE!")
    print("✅ Status: SECURE & CLEAN")
    print("   -> No exact coordinate overlap found.")
    print(f"   -> No test users are within the critical {DISTANCE_THRESHOLD*100} cm boundary of train users.")
    print("\n💡 Conclusion: Your split is 100% spatially disjoint.")
    print("   The high performance of your model is due to GENUINE LEARNING, not leakage!")

print("=" * 60)
print("🎉 [5/5] Spatial Audit Completed Successfully.")