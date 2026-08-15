"""
===============================================================================
Script: 02_visualize_user_layout.py
Author: Moutaz Marei Algharyani
Description:
    Generates a high-fidelity 2D spatial map comparing Train vs Test user 
    coordinate distributions (X, Y in meters) to visually inspect layout coverage
    and ensure spatial representation across split datasets.
===============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC PATH SETUP (PORTABLE ACROSS ALL MACHINES)
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_Dataset_Hybrid.csv"
TEST_FILE  = DATA_DIR / "test_Dataset_Hybrid.csv"
OUTPUT_MAP = DOCS_DIR / "Hybrid_Train_Test_Spatial_Map.png"

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
print("⏳ [1/4] Loading Datasets...")

if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    raise FileNotFoundError(f"[ERROR] Dataset files not found in: {DATA_DIR}\nPlease verify previous processing steps.")

df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

print(f"✅ Loaded Train Data: {len(df_train):,} users.")
print(f"✅ Loaded Test Data : {len(df_test):,} users.")

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: 2D PLOT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════
print("🎨 [2/4] Initializing 2D Spatial Map...")

# Creating a high-fidelity figure for sharp rendering
fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: HIGH-DENSITY DATA RENDERING (DEEP BLUE & VIBRANT RED)
# ═══════════════════════════════════════════════════════════════════════════
print("📍 [3/4] Rendering coordinates...")

# Render Training Data (Deep/Dark Professional Blue with high visibility)
ax.scatter(df_train['X'], df_train['Y'], 
           color='#1f4e79', alpha=0.75, s=2, edgecolors='none', marker='o',
           label=f'Train Data ({len(df_train):,} users)', zorder=2)

# Render Testing Data (Vibrant Red - High contrast 'x' markers)
ax.scatter(df_test['X'], df_test['Y'], 
           color='#e74c3c', alpha=0.9, s=4, marker='x',
           label=f'Test Data ({len(df_test):,} users)', zorder=3)

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: STYLING & IMAGE EXPORT
# ═══════════════════════════════════════════════════════════════════════════
print("💾 [4/4] Applying professional styles and exporting...")

# Typography and Axis Configurations
ax.set_title("Hybrid Dataset: Train vs. Test Spatial Distribution", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Spatial X Coordinate (meters)", fontsize=12, fontweight='bold')
ax.set_ylabel("Spatial Y Coordinate (meters)", fontsize=12, fontweight='bold')

# Background Grid Styling
ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc', zorder=1)

# High-Visibility Legend Box
ax.legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='#999999', shadow=True, markerscale=3)

# Export the final high-res figure
plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=300, bbox_inches='tight')
plt.close()

print(f"🎉 SUCCESS! High-contrast 2D Map generated beautifully!")
print(f"👉 Saved to: {OUTPUT_MAP}")
print("============================================================")