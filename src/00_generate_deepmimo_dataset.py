"""
===============================================================================
Script: 00_generate_deepmimo_dataset.py
Author: Moutaz Marei Algharyani
Description:
    Generates a Multi-Task Learning (MTL) dataset using DeepMIMO v4.
    Extracts 8-antenna MISO channels across 3 Base Stations (BS), filters out
    completely blocked users, calculates optimal beamforming weights using MRT,
    and exports a clean, shuffled CSV file for neural network training.

Prerequisites:
    - deepmimo library installed (pip install deepmimo)
    - DeepMIMO scenario 'city_4_phoenix_3p5' downloaded in the working directory
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import deepmimo as dm


# --- DYNAMIC PATH SETUP (PORTABLE ACROSS ALL MACHINES) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Create 'data' directory automatically if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define Output File Path
OUTPUT_FILE = DATA_DIR / "train_Dataset_Hybrid_8Ant.csv"


def generate_dataset():
    # --- BLOCK 1: CONFIGURATION ---
    SCENARIO       = 'city_4_phoenix_3p5'
    N_ANT_TX       = 8     # 8 Transmit Antennas
    N_ANT_RX       = 1     # Single Receiver Antenna
    SUBCARRIER_IDX = 0
    NUM_BS         = 3     # Number of Base Stations

    print("=" * 70)
    print("  DeepMIMO v4 — Hybrid MTL Dataset Generation | 3 BS | MISO 8x1")
    print("=" * 70)

    # --- BLOCK 2: LOAD & CONFIGURE SCENARIO ---
    print(f"\n[1/5] Loading scenario '{SCENARIO}' and computing channels...")
    try:
        dataset = dm.load(SCENARIO)
    except Exception as e:
        print(f"\n[ERROR] Failed to load DeepMIMO scenario '{SCENARIO}'.")
        print("Please ensure the scenario dataset is downloaded and placed in your working directory.")
        raise e

    params = dm.ChannelParameters()
    params.bs_antenna.shape = [1, N_ANT_TX, 1]
    params.bs_antenna.rotation = np.array([0, 0, 180])
    params.ue_antenna.shape = [1, N_ANT_RX, 1]
    dataset.compute_channels(params)

    # --- BLOCK 3: EXTRACT & RESHAPE DATA (DYNAMIC USERS) ---
    print("\n[2/5] Extracting Data...")
    bs_locs = [np.array(dataset.tx_pos[i]).flatten() for i in range(NUM_BS)]

    H_raw = dataset.channel
    ue_locs_raw = dataset.rx_pos
    H_full = np.concatenate([h for h in H_raw if len(h) > 0], axis=0)
    ue_locs = np.concatenate([loc for loc in ue_locs_raw if len(loc) > 0], axis=0)

    H_sq = np.squeeze(H_full) 
    if H_sq.ndim > 2:
        H_sq = H_sq[:, :, SUBCARRIER_IDX] if H_sq.shape[2] > H_sq.shape[1] else H_sq[:, 0, :]

    # Calculate valid user count dynamically (truly portable)
    total_samples = H_sq.shape[0]
    NUM_USERS = total_samples // NUM_BS
    
    # Truncate to exact multiple of NUM_BS
    H_sq = H_sq[:NUM_BS * NUM_USERS, :]
    ue_locs_base = ue_locs[:NUM_USERS]

    # Reshape to: (3 Base Stations, NUM_USERS, 8 Antennas)
    H_multi = H_sq.reshape((NUM_BS, NUM_USERS, -1)) 
    print(f"       Total unique users extracted: {NUM_USERS:,}")

    # --- BLOCK 4: DETERMINE BEST BS & CALCULATE WEIGHTS ---
    print("\n[3/5] Filtering Blocked Users, Comparing Channels & Calculating Weights...")

    # 1. Calculate signal norm for each BS per user
    H_norms = np.linalg.norm(H_multi, axis=2) # Shape: (3, NUM_USERS)

    # Remove completely blocked users across all 3 base stations
    BLOCKED_THR = 1e-12
    valid_mask = np.max(H_norms, axis=0) > BLOCKED_THR

    H_multi      = H_multi[:, valid_mask, :]
    H_norms      = H_norms[:, valid_mask]
    ue_locs_base = ue_locs_base[valid_mask]
    NUM_USERS    = valid_mask.sum() # Update active users count

    print(f"       Removed completely blocked users. Remaining valid users: {NUM_USERS:,}")

    # 2. Select BS with strongest signal
    best_bs_idx = np.argmax(H_norms, axis=0)
    best_bs_ids = best_bs_idx + 1  # Convert (0,1,2) to (1,2,3)

    # 3. Extract channel matrix for best BS
    H_best = H_multi[best_bs_idx, np.arange(NUM_USERS), :]
    H_best_norms = H_norms[best_bs_idx, np.arange(NUM_USERS)]

    # 4. Calculate MRT Precoding Weights for best BS
    H_norm_safe = H_best_norms[:, np.newaxis] + 1e-10
    W_best = np.conj(H_best) / H_norm_safe

    # --- BLOCK 5: BUILD, SHUFFLE & SAVE DATAFRAME ---
    print("\n[4/5] Building Unified Multi-Channel DataFrame...")
    ant_idx = range(1, N_ANT_TX + 1)

    df_list = [pd.DataFrame(ue_locs_base, columns=['X', 'Y', 'Z'])]

    # Add channel columns for all 3 Base Stations
    for bs in range(NUM_BS):
        H_curr = H_multi[bs]
        df_list.append(pd.DataFrame({f'H_BS{bs+1}_Real_{i}': H_curr[:, i-1].real for i in ant_idx}))
        df_list.append(pd.DataFrame({f'H_BS{bs+1}_Imag_{i}': H_curr[:, i-1].imag for i in ant_idx}))

    # Add Target BS ID and Weight Columns
    df_list.append(pd.DataFrame({'BS_ID': best_bs_ids}))
    df_list.append(pd.DataFrame({f'W_Real_{i}': W_best[:, i-1].real for i in ant_idx}))
    df_list.append(pd.DataFrame({f'W_Imag_{i}': W_best[:, i-1].imag for i in ant_idx}))

    # Concatenate all features
    df = pd.concat(df_list, axis=1)

    print("\n[5/5] Shuffling and Saving Data...")
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_shuffled.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("  Hybrid Dataset Generated Successfully!")
    print(f"       Output Path : {OUTPUT_FILE}")
    print(f"       Total rows  : {df_shuffled.shape[0]:,}")
    print(f"       Total cols  : {df_shuffled.shape[1]} (3 XYZ + 48 Channels + 1 BS_ID + 16 Weights)")
    print("=" * 70)


if __name__ == "__main__":
    generate_dataset()