"""
================================================================================
CASCADED SEQUENTIAL PIPELINE — Hybrid MTL for 5G Beamforming
================================================================================
Version : 6.1 — LayerNorm Fix + UnitNorm & Magnitude Extraction (Spatial Isolation)
================================================================================
"""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, BatchNormalization, LayerNormalization,Activation, Dropout, Add, Lambda, UnitNormalization  )
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
# ==============================================================================
# BLOCK 1: REPRODUCIBILITY & CONFIGURATION
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"  
DOCS_DIR = PROJECT_ROOT / "docs"      
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE      = DATA_DIR / "train_Dataset_Hybrid.csv"
MODEL_SAVE_PATH = MODELS_DIR / "best_cascaded_model_v6_unitnorm.keras"
CURVE_SAVE_PATH = DOCS_DIR / "cascaded_learning_curve_v6_unitnorm.png"
BATCH_SIZE          = 256    
MAX_EPOCHS          = 300
LEARNING_RATE       = 3e-4   
NORM_PENALTY_WEIGHT = 0.3   
CLS_BRANCH_L2       = 1e-4   
CLS_SHARED_L2       = 2e-4  
REG_L2              = 2e-4   
# ==============================================================================
# BLOCK 2: DATA LOADING & STRATIFIED SPLIT
# ==============================================================================
print("=" * 70)
print("BLOCK 2 — Loading Data & Stratified Split")
print("=" * 70)
df_full = pd.read_csv(TRAIN_FILE)
df_train, df_val = train_test_split(df_full,test_size=0.15,random_state=SEED,stratify=df_full['BS_ID'])
def extract_features(df):
    X_bs1 = df.iloc[:, 3:19].values.astype(np.float32)    
    X_bs2 = df.iloc[:, 19:35].values.astype(np.float32)   
    X_bs3 = df.iloc[:, 35:51].values.astype(np.float32)   
    Y_bs  = df.iloc[:, 51].values                         
    Y_w   = df.iloc[:, 52:68].values.astype(np.float32)   
    return X_bs1, X_bs2, X_bs3, Y_bs, Y_w
X_bs1_tr,  X_bs2_tr,  X_bs3_tr,  Y_bs_tr,  Y_w_tr  = extract_features(df_train)
X_bs1_val, X_bs2_val, X_bs3_val, Y_bs_val, Y_w_val = extract_features(df_val)
encoder      = LabelEncoder()
Y_bs_tr_enc  = encoder.fit_transform(Y_bs_tr)
Y_bs_val_enc = encoder.transform(Y_bs_val)
NUM_CLASSES  = len(np.unique(Y_bs_tr_enc))
Y_bs_tr_cat  = to_categorical(Y_bs_tr_enc,  NUM_CLASSES)
Y_bs_val_cat = to_categorical(Y_bs_val_enc, NUM_CLASSES)
print(f"Training samples   : {len(df_train):,}  (85%)")
print(f"Validation samples : {len(df_val):,}   (15%)")
print(f"Number of classes  : {NUM_CLASSES}")
# ==============================================================================
# BLOCK 3: CUSTOM LOSS FUNCTIONS & METRICS
# ==============================================================================
def mrt_loss(y_true, y_pred):
    mse_term     = tf.reduce_mean(tf.square(y_true - y_pred))
    w_norm       = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=1) + 1e-10)
    norm_penalty = tf.reduce_mean(tf.square(w_norm - 1.0))
    return mse_term + NORM_PENALTY_WEIGHT * norm_penalty
def cosine_similarity_metric(y_true, y_pred):
    y_true_norm = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
    return tf.reduce_mean(tf.reduce_sum(y_true_norm * y_pred_norm, axis=1))
# ==============================================================================
# BLOCK 4: BUILDING BLOCKS
# ==============================================================================
def cls_residual_block(x, units, dropout_rate=0.15, l2_reg=1e-4):
    shortcut = x
    x = LayerNormalization()(x)          
    x = Activation('swish')(x)
    
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    x = LayerNormalization()(x)
    x = Activation('swish')(x)
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
    if shortcut.shape[-1] != units:
        shortcut = Dense(units, kernel_regularizer=l2(l2_reg))(shortcut)
    return Add()([x, shortcut])

def reg_residual_block(x, units, dropout_rate=0.10, l2_reg=2e-4):
    shortcut = x

    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)

    if shortcut.shape[-1] != units:
        shortcut = Dense(units, kernel_regularizer=l2(l2_reg))(shortcut)
    return Add()([x, shortcut])
# ==============================================================================
# BLOCK 5: MODEL INPUTS
# ==============================================================================
input_bs1 = Input(shape=(16,), name="Input_BS1")
input_bs2 = Input(shape=(16,), name="Input_BS2")
input_bs3 = Input(shape=(16,), name="Input_BS3")
# ==============================================================================
# BLOCK 6: STAGE 1 — BASE STATION CLASSIFICATION (Raw Input)
# ==============================================================================
def build_classifier_branch(inp, name):
    x = Dense(96, kernel_regularizer=l2(CLS_BRANCH_L2), name=f"{name}_Entry")(inp) 
    x = cls_residual_block(x, 96, dropout_rate=0.10, l2_reg=CLS_BRANCH_L2)
    x = LayerNormalization()(x)
    x = Activation('swish', name=f"{name}_Exit")(x)
    return x
cls_b1 = build_classifier_branch(input_bs1, "CLS_B1")
cls_b2 = build_classifier_branch(input_bs2, "CLS_B2")
cls_b3 = build_classifier_branch(input_bs3, "CLS_B3")

cls_fusion = Concatenate(name="CLS_Fusion")([cls_b1, cls_b2, cls_b3])

cls_shared = Dense(192, kernel_regularizer=l2(CLS_SHARED_L2),name="CLS_Backbone_Entry")(cls_fusion)
cls_shared = Dropout(0.30)(cls_shared)
cls_shared = cls_residual_block(cls_shared, 192,dropout_rate=0.15,l2_reg=CLS_SHARED_L2)
cls_shared = LayerNormalization()(cls_shared)
cls_shared = Activation('swish')(cls_shared)

cls_head = Dense(96, kernel_regularizer=l2(CLS_SHARED_L2),name="CLS_Head_Dense")(cls_shared)
cls_head = LayerNormalization()(cls_head)
cls_head = Activation('swish')(cls_head)
cls_head = Dropout(0.35)(cls_head)

output_bs = Dense(NUM_CLASSES, activation="softmax",name="output_bs_id")(cls_head)
# ==============================================================================
# BLOCK 7: STAGE 2 — SOFT-GATED CONDITIONAL BEAMFORMING REGRESSION
# ==============================================================================
def build_regression_branch(inp, name):
    normalized_inp = UnitNormalization(axis=1, name=f"{name}_UnitNorm_Reg")(inp)
    x = Dense(64, kernel_regularizer=l2(REG_L2),name=f"{name}_Entry")(normalized_inp) 
    x = reg_residual_block(x, 64, dropout_rate=0.10, l2_reg=REG_L2)
    x = BatchNormalization()(x)
    x = Activation('swish', name=f"{name}_Exit")(x)
    return x
reg_b1 = build_regression_branch(input_bs1, "REG_B1")
reg_b2 = build_regression_branch(input_bs2, "REG_B2")
reg_b3 = build_regression_branch(input_bs3, "REG_B3")
# ── Soft gate ─────────────────────────────────────────────────────────────────
p1 = Lambda(lambda t: tf.expand_dims(t[:, 0], axis=-1), name="Gate_p1")(output_bs)
p2 = Lambda(lambda t: tf.expand_dims(t[:, 1], axis=-1), name="Gate_p2")(output_bs)
p3 = Lambda(lambda t: tf.expand_dims(t[:, 2], axis=-1), name="Gate_p3")(output_bs)
gated_b1 = Lambda(lambda t: t[0] * t[1], name="Gated_B1")([reg_b1, p1])
gated_b2 = Lambda(lambda t: t[0] * t[1], name="Gated_B2")([reg_b2, p2])
gated_b3 = Lambda(lambda t: t[0] * t[1], name="Gated_B3")([reg_b3, p3])
gated_fusion = Lambda(lambda t: t[0] + t[1] + t[2],name="Soft_Gate_Fusion")([gated_b1, gated_b2, gated_b3])
reg_shared = Dense(128, kernel_regularizer=l2(REG_L2),name="REG_Backbone_Entry")(gated_fusion)
reg_shared = Dropout(0.15)(reg_shared)
reg_shared = reg_residual_block(reg_shared, 128, dropout_rate=0.10, l2_reg=REG_L2)
reg_shared = reg_residual_block(reg_shared, 128, dropout_rate=0.10, l2_reg=REG_L2)
reg_shared = BatchNormalization()(reg_shared)
reg_shared = Activation('swish')(reg_shared)
output_w = Dense(16, activation="linear", name="output_w")(reg_shared)
# ==============================================================================
# BLOCK 8: FINAL MODEL ASSEMBLY
# ==============================================================================
model = Model(inputs=[input_bs1, input_bs2, input_bs3],outputs=[output_bs, output_w],name="Cascaded_Beamforming_v6.1_UnitNorm")
model.summary() 
# ==============================================================================
# BLOCK 9: COMPILATION
# ==============================================================================
model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,clipnorm=1.0),
               loss={"output_bs_id": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),"output_w": mrt_loss},
               loss_weights={"output_bs_id": 1.3,"output_w":     1.0},
               metrics={"output_bs_id": ["accuracy"],"output_w":     ["mae", cosine_similarity_metric]}       )
# ==============================================================================
# BLOCK 10: CALLBACKS
# ==============================================================================
callbacks = [     EarlyStopping(monitor="val_loss",patience=35,restore_best_weights=True,verbose=1),
                  ModelCheckpoint(filepath=MODEL_SAVE_PATH,monitor="val_loss",save_best_only=True,verbose=1),
                  ReduceLROnPlateau(monitor="val_loss",factor=0.4,patience=12,min_lr=2e-6,verbose=1)            ]
# ==============================================================================
# BLOCK 11: TRAINING
# ==============================================================================
print("=" * 70)
print("BLOCK 11 — Training (Cascaded Pipeline v6.1 — LayerNorm + UnitNorm)")
print("=" * 70)
history = model.fit(  x=[X_bs1_tr, X_bs2_tr, X_bs3_tr],
                      y={"output_bs_id": Y_bs_tr_cat, "output_w": Y_w_tr},
                      validation_data=( [X_bs1_val, X_bs2_val, X_bs3_val],
                                              {"output_bs_id": Y_bs_val_cat, "output_w": Y_w_val}),
    epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks,verbose=1)
# ==============================================================================
# BLOCK 12: LEARNING CURVES
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
# Plot 1 — Classification Accuracy
axes[0].plot(history.history["output_bs_id_accuracy"], label="Train Acc", linewidth=2)
axes[0].plot(history.history["val_output_bs_id_accuracy"],label="Val Acc", linewidth=2)
axes[0].axhline(y=0.90, color='red', linestyle='--',linewidth=1.2, label="90% target")
axes[0].set_title("Stage 1: BS Classification Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)
# Plot 2 — Beamforming Regression Loss
axes[1].plot(history.history["output_w_loss"],label="Train Reg Loss", linewidth=2)
axes[1].plot(history.history["val_output_w_loss"],label="Val Reg Loss", linewidth=2)
axes[1].set_title("Stage 2: Beamforming Regression Loss (MRT)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss (log scale)")
axes[1].legend()
axes[1].grid(True)
axes[1].set_yscale("log")
# Plot 3 — Cosine Similarity
axes[2].plot(history.history["output_w_cosine_similarity_metric"], label="Train Cosine Sim", linewidth=2)
axes[2].plot(history.history["val_output_w_cosine_similarity_metric"],label="Val Cosine Sim", linewidth=2)
axes[2].set_title("Stage 2: Beamforming Cosine Similarity")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Cosine Similarity")
axes[2].legend()
axes[2].grid(True)
plt.suptitle("Cascaded Sequential Pipeline v6.1 — Training Curves",fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(CURVE_SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()
# ── Final metrics summary ─────────────────────────────────────────────────────
best_val_acc = max(history.history["val_output_bs_id_accuracy"])
best_val_cos = max(history.history["val_output_w_cosine_similarity_metric"])
best_val_mae = min(history.history["val_output_w_mae"])
print("\n" + "=" * 70)
print("TRAINING COMPLETE — Final Best Validation Metrics")
print("=" * 70)
print(f"  BS Classification Accuracy : {best_val_acc * 100:.2f}%")
print(f"  Beamforming Cosine Sim     : {best_val_cos:.4f}")
print(f"  Beamforming MAE            : {best_val_mae:.4f}")
print(f"  Model saved  : {MODEL_SAVE_PATH}")
print(f"  Curves saved : {CURVE_SAVE_PATH}")