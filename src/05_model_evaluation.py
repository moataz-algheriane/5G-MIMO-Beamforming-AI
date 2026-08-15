"""
================================================================================
BLOCK 1: ENVIRONMENT SETUP & DEPENDENCIES
================================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, BatchNormalization, LayerNormalization,
    Activation, Dropout, Add, Lambda, UnitNormalization
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# DYNAMIC PATH SETUP (PORTABLE ACROSS ALL MACHINES)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Define paths
TEST_DATA_PATH = DATA_DIR / "test_Dataset_Hybrid.csv"
MODEL_PATH = MODELS_DIR / "best_cascaded_model_v6_unitnorm.keras"

# Output save paths
CONF_MATRIX_PATH = DOCS_DIR / "classification_confusion_matrix.png"
REG_METRICS_PATH = DOCS_DIR / "regression_error_distribution.png"
REPORT_TXT_PATH = DOCS_DIR / "model_evaluation_report.txt"
# ==============================================================================
# BLOCK 2: REBUILDING EXACT ARCHITECTURE FOR SAFE WEIGHT LOADING
# ==============================================================================
print("--> Rebuilding model architecture internally for safe weight loading...")

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

# Inputs
input_bs1 = Input(shape=(16,), name="Input_BS1")
input_bs2 = Input(shape=(16,), name="Input_BS2")
input_bs3 = Input(shape=(16,), name="Input_BS3")

# Classifier Branch (Raw Input) - Matches Training Exactly
def build_classifier_branch(inp, name):
    x = Dense(96, kernel_regularizer=l2(1e-4), name=f"{name}_Entry")(inp)
    x = cls_residual_block(x, 96, dropout_rate=0.10, l2_reg=1e-4)
    x = LayerNormalization()(x)
    x = Activation('swish', name=f"{name}_Exit")(x)
    return x

cls_b1 = build_classifier_branch(input_bs1, "CLS_B1")
cls_b2 = build_classifier_branch(input_bs2, "CLS_B2")
cls_b3 = build_classifier_branch(input_bs3, "CLS_B3")
cls_fusion = Concatenate(name="CLS_Fusion")([cls_b1, cls_b2, cls_b3])

cls_shared = Dense(192, kernel_regularizer=l2(2e-4), name="CLS_Backbone_Entry")(cls_fusion)
cls_shared = Dropout(0.30)(cls_shared)
cls_shared = cls_residual_block(cls_shared, 192, dropout_rate=0.15, l2_reg=2e-4)
cls_shared = LayerNormalization()(cls_shared)
cls_shared = Activation('swish')(cls_shared)

cls_head = Dense(96, kernel_regularizer=l2(2e-4), name="CLS_Head_Dense")(cls_shared)
cls_head = LayerNormalization()(cls_head)
cls_head = Activation('swish')(cls_head)
cls_head = Dropout(0.35)(cls_head)
output_bs = Dense(3, activation="softmax", name="output_bs_id")(cls_head)

# Regression Branch (UnitNorm) - Matches Training Exactly
def build_regression_branch(inp, name):
    normalized_inp = UnitNormalization(axis=1, name=f"{name}_UnitNorm_Reg")(inp)
    x = Dense(64, kernel_regularizer=l2(2e-4), name=f"{name}_Entry")(normalized_inp)
    x = reg_residual_block(x, 64, dropout_rate=0.10, l2_reg=2e-4)
    x = BatchNormalization()(x)
    x = Activation('swish', name=f"{name}_Exit")(x)
    return x

reg_b1 = build_regression_branch(input_bs1, "REG_B1")
reg_b2 = build_regression_branch(input_bs2, "REG_B2")
reg_b3 = build_regression_branch(input_bs3, "REG_B3")

# Soft Gating
p1 = Lambda(lambda t: tf.expand_dims(t[:, 0], axis=-1), name="Gate_p1")(output_bs)
p2 = Lambda(lambda t: tf.expand_dims(t[:, 1], axis=-1), name="Gate_p2")(output_bs)
p3 = Lambda(lambda t: tf.expand_dims(t[:, 2], axis=-1), name="Gate_p3")(output_bs)

gated_b1 = Lambda(lambda t: t[0] * t[1], name="Gated_B1")([reg_b1, p1])
gated_b2 = Lambda(lambda t: t[0] * t[1], name="Gated_B2")([reg_b2, p2])
gated_b3 = Lambda(lambda t: t[0] * t[1], name="Gated_B3")([reg_b3, p3])

gated_fusion = Lambda(lambda t: t[0] + t[1] + t[2], name="Soft_Gate_Fusion")([gated_b1, gated_b2, gated_b3])

reg_shared = Dense(128, kernel_regularizer=l2(2e-4), name="REG_Backbone_Entry")(gated_fusion)
reg_shared = Dropout(0.15)(reg_shared)
reg_shared = reg_residual_block(reg_shared, 128, dropout_rate=0.10, l2_reg=2e-4)
reg_shared = reg_residual_block(reg_shared, 128, dropout_rate=0.10, l2_reg=2e-4)
reg_shared = BatchNormalization()(reg_shared)
reg_shared = Activation('swish')(reg_shared)
output_w = Dense(16, activation="linear", name="output_w")(reg_shared)

# Assemble Model
model = Model(inputs=[input_bs1, input_bs2, input_bs3], outputs=[output_bs, output_w])# ==============================================================================
# BLOCK 3: LOADING WEIGHTS ONLY (THE BULLETPROOF METHOD)
# ==============================================================================
print(f"--> Loading trained weights from: {MODEL_PATH}")
model.load_weights(MODEL_PATH)
print("--> Weights loaded successfully!")


# ==============================================================================
# BLOCK 4: DATA LOADING & PREPROCESSING (UPDATED FOR X,Y,Z OFFSET)
# ==============================================================================
print(f"--> Reading test dataset from: {TEST_DATA_PATH}")
df_test = pd.read_csv(TEST_DATA_PATH)

# استخراج الميزات (Inputs) - بإزاحة 3 أعمدة لتجاوز X, Y, Z
X_bs1 = df_test.iloc[:, 3:19].values.astype(np.float32)
X_bs2 = df_test.iloc[:, 19:35].values.astype(np.float32)
X_bs3 = df_test.iloc[:, 35:51].values.astype(np.float32)

# استخراج المخرجات (Targets)
Y_bs_raw = df_test.iloc[:, 51].values
y_reg = df_test.iloc[:, 52:68].values.astype(np.float32)

print(f"--> Dataset Parsed Successfully. Test Samples: {df_test.shape[0]}")# ==============================================================================
# BLOCK 5: INFERENCE AND EVALUATION (UPDATED)
# ==============================================================================
print("--> Running inference over test dataset...")
pred_cls_prob, pred_reg_weights = model.predict([X_bs1, X_bs2, X_bs3], batch_size=256)

# تحويل التسميات الحقيقية إلى أرقام (0, 1, 2) في حال كانت نصوص
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_cls_true_idx = encoder.fit_transform(Y_bs_raw)

# استنتاج قرار الموديل
y_cls_pred_idx = np.argmax(pred_cls_prob, axis=1)

# حساب الدقة
cls_accuracy = accuracy_score(y_cls_true_idx, y_cls_pred_idx)
cls_report = classification_report(y_cls_true_idx, y_cls_pred_idx, target_names=["BS1", "BS2", "BS3"])

print(f"\n[STAGE 1] Classification Accuracy: {cls_accuracy * 100:.2f}%")
print(cls_report)

# رسم مصفوفة الارتباك
cm = confusion_matrix(y_cls_true_idx, y_cls_pred_idx)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BS1", "BS2", "BS3"], yticklabels=["BS1", "BS2", "BS3"])
plt.title("Stage 1: Base Station Classification Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH, dpi=300)
plt.close()

# حساب مقاييس مسار الأوزان
mae_per_sample = np.mean(np.abs(y_reg - pred_reg_weights), axis=1)
overall_mae = np.mean(mae_per_sample)

y_reg_norm = y_reg / (np.linalg.norm(y_reg, axis=1, keepdims=True) + 1e-10)
pred_reg_norm = pred_reg_weights / (np.linalg.norm(pred_reg_weights, axis=1, keepdims=True) + 1e-10)
cosine_sim_per_sample = np.sum(y_reg_norm * pred_reg_norm, axis=1)
overall_cosine_sim = np.mean(cosine_sim_per_sample)

print(f"[STAGE 2] Overall Mean Absolute Error (MAE): {overall_mae:.5f}")
print(f"[STAGE 2] Overall Beam Vector Cosine Similarity: {overall_cosine_sim:.5f}\n")


# ==============================================================================
# BLOCK 6: VISUALIZATION & REPORTING
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(mae_per_sample, kde=True, ax=axes[0], color="crimson")
axes[0].set_title("Distribution of Mean Absolute Error (MAE)")
axes[0].set_xlabel("MAE")
axes[0].grid(True)

sns.histplot(cosine_sim_per_sample, kde=True, ax=axes[1], color="forestgreen")
axes[1].set_title("Distribution of Cosine Similarity")
axes[1].set_xlabel("Cosine Similarity (Max = 1.0)")
axes[1].grid(True)

plt.suptitle("Stage 2: Beamforming Regression Test Metrics", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(REG_METRICS_PATH, dpi=300)
plt.close()

with open(REPORT_TXT_PATH, "w") as f:
    f.write("========================================================================\n")
    f.write("                5G BEAMFORMING MODEL PERFORMANCE REPORT                 \n")
    f.write("========================================================================\n")
    f.write(f"Overall Classification Accuracy: {cls_accuracy * 100:.2f}%\n")
    f.write(f"Overall Mean Absolute Error (MAE): {overall_mae:.6f}\n")
    f.write(f"Overall Vector Cosine Similarity: {overall_cosine_sim:.6f}\n")
    f.write("========================================================================\n")

print("--> [SUCCESS] Testing pipeline completed flawlessly. Check your TEST FILE folder for results!")