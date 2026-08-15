"""
================================================================================
5G/6G AI-Powered Beamforming & Base Station Selection — Executive Dashboard
================================================================================
Senior AI Telecom Engineer / Streamlit UI-UX rebuild of the original Matplotlib
demo. Fully interactive, cached, and built around the updated dataset schema
(spatial X/Y/Z coordinates prepended to the per-BS channel blocks).

Run with:  streamlit run app.py
================================================================================
"""

import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Lambda

# ==============================================================================
# BLOCK 0 — PAGE CONFIG (must be the first Streamlit call)
# ==============================================================================
st.set_page_config(
    page_title="5G/6G AI Beamforming Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# BLOCK 1 — CONFIGURATION & PORTABLE PATHS
# ==============================================================================
# Portable project root -> works regardless of where the repo is cloned/deployed.
PROJECT_ROOT = Path(__file__).resolve().parent

# Data / model locations are configurable from the sidebar (with sane defaults),
# so the app never hard-fails just because a laptop-specific path changed.
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "data" / "test_Dataset_Hybrid.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_cascaded_model_v6_unitnorm.keras"

# Base-station static deployment (X, Y, Z in meters). Z is antenna mast height —
# the source dataset only carried X/Y for the BS, so we assign a realistic
# rooftop/mast height per site for the 3D scene.
BS_COORDS = {
    1: {"name": "BS 1", "x": 3.47, "y": 72.91, "z": 25.0, "color": "#2E86FF", "symbol": "diamond"},
    2: {"name": "BS 2", "x": -39.11, "y": 6.35, "z": 25.0, "color": "#1BAF7A", "symbol": "square"},
    3: {"name": "BS 3", "x": 58.69, "y": -34.37, "z": 25.0, "color": "#E34948", "symbol": "cross"},
}

N_ANT = 8              # antennas per array
D_OVER_LAMBDA = 0.5    # element spacing / wavelength
UE_HEIGHT_DEFAULT = 1.5  # meters, used when the dataset has no Z for the UE

# Headline model metrics (fallback display values — recomputed live from the
# test set whenever the model + data are both loaded successfully).
FALLBACK_METRICS = {"accuracy": 0.9938, "cosine_sim": 0.9901, "mae": 0.0131}

# ==============================================================================
# BLOCK 2 — DARK / FUTURISTIC TELECOM THEME (custom CSS)
# ==============================================================================
CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #10182a 0%, #0a0f1c 45%, #060a12 100%);
        color: #e6edf3;
    }
    section[data-testid="stSidebar"] {
        background: #0d1424;
        border-right: 1px solid #1c2740;
    }
    h1, h2, h3, h4 { color: #f2f6ff !important; letter-spacing: 0.3px; }
    .app-header {
        padding: 22px 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(46,134,255,0.18), rgba(27,175,122,0.10));
        border: 1px solid rgba(90,140,255,0.35);
        margin-bottom: 18px;
    }
    .app-header h1 {
        font-size: 2.1rem;
        margin: 0;
        background: linear-gradient(90deg, #6fb1ff, #7CE2C4 60%, #f0c040);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p { color: #9fb0c9; margin-top: 6px; font-size: 0.95rem; }
    .kpi-card {
        background: #101a2e;
        border: 1px solid #22314e;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .kpi-card .kpi-label { color: #8b9ab5; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card .kpi-value { color: #f2f6ff; font-size: 1.9rem; font-weight: 700; margin-top: 4px; }
    .kpi-card .kpi-delta { font-size: 0.8rem; margin-top: 2px; }
    .status-pill {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem;
    }
    .status-ok { background: rgba(27,175,122,0.18); color: #4ee0a9; border: 1px solid #1baf7a; }
    .status-bad { background: rgba(227,73,72,0.18); color: #ff8a89; border: 1px solid #e34948; }
    div[data-testid="stMetricValue"] { color: #f2f6ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #101a2e; border-radius: 10px 10px 0 0; padding: 10px 18px; color: #9fb0c9;
    }
    .stTabs [aria-selected="true"] { background: #182742; color: #f2f6ff !important; }
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# BLOCK 3 — KERAS 3 LAMBDA-LAYER COMPATIBILITY PATCH
# ==============================================================================
# The cascaded model was trained with legacy `Lambda` layers whose serialized
# `compute_output_shape` can fail to resolve under Keras 3. We patch it with a
# name-aware fallback so the model loads cleanly without manual rebuilding.
builtins.tf = tf  # exposes `tf` to any legacy Lambda closures being deserialized

_original_compute_output_shape = Lambda.compute_output_shape


def _patched_compute_output_shape(self, input_shape):
    try:
        return _original_compute_output_shape(self, input_shape)
    except Exception:
        name = self.name or ""
        if name in ("Gate_p1", "Gate_p2", "Gate_p3"):
            return (input_shape[0], 1) if isinstance(input_shape, tuple) else (None, 1)
        elif name in ("Gated_B1", "Gated_B2", "Gated_B3", "Soft_Gate_Fusion"):
            return input_shape[0]
        return input_shape


Lambda.compute_output_shape = _patched_compute_output_shape

# ==============================================================================
# BLOCK 4 — CACHED LOADERS (model + data)
# ==============================================================================
@st.cache_resource(show_spinner="Loading cascaded Keras model…")
def load_model_safely(filepath: str):
    """Load the trained cascaded classification+regression model once per session."""
    path = Path(filepath)
    if not path.exists():
        return None
    model = tf.keras.models.load_model(
        str(path), compile=False, safe_mode=False, custom_objects={"tf": tf}
    )
    return model


@st.cache_data(show_spinner="Loading test dataset…")
def load_data(filepath: str):
    """Load the test CSV and slice it per the fixed schema contract."""
    path = Path(filepath)
    if not path.exists():
        return None
    df = pd.read_csv(path)

    # --- STRICT COLUMN CONTRACT (do not reorder / do not guess) -------------
    coord_cols = df.columns[0:3].tolist()      # X, Y, Z
    bs1_cols   = df.columns[3:19].tolist()     # BS1 channel features
    bs2_cols   = df.columns[19:35].tolist()    # BS2 channel features
    bs3_cols   = df.columns[35:51].tolist()    # BS3 channel features
    bs_col     = df.columns[51]                # target BS id (classification)
    w_cols     = df.columns[52:68].tolist()    # target beamforming weights
    # --------------------------------------------------------------------

    return {
        "df": df,
        "coord_cols": coord_cols,
        "bs1_cols": bs1_cols,
        "bs2_cols": bs2_cols,
        "bs3_cols": bs3_cols,
        "bs_col": bs_col,
        "w_cols": w_cols,
    }


@st.cache_data(show_spinner="Running full batch inference…")
def run_batch_inference(_model, df_dict_key: str, df: pd.DataFrame, bs1_cols, bs2_cols, bs3_cols, bs_col, w_cols):
    """Run the model across the whole test set once, cache predictions for the
    Batch Analytics tab. `_model` is excluded from hashing (Streamlit convention:
    leading underscore); `df_dict_key` is a cheap fingerprint to key the cache."""
    X1 = df[bs1_cols].values.astype(np.float32)
    X2 = df[bs2_cols].values.astype(np.float32)
    X3 = df[bs3_cols].values.astype(np.float32)

    probs, w_pred = _model.predict([X1, X2, X3], verbose=0)
    pred_bs = np.argmax(probs, axis=1) + 1
    conf = probs[np.arange(len(probs)), pred_bs - 1] * 100

    true_bs = parse_true_bs(df[bs_col])
    w_true = df[w_cols].values.astype(np.float32)

    cos_sim = row_cosine_similarity(w_pred, w_true)
    mae = np.mean(np.abs(w_pred - w_true), axis=1)

    return pd.DataFrame({
        "true_bs": true_bs,
        "pred_bs": pred_bs,
        "confidence": conf,
        "cosine_sim": cos_sim,
        "mae": mae,
    }), w_pred, w_true, probs


def parse_true_bs(series: pd.Series) -> np.ndarray:
    """Robustly parse the raw BS-id column into integer BS indices (1..3)."""
    out = []
    for raw in series:
        try:
            s = str(raw).strip().upper()
            if "BS" in s:
                out.append(int(s.replace("BS", "").strip()))
            else:
                out.append(int(float(raw)))
        except Exception:
            out.append(1)
    return np.array(out)


def row_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-10
    return num / den

# ==============================================================================
# BLOCK 5 — BEAMFORMING MATH HELPERS
# ==============================================================================
def compute_beam_angle(w_vec: np.ndarray):
    """Derive steering angle (deg) + complex weight vector from a real/imag pair."""
    w_cpx = w_vec[:N_ANT] + 1j * w_vec[N_ANT:]
    phase_diff = np.mean(np.angle(w_cpx[1:] * np.conj(w_cpx[:-1])))
    sin_theta = np.clip(phase_diff / (2 * np.pi * D_OVER_LAMBDA), -1.0, 1.0)
    theta_deg = np.degrees(np.arcsin(sin_theta))
    return theta_deg, w_cpx


def array_factor(w_cpx: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Normalized array factor magnitude across a sweep of angles."""
    n = np.arange(len(w_cpx))
    af = np.array([
        abs(np.dot(w_cpx.conj(), np.exp(1j * 2 * np.pi * D_OVER_LAMBDA * n * np.sin(a))))
        for a in angles_rad
    ])
    return af / (af.max() + 1e-10)

# ==============================================================================
# BLOCK 6 — SIDEBAR: DATA / MODEL SOURCES
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    model_path_input = st.text_input("Model path (.keras)", value=str(DEFAULT_MODEL_PATH))
    data_path_input = st.text_input("Test dataset path (.csv)", value=str(DEFAULT_TEST_DATA_PATH))
    st.caption("Update these paths to point at your local training artifacts.")
    st.divider()
    st.markdown("### 📡 Network Topology")
    for bs in BS_COORDS.values():
        st.markdown(
            f"<span style='color:{bs['color']}'>●</span> **{bs['name']}** — "
            f"x={bs['x']:.1f}, y={bs['y']:.1f}, z={bs['z']:.0f}m",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Built for the cascaded classification + beamforming-regression model.")

# ==============================================================================
# BLOCK 7 — LOAD MODEL & DATA (graceful failure handling)
# ==============================================================================
model = load_model_safely(model_path_input)
data_bundle = load_data(data_path_input)

model_ok = model is not None
data_ok = data_bundle is not None

# ==============================================================================
# BLOCK 8 — HEADER + TOP KPI BAR
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>📡 5G/6G AI-Powered Beamforming & Base Station Selection</h1>
        <p>Cascaded deep-learning inference for serving-cell selection and hybrid beamforming
        weight synthesis — real-time spatial inspection and batch performance analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_ok:
    st.warning(
        f"⚠️ Model file not found at `{model_path_input}`. "
        "Update the path in the sidebar. The dashboard will run in limited (data-only) mode."
    )
if not data_ok:
    st.warning(
        f"⚠️ Test dataset not found at `{data_path_input}`. "
        "Update the path in the sidebar to enable spatial and analytics views."
    )

# Compute live headline metrics if both model+data are available, otherwise
# fall back to the last known benchmark values.
metrics = dict(FALLBACK_METRICS)
batch_results = None
w_pred_all = w_true_all = probs_all = None

if model_ok and data_ok:
    df = data_bundle["df"]
    try:
        batch_results, w_pred_all, w_true_all, probs_all = run_batch_inference(
            model, str(len(df)), df,
            data_bundle["bs1_cols"], data_bundle["bs2_cols"], data_bundle["bs3_cols"],
            data_bundle["bs_col"], data_bundle["w_cols"],
        )
        metrics["accuracy"] = (batch_results["pred_bs"] == batch_results["true_bs"]).mean()
        metrics["cosine_sim"] = batch_results["cosine_sim"].mean()
        metrics["mae"] = batch_results["mae"].mean()
    except Exception as e:
        st.error(f"Batch inference failed, showing fallback benchmark metrics instead. Details: {e}")

kpi_cols = st.columns(4)
kpi_defs = [
    ("Stage 1 Accuracy", f"{metrics['accuracy']*100:.2f}%", "BS classification"),
    ("Cosine Similarity", f"{metrics['cosine_sim']:.4f}", "Beam vector alignment"),
    ("Mean Absolute Error", f"{metrics['mae']:.4f}", "Weight regression"),
    ("Test Samples", f"{len(data_bundle['df']):,}" if data_ok else "—", "Loaded dataset size"),
]
for col, (label, value, sub) in zip(kpi_cols, kpi_defs):
    col.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta" style="color:#8b9ab5;">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# BLOCK 9 — TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🛰️ 3D Spatial & Sample Inspector",
    "📊 Batch Analytics & Performance",
    "🧪 What-If Live Simulation",
])

# ------------------------------------------------------------------------------
# TAB 1 — 3D Spatial + single-sample inspector
# ------------------------------------------------------------------------------
with tab1:
    if not (model_ok and data_ok):
        st.info("Load both the model and dataset from the sidebar to use this tab.")
    else:
        df = data_bundle["df"]
        coord_cols = data_bundle["coord_cols"]
        bs1_cols, bs2_cols, bs3_cols = data_bundle["bs1_cols"], data_bundle["bs2_cols"], data_bundle["bs3_cols"]
        bs_col, w_cols = data_bundle["bs_col"], data_bundle["w_cols"]

        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            idx = st.slider("Select test sample index", 0, len(df) - 1, min(0, len(df) - 1))
        with sel_col2:
            if st.button("🎲 Random sample", use_container_width=True):
                idx = int(np.random.randint(0, len(df)))
                st.session_state["_idx_override"] = idx
        if "_idx_override" in st.session_state:
            idx = st.session_state.pop("_idx_override")

        row = df.iloc[idx]
        user_x, user_y = float(row[coord_cols[0]]), float(row[coord_cols[1]])
        user_z = float(row[coord_cols[2]]) if len(coord_cols) > 2 else UE_HEIGHT_DEFAULT

        feats1 = row[bs1_cols].values.astype(np.float32).reshape(1, -1)
        feats2 = row[bs2_cols].values.astype(np.float32).reshape(1, -1)
        feats3 = row[bs3_cols].values.astype(np.float32).reshape(1, -1)

        pred_prob, pred_w = model.predict([feats1, feats2, feats3], verbose=0)
        pred_bs_idx = int(np.argmax(pred_prob[0])) + 1
        conf = float(pred_prob[0][pred_bs_idx - 1]) * 100

        true_bs_idx = int(parse_true_bs(pd.Series([row[bs_col]]))[0])
        correct = pred_bs_idx == true_bs_idx

        pred_w_vec = pred_w[0]
        true_w_vec = row[w_cols].values.astype(np.float32)
        theta_pred, w_cpx_pred = compute_beam_angle(pred_w_vec)
        theta_true, w_cpx_true = compute_beam_angle(true_w_vec)
        cos_sim_sample = float(row_cosine_similarity(pred_w_vec.reshape(1, -1), true_w_vec.reshape(1, -1))[0])
        mae_sample = float(np.mean(np.abs(pred_w_vec - true_w_vec)))

        bs_pred = BS_COORDS[pred_bs_idx]
        bs_true = BS_COORDS.get(true_bs_idx, BS_COORDS[1])

        map_col, side_col = st.columns([2, 1])

        # ---- 3D Spatial Coverage Map ----
        with map_col:
            fig3d = go.Figure()
            for bs_id, bs in BS_COORDS.items():
                is_selected = bs_id == pred_bs_idx
                fig3d.add_trace(go.Scatter3d(
                    x=[bs["x"]], y=[bs["y"]], z=[bs["z"]],
                    mode="markers+text",
                    marker=dict(size=16 if is_selected else 10, color=bs["color"], symbol=bs["symbol"],
                                line=dict(width=2, color="white")),
                    text=[bs["name"]], textposition="top center", textfont=dict(color=bs["color"], size=13),
                    name=bs["name"],
                ))
            # UE marker
            fig3d.add_trace(go.Scatter3d(
                x=[user_x], y=[user_y], z=[user_z], mode="markers+text",
                marker=dict(size=9, color="#f0c040", symbol="diamond", line=dict(width=1, color="white")),
                text=["UE"], textposition="bottom center", textfont=dict(color="#f0c040", size=12),
                name="User Equipment",
            ))
            # Serving-link line from predicted BS to UE
            fig3d.add_trace(go.Scatter3d(
                x=[bs_pred["x"], user_x], y=[bs_pred["y"], user_y], z=[bs_pred["z"], user_z],
                mode="lines", line=dict(color="#f0c040", width=6, dash="solid"),
                name=f"Serving link → {bs_pred['name']}",
            ))
            fig3d.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", y=1.05),
                height=480,
                title=f"3D Spatial Coverage — Sample #{idx}",
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with side_col:
            status_html = (
                '<span class="status-pill status-ok">✓ Correct Prediction</span>'
                if correct else
                '<span class="status-pill status-bad">✗ Incorrect Prediction</span>'
            )
            st.markdown(status_html, unsafe_allow_html=True)
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Predicted BS", bs_pred["name"], f"{conf:.1f}% conf.")
            m2.metric("True BS", bs_true["name"])
            m3, m4 = st.columns(2)
            m3.metric("Cosine Similarity", f"{cos_sim_sample:.4f}")
            m4.metric("MAE (weights)", f"{mae_sample:.4f}")
            st.metric("Predicted Steering Angle θ", f"{theta_pred:.2f}°", f"{theta_pred - theta_true:+.2f}° vs true")

        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)

        # ---- Stage 1: classification probability bar ----
        with chart_col1:
            bs_names = [BS_COORDS[i]["name"] for i in range(1, 4)]
            bs_colors = [BS_COORDS[i]["color"] for i in range(1, 4)]
            fig_prob = go.Figure(go.Bar(
                x=bs_names, y=pred_prob[0] * 100, marker_color=bs_colors,
                text=[f"{p*100:.1f}%" for p in pred_prob[0]], textposition="outside",
            ))
            fig_prob.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title="Stage 1 — Base Station Selection Confidence",
                yaxis_title="Confidence (%)", yaxis_range=[0, 105], height=380,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # ---- Stage 2: polar radiation pattern ----
        with chart_col2:
            angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            af_pred = array_factor(w_cpx_pred, angles)
            af_true = array_factor(w_cpx_true, angles)
            fig_polar = go.Figure()
            fig_polar.add_trace(go.Scatterpolar(
                r=af_pred, theta=np.degrees(angles), mode="lines",
                line=dict(color=bs_pred["color"], width=3), name="Predicted",
            ))
            fig_polar.add_trace(go.Scatterpolar(
                r=af_true, theta=np.degrees(angles), mode="lines",
                line=dict(color="#f0c040", width=2, dash="dash"), name="Ground Truth",
            ))
            fig_polar.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    sector=[-90, 90], radialaxis=dict(range=[0, 1], showticklabels=False),
                    angularaxis=dict(direction="clockwise", rotation=90),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title="Stage 2 — Beam Radiation Pattern (Pred vs. True)",
                height=380, legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_polar, use_container_width=True)

        # ---- Beamforming weight comparison ----
        ant = np.arange(N_ANT)
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=[f"A{i+1} Re" for i in ant], y=pred_w_vec[:N_ANT],
                                marker_color=bs_pred["color"], name="Real (Pred)"))
        fig_w.add_trace(go.Bar(x=[f"A{i+1} Im" for i in ant], y=pred_w_vec[N_ANT:],
                                marker_color=bs_pred["color"], opacity=0.5, name="Imag (Pred)"))
        fig_w.add_trace(go.Scatter(x=[f"A{i+1} Re" for i in ant], y=true_w_vec[:N_ANT],
                                    mode="markers+lines", line=dict(color="#f0c040", dash="dot"), name="Real (True)"))
        fig_w.add_trace(go.Scatter(x=[f"A{i+1} Im" for i in ant], y=true_w_vec[N_ANT:],
                                    mode="markers+lines", line=dict(color="#f0c040", dash="dot"), opacity=0.6, name="Imag (True)"))
        fig_w.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Beamforming Weight Vector — Predicted vs. Ground Truth",
            height=360, barmode="overlay", legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_w, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2 — Batch analytics & model performance
# ------------------------------------------------------------------------------
with tab2:
    if not (model_ok and data_ok) or batch_results is None:
        st.info("Load both the model and dataset from the sidebar to run batch analytics.")
    else:
        cm_col, dist_col = st.columns(2)

        with cm_col:
            labels = [1, 2, 3]
            cm = confusion_matrix(batch_results["true_bs"], batch_results["pred_bs"], labels=labels)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=[f"Pred BS{i}" for i in labels], y=[f"True BS{i}" for i in labels],
                labels=dict(color="Count"),
            )
            fig_cm.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Stage 1 Confusion Matrix — BS Classification", height=420,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with dist_col:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=batch_results["cosine_sim"], nbinsx=40, marker_color="#1baf7a",
                name="Cosine Similarity", opacity=0.85,
            ))
            fig_dist.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title="Cosine Similarity Distribution (Test Set)",
                xaxis_title="Cosine Similarity", yaxis_title="Count", height=200,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            fig_mae = go.Figure()
            fig_mae.add_trace(go.Histogram(
                x=batch_results["mae"], nbinsx=40, marker_color="#e34948",
                name="MAE", opacity=0.85,
            ))
            fig_mae.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title="MAE Distribution (Test Set)",
                xaxis_title="Mean Absolute Error", yaxis_title="Count", height=200,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔎 Data Filter & Inspector")

        filt_col1, filt_col2 = st.columns([1, 3])
        with filt_col1:
            bs_filter = st.multiselect("Filter by true BS", options=[1, 2, 3], default=[1, 2, 3])
            correct_filter = st.radio("Prediction outcome", ["All", "Correct only", "Incorrect only"], index=0)

        filtered = batch_results[batch_results["true_bs"].isin(bs_filter)]
        if correct_filter == "Correct only":
            filtered = filtered[filtered["true_bs"] == filtered["pred_bs"]]
        elif correct_filter == "Incorrect only":
            filtered = filtered[filtered["true_bs"] != filtered["pred_bs"]]

        with filt_col2:
            st.dataframe(filtered, use_container_width=True, height=280)

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered evaluation report (CSV)",
            data=csv_bytes, file_name="beamforming_evaluation_report.csv", mime="text/csv",
        )

# ------------------------------------------------------------------------------
# TAB 3 — What-If live simulation
# ------------------------------------------------------------------------------
with tab3:
    if not model_ok:
        st.info("Load the model from the sidebar to run live simulations.")
    elif not data_ok:
        st.info("Load the dataset from the sidebar — the simulator perturbs a real sample's channel features.")
    else:
        st.markdown(
            "Pick a base sample, then perturb its spatial position and channel features with noise "
            "to see how the cascaded model reacts in real time."
        )
        df = data_bundle["df"]
        coord_cols = data_bundle["coord_cols"]
        bs1_cols, bs2_cols, bs3_cols = data_bundle["bs1_cols"], data_bundle["bs2_cols"], data_bundle["bs3_cols"]

        base_idx = st.number_input("Base sample index", 0, len(df) - 1, 0, step=1)
        base_row = df.iloc[int(base_idx)]

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("**Spatial position (for visualization only)**")
            sim_x = st.slider("X (m)", -150.0, 150.0, float(base_row[coord_cols[0]]))
            sim_y = st.slider("Y (m)", -150.0, 150.0, float(base_row[coord_cols[1]]))
            sim_z = st.slider("Z (m)", 0.0, 30.0, float(base_row[coord_cols[2]]) if len(coord_cols) > 2 else UE_HEIGHT_DEFAULT)
        with sim_col2:
            st.markdown("**Channel noise injection**")
            noise_std = st.slider("Gaussian noise σ (applied to channel features)", 0.0, 1.0, 0.05, 0.01)
            seed = st.number_input("Random seed", 0, 9999, 42)

        rng = np.random.default_rng(int(seed))
        f1 = base_row[bs1_cols].values.astype(np.float32) + rng.normal(0, noise_std, N_ANT * 2)
        f2 = base_row[bs2_cols].values.astype(np.float32) + rng.normal(0, noise_std, N_ANT * 2)
        f3 = base_row[bs3_cols].values.astype(np.float32) + rng.normal(0, noise_std, N_ANT * 2)

        if st.button("▶️ Run inference", type="primary"):
            prob, w_out = model.predict(
                [f1.reshape(1, -1), f2.reshape(1, -1), f3.reshape(1, -1)], verbose=0
            )
            pred_idx = int(np.argmax(prob[0])) + 1
            conf = float(prob[0][pred_idx - 1]) * 100
            theta, w_cpx = compute_beam_angle(w_out[0])
            bs_sel = BS_COORDS[pred_idx]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Serving BS", bs_sel["name"])
            r2.metric("Confidence", f"{conf:.1f}%")
            r3.metric("Steering Angle θ", f"{theta:.2f}°")
            r4.metric("Noise σ Applied", f"{noise_std:.2f}")

            sim_fig = go.Figure()
            for bs_id, bs in BS_COORDS.items():
                sim_fig.add_trace(go.Scatter3d(
                    x=[bs["x"]], y=[bs["y"]], z=[bs["z"]], mode="markers+text",
                    marker=dict(size=16 if bs_id == pred_idx else 10, color=bs["color"], symbol=bs["symbol"]),
                    text=[bs["name"]], textposition="top center", name=bs["name"],
                ))
            sim_fig.add_trace(go.Scatter3d(
                x=[sim_x], y=[sim_y], z=[sim_z], mode="markers+text",
                marker=dict(size=9, color="#f0c040", symbol="diamond"), text=["Simulated UE"], name="Simulated UE",
            ))
            sim_fig.add_trace(go.Scatter3d(
                x=[bs_sel["x"], sim_x], y=[bs_sel["y"], sim_y], z=[bs_sel["z"], sim_z],
                mode="lines", line=dict(color="#f0c040", width=6), name="Simulated link",
            ))
            sim_fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", bgcolor="rgba(0,0,0,0)"),
                height=450, margin=dict(l=0, r=0, t=20, b=0), title="Simulated Serving Decision",
            )
            st.plotly_chart(sim_fig, use_container_width=True)

            angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            af = array_factor(w_cpx, angles)
            fig_polar_sim = go.Figure(go.Scatterpolar(
                r=af, theta=np.degrees(angles), mode="lines", line=dict(color=bs_sel["color"], width=3),
            ))
            fig_polar_sim.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(sector=[-90, 90], radialaxis=dict(range=[0, 1], showticklabels=False),
                           angularaxis=dict(direction="clockwise", rotation=90), bgcolor="rgba(0,0,0,0)"),
                title="Simulated Beam Pattern", height=360,
            )
            st.plotly_chart(fig_polar_sim, use_container_width=True)
        else:
            st.caption("Adjust the parameters above, then click **Run inference**.")