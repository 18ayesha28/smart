"""
Smart NeuroCare — Premium Medical Dashboard (Streamlit)

2026-level professional neuro-oncology UI with:
  - Dark glassmorphism theme, gradient accents, premium typography
  - MRI upload with red-circle tumor overlay (replaces raw segmentation mask)
  - Interactive tumor details panel (diameter, severity, area, type)
  - Deep patient profile, hospital recommendations, lifestyle guidance
  - Full PDF report generation

HOW TO RUN (Windows 11) — see README_WINDOWS.md.
    python -m venv venv
    venv\\Scripts\\activate
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import cv2
import streamlit as st
import torch
# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image

from cnn_detection_model import TumorDetectionModel, eval_transforms
from train_classification import TumorClassificationModel, CLASSES as CLASSIFICATION_CLASSES
from unet_segmentation import UNet, compute_tumor_measurements, find_tumor_circle
from hospital_recommendation import PatientContext, Hospital, recommend_hospitals
from patient_lifestyle import (
    PatientDetails, TumorAnalysisResult,
    generate_full_report, generate_lifestyle_recommendations,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart NeuroCare — AI Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
DETECTION_CKPT = "best_detection_model.pt"
CLASSIFICATION_CKPT = "best_classification_model.pt"
SEGMENTATION_CKPT = "best_segmentation_model.pt"
using_trained_weights = all(os.path.exists(p) for p in [DETECTION_CKPT, CLASSIFICATION_CKPT, SEGMENTATION_CKPT])

# ---------------------------------------------------------------------------
# Custom CSS — Premium 2026 Medical Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #060b18;
    --bg-secondary: #0c1225;
    --bg-card: rgba(15, 23, 50, 0.65);
    --bg-card-hover: rgba(20, 30, 65, 0.8);
    --glass-border: rgba(100, 140, 255, 0.12);
    --glass-glow: rgba(80, 120, 255, 0.06);
    --accent-primary: #4f8cff;
    --accent-gradient: linear-gradient(135deg, #4f8cff 0%, #a855f7 50%, #ec4899 100%);
    --accent-gradient-soft: linear-gradient(135deg, rgba(79,140,255,0.15) 0%, rgba(168,85,247,0.15) 50%, rgba(236,72,153,0.15) 100%);
    --text-primary: #e8ecf4;
    --text-secondary: #8b95b0;
    --text-muted: #5a6480;
    --severity-low: #22c55e;
    --severity-moderate: #f59e0b;
    --severity-high: #ef4444;
    --severity-critical: #dc2626;
    --border-radius: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Overrides ── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hide default Streamlit header/footer */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer, .stDeployButton {
    display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(79,140,255,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79,140,255,0.45); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1028 0%, #060b18 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}

section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div > input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(15, 23, 50, 0.8) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div:focus-within,
section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within,
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px rgba(79,140,255,0.15) !important;
}

section[data-testid="stSidebar"] .stCheckbox label span {
    color: var(--text-secondary) !important;
}

/* ── Main Content ── */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
}

.hero-header .hero-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 30px rgba(79,140,255,0.5));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

.hero-header h1 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem !important;
    line-height: 1.1 !important;
    letter-spacing: -0.03em;
}

.hero-header .hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    max-width: 600px;
    margin: 0 auto;
}

/* ── Status Banner ── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    font-size: 0.88rem;
    font-weight: 500;
    backdrop-filter: blur(12px);
}

.status-banner.success {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    color: #4ade80;
}

.status-banner.warning {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    color: #fbbf24;
}

.status-banner .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse-dot 2s infinite;
}

.status-banner.success .status-dot { background: #22c55e; }
.status-banner.warning .status-dot { background: #f59e0b; }

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(79,140,255,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(79,140,255,0); }
}

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,140,255,0.3), transparent);
}

.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(100, 140, 255, 0.2);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}

.glass-card .card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.glass-card .card-title .title-icon {
    font-size: 1.3rem;
}

/* ── File Uploader ── */
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(79,140,255,0.25) !important;
    border-radius: var(--border-radius) !important;
    transition: var(--transition) !important;
    padding: 2rem !important;
}

.stFileUploader > div:hover {
    border-color: rgba(79,140,255,0.5) !important;
    background: rgba(15, 23, 50, 0.8) !important;
    box-shadow: 0 0 30px rgba(79,140,255,0.08) !important;
}

.stFileUploader label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* ── Button ── */
.stButton > button {
    background: var(--accent-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em;
    transition: var(--transition) !important;
    box-shadow: 0 4px 20px rgba(79,140,255,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(79,140,255,0.35) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(79,140,255,0.15), rgba(168,85,247,0.15)) !important;
    color: var(--accent-primary) !important;
    border: 1px solid rgba(79,140,255,0.3) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(79,140,255,0.25), rgba(168,85,247,0.25)) !important;
    border-color: rgba(79,140,255,0.5) !important;
    box-shadow: 0 4px 20px rgba(79,140,255,0.15) !important;
}

/* ── Metric Cards Grid ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.25rem 0;
}

.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: var(--transition);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 3px 3px 0 0;
}

.metric-card.blue::before { background: linear-gradient(90deg, #4f8cff, #60a5fa); }
.metric-card.purple::before { background: linear-gradient(90deg, #a855f7, #c084fc); }
.metric-card.green::before { background: linear-gradient(90deg, #22c55e, #4ade80); }
.metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.metric-card.red::before { background: linear-gradient(90deg, #ef4444, #f87171); }
.metric-card.pink::before { background: linear-gradient(90deg, #ec4899, #f472b6); }

.metric-card:hover {
    border-color: rgba(100, 140, 255, 0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transform: translateY(-2px);
}

.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.metric-card .metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-card .metric-sub {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* ── Detection Result Badge ── */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}

.detection-badge.detected {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
}

.detection-badge.clear {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
}

.detection-badge .badge-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.detection-badge.detected .badge-dot {
    background: #ef4444;
    box-shadow: 0 0 12px rgba(239,68,68,0.5);
    animation: pulse-dot 1.5s infinite;
}

.detection-badge.clear .badge-dot {
    background: #22c55e;
    box-shadow: 0 0 12px rgba(34,197,94,0.5);
}

/* ── Severity Badge ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.severity-badge.low {
    background: rgba(34,197,94,0.12);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.25);
}

.severity-badge.moderate {
    background: rgba(245,158,11,0.12);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.25);
}

.severity-badge.high {
    background: rgba(239,68,68,0.12);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.25);
}

.severity-badge.critical {
    background: rgba(220,38,38,0.15);
    color: #fca5a5;
    border: 1px solid rgba(220,38,38,0.3);
}

/* ── MRI Image Container ── */
.mri-container {
    position: relative;
    border-radius: var(--border-radius);
    overflow: hidden;
    border: 1px solid var(--glass-border);
    background: #000;
}

.mri-container img {
    width: 100%;
    display: block;
}

/* ── Tumor Tooltip (details on image) ── */
.tumor-details-overlay {
    background: rgba(10, 14, 30, 0.92);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    box-shadow: 0 4px 24px rgba(239,68,68,0.1);
}

.tumor-details-overlay .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.tumor-details-overlay .detail-row:last-child {
    border-bottom: none;
}

.tumor-details-overlay .detail-label {
    color: var(--text-muted);
    font-size: 0.82rem;
    font-weight: 500;
}

.tumor-details-overlay .detail-value {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 0.88rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 2rem 0;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
}

.streamlit-expanderHeader:hover {
    background: var(--bg-card-hover) !important;
    border-color: rgba(100, 140, 255, 0.2) !important;
}

.streamlit-expanderContent {
    background: rgba(10, 14, 30, 0.5) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: var(--accent-primary) !important;
}

/* ── General text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: var(--text-secondary) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Subheader (section) styling ── */
.section-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.section-header .section-icon {
    font-size: 1.3rem;
}

/* ── Info/Warning boxes ── */
.stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
}

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-style: italic;
    border-top: 1px solid rgba(100,140,255,0.08);
}

/* ── Hospital Card ── */
.hospital-card {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: var(--transition);
}

.hospital-card:hover {
    border-color: rgba(79,140,255,0.25);
    background: var(--bg-card-hover);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}

.hospital-card .hospital-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}

.hospital-card .hospital-meta {
    display: flex;
    gap: 1.25rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    flex-wrap: wrap;
}

.hospital-card .hospital-meta span {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.hospital-card .match-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Lifestyle Category ── */
.lifestyle-category {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: var(--transition);
}

.lifestyle-category:hover {
    border-color: rgba(100,140,255,0.2);
    background: var(--bg-card-hover);
}

.lifestyle-category .category-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.lifestyle-category ul {
    margin: 0;
    padding-left: 1.25rem;
}

.lifestyle-category li {
    color: var(--text-secondary) !important;
    font-size: 0.88rem;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    detection_model = TumorDetectionModel(pretrained=False)
    detection_model.load_state_dict(torch.load("best_detection_model.pt", map_location="cpu"))
    detection_model.eval()

    classification_model = TumorClassificationModel(num_classes=4)
    classification_model.load_state_dict(torch.load("best_classification_model.pt", map_location="cpu"))
    classification_model.eval()

    segmentation_model = UNet(in_channels=1, out_channels=1)
    segmentation_model.load_state_dict(torch.load("best_segmentation_model.pt", map_location="cpu"))
    segmentation_model.eval()

    return detection_model, classification_model, segmentation_model


detection_model, classification_model, segmentation_model = load_models()


# ---------------------------------------------------------------------------
# Helper: draw red circle overlay on image
# ---------------------------------------------------------------------------
def draw_tumor_overlay(image: Image.Image, circle_info: dict) -> Image.Image:
    """Draw a red circle on the PIL image using OpenCV, return annotated PIL image."""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cx, cy, r = circle_info["center_x"], circle_info["center_y"], circle_info["radius"]

    # Draw outer glow ring (semi-transparent)
    overlay = img_bgr.copy()
    cv2.circle(overlay, (cx, cy), r + 4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)

    # Draw main red circle
    cv2.circle(img_bgr, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw crosshair tick marks
    tick_len = max(8, r // 5)
    color = (0, 0, 255)
    cv2.line(img_bgr, (cx - r, cy), (cx - r + tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx + r, cy), (cx + r - tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy - r), (cx, cy - r + tick_len), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy + r), (cx, cy + r - tick_len), color, 1, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ---------------------------------------------------------------------------
# Helper: severity badge HTML
# ---------------------------------------------------------------------------
def severity_html(severity: str) -> str:
    cls = severity.lower() if severity else "moderate"
    return f'<span class="severity-badge {cls}">{severity.upper() if severity else "N/A"}</span>'


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🧠</div>
    <h1>Smart NeuroCare</h1>
    <div class="hero-subtitle">AI-Powered Brain Tumor Detection & Analysis Platform</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model status banner
# ---------------------------------------------------------------------------
if using_trained_weights:
    st.markdown("""
    <div class="status-banner success">
        <div class="status-dot"></div>
        <span>Trained model checkpoints loaded — ready for clinical-grade analysis</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-banner warning">
        <div class="status-dot"></div>
        <span>⚠ Demo mode — using untrained weights. Results are NOT medically meaningful. See TRAINING_GUIDE.md.</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: Patient Profile
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 👤 Patient Profile")
name = st.sidebar.text_input("Full name", "Jane Doe")
patient_id = st.sidebar.text_input("Patient ID", "P-10293")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
sex = st.sidebar.selectbox("Sex", ["female", "male", "other"])
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=70.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=30.0, value=165.0)

st.sidebar.markdown("### 🚬 Habits")
smoker = st.sidebar.checkbox("Currently smokes")
cigarettes_per_day = st.sidebar.number_input("Cigarettes/day", min_value=0, value=0) if smoker else 0
alcohol_use = st.sidebar.selectbox("Alcohol use", ["none", "occasional", "regular", "heavy"])
physical_activity = st.sidebar.selectbox("Physical activity", ["sedentary", "light", "moderate", "active"], index=2)

st.sidebar.markdown("### 🏥 Medical History")
existing_conditions = st.sidebar.multiselect(
    "Existing conditions",
    ["diabetes", "hypertension", "heart disease", "asthma", "epilepsy", "none"],
    default=["none"],
)
family_history_cancer = st.sidebar.checkbox("Family history of cancer")
symptoms = st.sidebar.multiselect(
    "Current symptoms",
    ["headaches", "seizures", "vision changes", "balance issues", "nausea", "memory issues", "none"],
    default=["none"],
)

st.sidebar.markdown("### 📍 Location & Insurance")
budget = st.sidebar.number_input("Max budget (₹)", value=800000, step=50000)
insurance = st.sidebar.text_input("Insurance provider", "StarHealth")
lat = st.sidebar.number_input("Latitude", value=12.9716, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=77.5946, format="%.4f")
pixel_spacing = st.sidebar.number_input("Pixel spacing (mm)", value=1.0, step=0.1)


# ---------------------------------------------------------------------------
# Main: Upload + Analysis
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header"><span class="section-icon">📤</span> Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a brain MRI image (PNG / JPEG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Center column for the image
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_center:
        # Show uploaded image initially
        image_placeholder = st.empty()
        image_placeholder.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    if st.button("🔬 Run Analysis", use_container_width=True):
        # ----- Detection -----
        with st.spinner("Running tumor detection..."):
            tensor = eval_transforms(image).unsqueeze(0)
            with torch.no_grad():
                logit = detection_model(tensor)
                prob = torch.sigmoid(logit).item()
            tumor_detected = prob > 0.5

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Detection result badge
        if tumor_detected:
            st.markdown(f"""
            <div style="text-align:center; margin: 1rem 0;">
                <span class="detection-badge detected">
                    <span class="badge-dot"></span>
                    TUMOR DETECTED — {prob*100:.1f}% confidence
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; margin: 1rem 0;">
                <span class="detection-badge clear">
                    <span class="badge-dot"></span>
                    NO TUMOR DETECTED — {prob*100:.1f}% confidence
                </span>
            </div>
            """, unsafe_allow_html=True)

        tumor_type = None
        classification_confidence = None
        max_diameter_mm = None
        area_mm2 = None
        severity = None
        overlay_path = None
        circle_info = None

        if tumor_detected:
            # ----- Classification -----
            with st.spinner("Classifying tumor type..."):
                with torch.no_grad():
                    class_logits = classification_model(tensor)
                    class_probs = torch.softmax(class_logits, dim=1)[0]
                    pred_idx = class_probs.argmax().item()
                    tumor_type = CLASSIFICATION_CLASSES[pred_idx]
                    classification_confidence = class_probs[pred_idx].item()

            # ----- Segmentation + Red Circle Overlay -----
            with st.spinner("Localizing tumor region..."):
                gray = image.convert("L").resize((256, 256))
                seg_input = torch.tensor(np.array(gray), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                with torch.no_grad():
                    mask = segmentation_model(seg_input)[0, 0]
                    # st.write(f"🔍 Debug — mask min: {mask.min().item():.4f}, max: {mask.max().item():.4f}, mean: {mask.mean().item():.4f}")
                    # debug_heatmap = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    # st.image(debug_heatmap, caption="Raw probability heatmap (white = model thinks tumor)", width=256)
                # Compute measurements
                measurements = compute_tumor_measurements(mask, pixel_spacing_mm=pixel_spacing)
                area_mm2 = measurements["area_mm2"]
                max_diameter_mm = measurements["max_diameter_mm"]

                # Find tumor circle for overlay
                orig_w, orig_h = image.size
                circle_info = find_tumor_circle(mask, orig_w, orig_h)

            # Draw overlay and display
            with col_center:
                if circle_info:
                    annotated = draw_tumor_overlay(image, circle_info)
                    image_placeholder.image(annotated, caption="Tumor Detected — Hover below for details", use_container_width=True)

                    # Save overlay for PDF
                    overlay_path = "demo_overlay.png"
                    annotated.save(overlay_path)
                else:
                    image_placeholder.image(image, caption="Tumor detected (spatial localization unavailable)", use_container_width=True)

            # Severity calculation
            if area_mm2 is not None and area_mm2 > 0:
                if area_mm2 < 200:
                    severity = "low"
                elif area_mm2 < 500:
                    severity = "moderate"
                elif area_mm2 < 1000:
                    severity = "high"
                else:
                    severity = "critical"
            else:
                severity = "moderate"

            # ----- Tumor Details Panel (below image) -----
            st.markdown(f"""
            <div class="tumor-details-overlay">
                <div style="text-align:center; margin-bottom:0.5rem; font-weight:700; color: #f87171; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.95rem;">
                    🎯 Tumor Analysis Details
                </div>
                <div class="detail-row">
                    <span class="detail-label">Tumor Type</span>
                    <span class="detail-value" style="text-transform:capitalize;">{tumor_type or 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Classification Confidence</span>
                    <span class="detail-value">{f'{classification_confidence*100:.1f}%' if classification_confidence else 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Max Diameter</span>
                    <span class="detail-value">{f'{max_diameter_mm:.1f} mm' if max_diameter_mm else 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Cross-sectional Area</span>
                    <span class="detail-value">{f'{area_mm2:.1f} mm²' if area_mm2 else 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Severity Level</span>
                    <span class="detail-value">{severity_html(severity)}</span>
                </div>
                {f'''<div class="detail-row">
                    <span class="detail-label">Circle Radius (px)</span>
                    <span class="detail-value">{circle_info["radius"]} px</span>
                </div>''' if circle_info else ''}
            </div>
            """, unsafe_allow_html=True)

            # ----- Metric Cards -----
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="section-icon">📊</span> Analysis Metrics</div>', unsafe_allow_html=True)

            severity_color_map = {"low": "green", "moderate": "amber", "high": "red", "critical": "red"}
            sev_color = severity_color_map.get(severity, "amber")

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card blue">
                    <div class="metric-label">Detection</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div class="metric-sub">confidence score</div>
                </div>
                <div class="metric-card purple">
                    <div class="metric-label">Tumor Type</div>
                    <div class="metric-value" style="font-size:1.3rem; text-transform:capitalize;">{tumor_type or 'N/A'}</div>
                    <div class="metric-sub">{f'{classification_confidence*100:.1f}% confidence' if classification_confidence else ''}</div>
                </div>
                <div class="metric-card pink">
                    <div class="metric-label">Max Diameter</div>
                    <div class="metric-value">{f'{max_diameter_mm:.1f}' if max_diameter_mm else 'N/A'}<span style="font-size:0.9rem; font-weight:500;"> mm</span></div>
                    <div class="metric-sub">estimated max width</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-label">Area</div>
                    <div class="metric-value">{f'{area_mm2:.1f}' if area_mm2 else 'N/A'}<span style="font-size:0.9rem; font-weight:500;"> mm²</span></div>
                    <div class="metric-sub">cross-sectional</div>
                </div>
                <div class="metric-card {sev_color}">
                    <div class="metric-label">Severity</div>
                    <div class="metric-value">{severity.upper() if severity else 'N/A'}</div>
                    <div class="metric-sub">AI-estimated level</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ----- Build patient profile + analysis result -----
        patient = PatientDetails(
            name=name, patient_id=patient_id, age=age, sex=sex,
            weight_kg=weight_kg, height_cm=height_cm,
            smoker=smoker, cigarettes_per_day=cigarettes_per_day,
            alcohol_use=alcohol_use, physical_activity=physical_activity,
            existing_conditions=[c for c in existing_conditions if c != "none"],
            family_history_cancer=family_history_cancer,
            symptoms=[s for s in symptoms if s != "none"],
        )
        analysis = TumorAnalysisResult(
            tumor_detected=tumor_detected,
            detection_confidence=prob,
            tumor_type=tumor_type,
            classification_confidence=classification_confidence,
            tumor_area_mm2=area_mm2,
            tumor_volume_mm3=None,
            max_diameter_mm=max_diameter_mm,
            severity_score=severity,
            segmentation_overlay_path=overlay_path,
            model_version="trained" if using_trained_weights else "demo-untrained-v0",
            scan_date=str(np.datetime64("today")),
        )

        # ----- Hospital Recommendations -----
        recommendations = []
        if tumor_detected:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="section-icon">🏥</span> Recommended Hospitals</div>', unsafe_allow_html=True)

            patient_ctx = PatientContext(
                latitude=lat, longitude=lon,
                tumor_type=tumor_type, severity_score=severity,
                max_budget=budget, insurance_provider=insurance,
            )
            sample_hospitals = [
                Hospital("h1", "NeuroCare Institute", 12.9352, 77.6245,
                         ["neuro-oncology", "neurosurgery"], rating=4.7, success_rate=88,
                         avg_cost_min=500000, avg_cost_max=750000,
                         accepted_insurance=["StarHealth", "HDFC Ergo"]),
                Hospital("h2", "City General Hospital", 13.0827, 80.2707,
                         ["general surgery"], rating=4.0, success_rate=70,
                         avg_cost_min=300000, avg_cost_max=500000,
                         accepted_insurance=["ICICI Lombard"]),
                Hospital("h3", "Apex Brain & Spine Center", 12.9784, 77.6408,
                         ["neuro-oncology", "pediatric neurosurgery"], rating=4.9, success_rate=91,
                         avg_cost_min=600000, avg_cost_max=900000,
                         accepted_insurance=["StarHealth"]),
            ]
            recommendations = recommend_hospitals(patient_ctx, sample_hospitals, top_k=3)

            for rec in recommendations:
                reasons = rec["match_reasons"]
                st.markdown(f"""
                <div class="hospital-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="hospital-name">🏥 {rec['name']}</div>
                        <div class="match-score">{rec['match_score']:.2f}</div>
                    </div>
                    <div class="hospital-meta">
                        <span>📍 {rec['distance_km']} km</span>
                        <span>🎯 Specialization: {reasons.get('specialization_match', 'N/A')}</span>
                        <span>⭐ Quality: {reasons.get('hospital_quality', 'N/A')}</span>
                        <span>💰 Cost fit: {reasons.get('cost_fit', 'N/A')}</span>
                        <span>🛡️ Insurance: {reasons.get('insurance_fit', 'N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ----- Lifestyle Recommendations -----
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">💚</span> Personalized Lifestyle & Recovery Guidance</div>', unsafe_allow_html=True)

        lifestyle = generate_lifestyle_recommendations(patient, analysis)

        category_icons = {
            "diet": "🥗",
            "exercise": "🏃",
            "habits": "🔄",
            "monitoring": "📋",
            "warning_signs": "⚠️",
        }

        for category, items in lifestyle.items():
            if not items:
                continue
            icon = category_icons.get(category, "📌")
            title = category.replace("_", " ").title()
            items_html = "".join(f"<li>{item}</li>" for item in items)
            is_warning = "border-color: rgba(239,68,68,0.3);" if category == "warning_signs" else ""

            st.markdown(f"""
            <div class="lifestyle-category" style="{is_warning}">
                <div class="category-title">{icon} {title}</div>
                <ul>{items_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # ----- Full Report -----
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">📄</span> Full Medical Report</div>', unsafe_allow_html=True)

        report_path = "generated_report.pdf"
        generate_full_report(patient, analysis, recommendations, report_path)
        with open(report_path, "rb") as f:
            st.download_button(
                "📄 Download Full PDF Report",
                f,
                file_name="neurocare_report.pdf",
                use_container_width=True,
            )

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            ⚕️ This tool provides AI-assisted decision support only and does not replace professional medical diagnosis.
            Always consult a licensed physician or neurologist before making clinical decisions.
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 3rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">🧠</div>
        <div style="font-size: 1.1rem; color: var(--text-secondary); font-weight: 500;">
            Upload a brain MRI scan above to begin AI-powered tumor analysis
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem;">
            Supports PNG and JPEG formats
        </div>
    </div>
    """, unsafe_allow_html=True)
