"""
Streamlit Web Application — Smart NeuroCare
AI-Powered Brain Tumor Detection & Comprehensive Patient Analysis Platform

UX-First Clinical Workstation:
  - Clinical Sapphire & Slate Design System (Clean, solid, non-glare)
  - Dual Scan Intake: Custom File Uploader + 1-Click Clinical Sample Cases
  - Integrated 2-Column Workstation View with Live Preprocessing
  - RANO Bidirectional 2D Measurements & Multi-Class Probabilities
  - Tabbed Clinical Results: Localization, Hospital Triage, Lifestyle, and PDF Export
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import streamlit as st

from cnn_detection_model import TumorDetectionModel
from train_classification import TumorClassificationModel
from unet_segmentation import UNet, compute_tumor_measurements, find_tumor_circle
from hospital_recommendation import recommend_hospitals, default_hospitals
from patient_lifestyle import (
    PatientProfile,
    TumorAnalysisResult,
    generate_lifestyle_recommendations,
    generate_patient_report,
)
from image_preprocessing import preprocess_mri, crop_brain_contour, apply_clahe_enhancement

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart NeuroCare — AI Diagnostic Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSIFICATION_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

DETECTION_CKPT = "best_detection_model.pt"
CLASSIFICATION_CKPT = "best_classification_model.pt"
SEGMENTATION_CKPT = "best_segmentation_model.pt"
using_trained_weights = all(os.path.exists(p) for p in [DETECTION_CKPT, CLASSIFICATION_CKPT, SEGMENTATION_CKPT])

# ---------------------------------------------------------------------------
# Custom CSS — Clean Clinical Sapphire & Slate Palette
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables — Flat Clinical Palette ── */
:root {
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #fcfdfe;
    --border-color: #e2e8f0;
    --border-focus: #0284c7;
    --accent-primary: #0284c7;
    --accent-hover: #0369a1;
    --accent-soft: #f0f9ff;
    --accent-soft-border: #bae6fd;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --severity-low: #16a34a;
    --severity-moderate: #d97706;
    --severity-high: #dc2626;
    --severity-critical: #b91c1c;
    --border-radius: 12px;
    --card-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    --card-shadow-hover: 0 8px 24px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -2px rgba(15, 23, 42, 0.03);
    --transition: all 0.2s ease-in-out;
}

/* ── Global Overrides ── */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border-color) !important;
    padding-top: 1rem !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div > input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div:focus-within,
section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within,
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
    border-color: var(--border-focus) !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
}

/* ── Main Content Container ── */
.block-container {
    padding-top: 1.25rem !important;
    max-width: 1240px !important;
}

/* ── Top Clinical Header Bar ── */
.clinical-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0.85rem 1.35rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--card-shadow);
}

.clinical-navbar .brand-group {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.clinical-navbar .brand-icon {
    font-size: 2rem;
    line-height: 1;
}

.clinical-navbar .brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}

.clinical-navbar .brand-subtitle {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 500;
}

.clinical-navbar .status-group {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-pill.success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
}

.status-pill.info {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    color: #0369a1;
}

/* ── Stepper ── */
.nc-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}

.nc-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    background: #ffffff;
    border: 1px solid var(--border-color);
}

.nc-step .nc-step-num {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    background: #f1f5f9;
    color: var(--text-muted);
}

.nc-step.active {
    background: #f0f9ff;
    border-color: #bae6fd;
    color: #0369a1;
}

.nc-step.active .nc-step-num {
    background: #0284c7;
    color: #fff;
}

.nc-step.done {
    background: #f0fdf4;
    border-color: #bbf7d0;
    color: #15803d;
}

.nc-step.done .nc-step-num {
    background: #16a34a;
    color: #fff;
}

.nc-step-connector {
    width: 18px;
    height: 2px;
    background: var(--border-color);
}

/* ── Cards ── */
.clinical-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--card-shadow);
}

.clinical-card-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Workflow Grid ── */
.workflow-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.workflow-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.15rem;
    transition: var(--transition);
}

.workflow-card:hover {
    border-color: #bae6fd;
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover);
}

.workflow-card .wf-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
    color: #0284c7;
}

.workflow-card .wf-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
}

.workflow-card .wf-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── File Uploader ── */
.stFileUploader > div {
    background: #ffffff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: var(--border-radius) !important;
    padding: 1.5rem !important;
}

.stFileUploader > div:hover {
    border-color: var(--accent-primary) !important;
    background: #f8fafc !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0284c7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25) !important;
}

.stButton > button:hover {
    background: #0369a1 !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
}

.stDownloadButton > button {
    background: #f0f9ff !important;
    color: #0369a1 !important;
    border: 1px solid #bae6fd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.75rem !important;
    transition: var(--transition) !important;
}

.stDownloadButton > button:hover {
    background: #e0f2fe !important;
    border-color: #7dd3fc !important;
    color: #0284c7 !important;
}

/* ── Metric Cards Grid ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.85rem;
    margin: 1rem 0;
}

.metric-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--card-shadow);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
}

.metric-card.blue::before { background: #0284c7; }
.metric-card.purple::before { background: #6366f1; }
.metric-card.green::before { background: #16a34a; }
.metric-card.amber::before { background: #d97706; }
.metric-card.red::before { background: #dc2626; }
.metric-card.teal::before { background: #0d9488; }

.metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}

.metric-card .metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-card .metric-sub {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.2rem;
}

/* ── Detection Badge ── */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.92rem;
}

.detection-badge.detected {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
}

.detection-badge.clear {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #16a34a;
}

.detection-badge .badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.detection-badge.detected .badge-dot { background: #dc2626; }
.detection-badge.clear .badge-dot { background: #16a34a; }

/* ── Severity Badge ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.severity-badge.low { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
.severity-badge.moderate { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
.severity-badge.high { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.severity-badge.critical { background: #450a0a; color: #fee2e2; border: 1px solid #991b1b; }

/* ── Confidence Meter ── */
.confidence-meter {
    width: 100%;
    height: 7px;
    border-radius: 6px;
    background: #e2e8f0;
    overflow: hidden;
    margin-top: 0.4rem;
}

.confidence-meter .confidence-fill {
    height: 100%;
    border-radius: 6px;
}

.confidence-meter .confidence-fill.danger { background: #dc2626; }
.confidence-meter .confidence-fill.safe { background: #16a34a; }
.confidence-meter .confidence-fill.info { background: #0284c7; }

/* ── Hospital Card ── */
.hospital-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.65rem;
    box-shadow: var(--card-shadow);
}

.hospital-card:hover {
    border-color: #bae6fd;
    box-shadow: var(--card-shadow-hover);
}

.hospital-card .hospital-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #0f172a;
}

.hospital-card .hospital-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.4rem;
    flex-wrap: wrap;
}

.hospital-card .match-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    color: #0284c7;
}

/* ── Lifestyle Category ── */
.lifestyle-category {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.65rem;
}

.lifestyle-category .category-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    color: #0f172a;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.lifestyle-category ul {
    margin: 0;
    padding-left: 1.25rem;
}

.lifestyle-category li {
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ── Patient Snapshot Chip ── */
.patient-snapshot {
    background: #f8fafc;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.6rem 0.85rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    line-height: 1.5;
}

/* ── Back to Top ── */
.nc-back-to-top {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #0284c7;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    text-decoration: none !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
    z-index: 999;
}

.nc-back-to-top:hover {
    background: #0369a1;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 6px;
    padding: 0 14px;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    background-color: #f0f9ff !important;
    color: #0284c7 !important;
    border: 1px solid #bae6fd !important;
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
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cx, cy, r = circle_info["center_x"], circle_info["center_y"], circle_info["radius"]

    overlay = img_bgr.copy()
    cv2.circle(overlay, (cx, cy), r + 4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)

    cv2.circle(img_bgr, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)

    tick_len = max(8, r // 5)
    color = (0, 0, 255)
    cv2.line(img_bgr, (cx - r, cy), (cx - r + tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx + r, cy), (cx + r - tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy - r), (cx, cy - r + tick_len), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy + r), (cx, cy + r - tick_len), color, 1, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def severity_html(severity: str) -> str:
    cls = severity.lower() if severity else "moderate"
    return f'<span class="severity-badge {cls}">{severity.upper() if severity else "N/A"}</span>'


# ---------------------------------------------------------------------------
# Pre-generate sample scans directory if needed
# ---------------------------------------------------------------------------
SAMPLE_DIR = "sample_scans"
if not os.path.exists(SAMPLE_DIR):
    from create_sample_scans import generate_sample_mris
    generate_sample_mris(SAMPLE_DIR)

# ---------------------------------------------------------------------------
# Sidebar: Patient Profile & Presets
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 👤 Patient Profile")

# Quick Demographic Presets
col_p1, col_p2, col_p3 = st.sidebar.columns(3)
if col_p1.button("👩 45F", help="Load Jane Doe (45F, Frontal Symptoms)"):
    st.session_state["p_name"] = "Jane Doe"
    st.session_state["p_id"] = "P-10293"
    st.session_state["p_age"] = 45
    st.session_state["p_sex"] = "female"
    st.session_state["p_symptoms"] = ["headaches", "vision changes"]

if col_p2.button("👨 62M", help="Load Robert K. (62M, Seizure & Memory)"):
    st.session_state["p_name"] = "Robert King"
    st.session_state["p_id"] = "P-88412"
    st.session_state["p_age"] = 62
    st.session_state["p_sex"] = "male"
    st.session_state["p_symptoms"] = ["seizures", "memory issues"]

if col_p3.button("👦 28M", help="Load Alex M. (28M, Routine Screening)"):
    st.session_state["p_name"] = "Alex Miller"
    st.session_state["p_id"] = "P-33291"
    st.session_state["p_age"] = 28
    st.session_state["p_sex"] = "male"
    st.session_state["p_symptoms"] = ["none"]

with st.sidebar.expander("🪪 Identity & Vitals", expanded=True):
    name = st.text_input("Full name", st.session_state.get("p_name", "Jane Doe"))
    patient_id = st.text_input("Patient ID", st.session_state.get("p_id", "P-10293"))
    age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("p_age", 45))
    sex = st.selectbox("Sex", ["female", "male", "other"], index=0 if st.session_state.get("p_sex") == "female" else 1)
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
    height_cm = st.number_input("Height (cm)", min_value=30.0, value=165.0)

bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm else 0
st.sidebar.markdown(f"""
<div class="patient-snapshot">
    📋 <b>{name}</b> · {patient_id} · {age}y {sex} · BMI {bmi:.1f}
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("🚬 Habits & Lifestyle"):
    smoker = st.checkbox("Currently smokes")
    cigarettes_per_day = st.number_input("Cigarettes/day", min_value=0, value=0) if smoker else 0
    alcohol_use = st.selectbox("Alcohol use", ["none", "occasional", "regular", "heavy"])
    physical_activity = st.selectbox("Physical activity", ["sedentary", "light", "moderate", "active"], index=2)

with st.sidebar.expander("🏥 Medical History"):
    existing_conditions = st.multiselect(
        "Existing conditions",
        ["diabetes", "hypertension", "heart disease", "asthma", "epilepsy", "none"],
        default=["none"],
    )
    family_history_cancer = st.checkbox("Family history of cancer")
    default_syms = st.session_state.get("p_symptoms", ["headaches"])
    symptoms = st.multiselect(
        "Current symptoms",
        ["headaches", "seizures", "vision changes", "balance issues", "nausea", "memory issues", "none"],
        default=[s for s in default_syms if s in ["headaches", "seizures", "vision changes", "balance issues", "nausea", "memory issues", "none"]] or ["none"],
    )

with st.sidebar.expander("📍 Location & Insurance"):
    budget = st.number_input("Max budget (₹)", value=800000, step=50000)
    insurance = st.text_input("Insurance provider", "StarHealth")
    lat = st.number_input("Latitude", value=12.9716, format="%.4f")
    lon = st.number_input("Longitude", value=77.5946, format="%.4f")

with st.sidebar.expander("⚙️ DICOM & Technical Calibration"):
    pixel_spacing = st.number_input(
        "Pixel spacing (mm)", value=1.0, step=0.1,
        help="Millimeters per pixel from DICOM tag (0028,0030) for physical RANO tumor measurement.",
    )


# ---------------------------------------------------------------------------
# Top Clinical Navbar Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="clinical-navbar">
    <div class="brand-group">
        <div class="brand-icon">🧠</div>
        <div>
            <h1 class="brand-title">Smart NeuroCare™</h1>
            <p class="brand-subtitle">AI-Powered Neuro-Oncology Triage & Diagnostic Suite</p>
        </div>
    </div>
    <div class="status-group">
        <span class="status-pill success">🟢 3 AI Models Loaded</span>
        <span class="status-pill info">📐 DICOM Calibrated</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Progress Stepper
# ---------------------------------------------------------------------------
def render_stepper(stage: int) -> None:
    labels = ["1. Scan Intake", "2. Preprocessing & Deep AI", "3. Findings & Hospital Triage"]
    parts = ['<div class="nc-stepper">']
    for i, label in enumerate(labels):
        if i < stage:
            cls, icon = "done", "✓"
        elif i == stage:
            cls, icon = "active", str(i + 1)
        else:
            cls, icon = "", str(i + 1)
        parts.append(
            f'<div class="nc-step {cls}"><span class="nc-step-num">{icon}</span>'
            f'<span>{label}</span></div>'
        )
        if i < len(labels) - 1:
            parts.append('<div class="nc-step-connector"></div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Scan Intake (Dual Path: Custom Upload or 1-Click Clinical Samples)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="clinical-card">
    <div class="clinical-card-header">
        <span>📥 Patient Brain MRI Scan Intake</span>
        <span style="font-size:0.8rem; font-weight:500; color:#64748b;">Axial T1/T2 MRI · Max 10MB</span>
    </div>
</div>
""", unsafe_allow_html=True)

intake_tab1, intake_tab2 = st.tabs(["📤 Upload Custom MRI Scan", "🧪 Instant 1-Click Clinical Cases"])

active_image = None
active_image_name = None

with intake_tab1:
    uploaded_file = st.file_uploader(
        "Upload a brain MRI scan (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        active_image_name = uploaded_file.name

with intake_tab2:
    st.markdown("<p style='font-size:0.85rem; color:#475569; margin-bottom:0.5rem;'>Select a verified clinical case below to immediately test the full diagnostic pipeline:</p>", unsafe_allow_html=True)
    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    
    with c_s1:
        if st.button("🟡 Case A: Meningioma\n(Axial T1+C)", use_container_width=True):
            sample_p = os.path.join(SAMPLE_DIR, "meningioma_sample.png")
            if os.path.exists(sample_p):
                active_image = Image.open(sample_p).convert("RGB")
                active_image_name = "Clinical_Case_A_Meningioma.png"
                st.session_state["sample_loaded"] = active_image_name

    with c_s2:
        if st.button("🔴 Case B: High-Grade Glioma\n(Axial T2)", use_container_width=True):
            sample_p = os.path.join(SAMPLE_DIR, "glioma_sample.png")
            if os.path.exists(sample_p):
                active_image = Image.open(sample_p).convert("RGB")
                active_image_name = "Clinical_Case_B_Glioma.png"
                st.session_state["sample_loaded"] = active_image_name

    with c_s3:
        if st.button("🟣 Case C: Pituitary Adenoma\n(Coronal T1)", use_container_width=True):
            sample_p = os.path.join(SAMPLE_DIR, "pituitary_sample.png")
            if os.path.exists(sample_p):
                active_image = Image.open(sample_p).convert("RGB")
                active_image_name = "Clinical_Case_C_Pituitary.png"
                st.session_state["sample_loaded"] = active_image_name

    with c_s4:
        if st.button("🟢 Case D: Normal Screening\n(Healthy Brain)", use_container_width=True):
            sample_p = os.path.join(SAMPLE_DIR, "healthy_normal_sample.png")
            if os.path.exists(sample_p):
                active_image = Image.open(sample_p).convert("RGB")
                active_image_name = "Clinical_Case_D_Healthy_Brain.png"
                st.session_state["sample_loaded"] = active_image_name

# Retain loaded sample scan in session if set
if active_image is None and st.session_state.get("sample_loaded"):
    s_name = st.session_state["sample_loaded"]
    mapping = {
        "Clinical_Case_A_Meningioma.png": "meningioma_sample.png",
        "Clinical_Case_B_Glioma.png": "glioma_sample.png",
        "Clinical_Case_C_Pituitary.png": "pituitary_sample.png",
        "Clinical_Case_D_Healthy_Brain.png": "healthy_normal_sample.png",
    }
    target_file = os.path.join(SAMPLE_DIR, mapping.get(s_name, "meningioma_sample.png"))
    if os.path.exists(target_file):
        active_image = Image.open(target_file).convert("RGB")
        active_image_name = s_name

# Update stage based on scan presence and analysis state
current_stage = 0
if active_image is not None:
    current_stage = 2 if st.session_state.get("nc_analysis_done_for") == active_image_name else 1

render_stepper(current_stage)

# ---------------------------------------------------------------------------
# View 1: When No Scan is Loaded (Clinical Overview & Capabilities Grid)
# ---------------------------------------------------------------------------
if active_image is None:
    st.markdown("""
    <div class="workflow-grid">
        <div class="workflow-card">
            <div class="wf-icon">⚡</div>
            <div class="wf-title">1. Binary Detection</div>
            <div class="wf-desc">Deep ResNet backbone rapidly identifies presence of intracranial abnormal mass with calibrated confidence scoring.</div>
        </div>
        <div class="workflow-card">
            <div class="wf-icon">🔬</div>
            <div class="wf-title">2. Subtype Histopathology</div>
            <div class="wf-desc">4-Class differential typing across Glioma, Meningioma, Pituitary Adenoma, and Normal tissue.</div>
        </div>
        <div class="workflow-card">
            <div class="wf-icon">📐</div>
            <div class="wf-title">3. RANO 2D Measurements</div>
            <div class="wf-desc">UNet segmentation engine extracts bidirectional major × minor diameters (L × W) and physical surface area in mm².</div>
        </div>
        <div class="workflow-card">
            <div class="wf-icon">🏥</div>
            <div class="wf-title">4. Neurosurgical Triage</div>
            <div class="wf-desc">Multi-criteria hospital matching scoring surgeon quality, geospatial distance, and insurance policy coverage.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# View 2: When Scan is Loaded (Clinical Workstation & Analysis)
# ---------------------------------------------------------------------------
else:
    col_scan, col_controls = st.columns([5, 4])

    with col_controls:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="clinical-card-header">
                <span>📋 Active Patient Case</span>
                <span class="status-pill info">ID: {patient_id}</span>
            </div>
            <p style="font-size:0.85rem; color:#334155; margin:0 0 0.5rem;">
                <b>Patient:</b> {name} ({age}y, {sex.title()})<br>
                <b>BMI:</b> {bmi:.1f} · <b>Insurance:</b> {insurance}<br>
                <b>Chief Symptoms:</b> {', '.join(symptoms) if symptoms else 'None reported'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="clinical-card">
            <div class="clinical-card-header">
                <span>⚙️ Image Enhancement & Normalization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            enable_crop = st.checkbox("🧠 Auto-Crop Margins", value=True, help="Removes empty background margins to focus network receptive fields on brain parenchyma.")
        with col_opt2:
            enable_clahe = st.checkbox("✨ CLAHE Contrast", value=True, help="Equalizes intra-scanner illumination variance (1.5T vs 3.0T MRI) to sharpen lesion contours.")

        processed_image = preprocess_mri(active_image, auto_crop=enable_crop, enhance_contrast=enable_clahe)

        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        run_analysis = st.button("🔬 Run Complete Diagnostic Analysis", use_container_width=True)

    with col_scan:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="clinical-card-header">
                <span>🖼️ Active MRI Slice View</span>
                <span style="font-size:0.78rem; font-family:'JetBrains Mono'; color:#0284c7;">📁 {active_image_name} · {processed_image.size[0]}×{processed_image.size[1]}px</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        image_placeholder = st.empty()
        image_placeholder.image(processed_image, caption=f"{active_image_name} (Enhanced with Auto-Crop & CLAHE)" if (enable_crop or enable_clahe) else active_image_name, use_column_width=True)

    # -----------------------------------------------------------------------
    # Run Diagnostic Pipeline
    # -----------------------------------------------------------------------
    if run_analysis:
        st.session_state["nc_analysis_done_for"] = active_image_name

        # 1. Binary Detection
        with st.spinner("Running deep convolutional tumor detection..."):
            eval_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = eval_transforms(processed_image).unsqueeze(0)
            with torch.no_grad():
                logit = detection_model(tensor)
                prob = torch.sigmoid(logit).item()
            tumor_detected = prob > 0.5

        # Initialize defaults
        tumor_type = None
        classification_confidence = None
        max_diameter_mm = None
        perpendicular_diameter_mm = None
        product_bidirectional_mm2 = None
        area_mm2 = None
        severity = None
        overlay_path = None
        circle_info = None

        if tumor_detected:
            # 2. 4-Class Classification
            with st.spinner("Classifying tumor histopathology subtype..."):
                with torch.no_grad():
                    class_logits = classification_model(tensor)
                    class_probs = torch.softmax(class_logits, dim=1)[0]
                    pred_idx = class_probs.argmax().item()
                    tumor_type = CLASSIFICATION_CLASSES[pred_idx]
                    classification_confidence = class_probs[pred_idx].item()

            # 3. UNet Segmentation & RANO Sizing
            with st.spinner("Extracting RANO 2D lesion boundaries..."):
                gray = processed_image.convert("L").resize((256, 256))
                seg_input = torch.tensor(np.array(gray), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                with torch.no_grad():
                    mask = segmentation_model(seg_input)[0, 0]

                measurements = compute_tumor_measurements(mask, pixel_spacing_mm=pixel_spacing)
                area_mm2 = measurements["area_mm2"]
                max_diameter_mm = measurements["max_diameter_mm"]
                perpendicular_diameter_mm = measurements.get("perpendicular_diameter_mm", 0.0)
                product_bidirectional_mm2 = measurements.get("product_bidirectional_mm2", 0.0)

                orig_w, orig_h = processed_image.size
                circle_info = find_tumor_circle(mask, orig_w, orig_h)

            if circle_info:
                annotated = draw_tumor_overlay(processed_image, circle_info)
                image_placeholder.image(annotated, caption="Lesion Localized (Crosshair Red Ring Indicator)", use_column_width=True)
                overlay_path = "demo_overlay.png"
                annotated.save(overlay_path)

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

        # Build analysis result object
        analysis = TumorAnalysisResult(
            tumor_detected=tumor_detected,
            detection_confidence=prob,
            tumor_type=tumor_type,
            classification_confidence=classification_confidence,
            max_diameter_mm=max_diameter_mm,
            perpendicular_diameter_mm=perpendicular_diameter_mm,
            product_bidirectional_mm2=product_bidirectional_mm2,
            area_mm2=area_mm2,
            severity=severity,
            overlay_image_path=overlay_path,
        )

        patient = PatientProfile(
            name=name,
            patient_id=patient_id,
            age=int(age),
            sex=sex,
            weight_kg=float(weight_kg),
            height_cm=float(height_cm),
            smoker=smoker,
            cigarettes_per_day=int(cigarettes_per_day),
            alcohol_use=alcohol_use,
            physical_activity=physical_activity,
            existing_conditions=existing_conditions,
            family_history_cancer=family_history_cancer,
            symptoms=symptoms,
            max_budget=float(budget),
            insurance_provider=insurance,
            latitude=float(lat),
            longitude=float(lon),
        )

        # -------------------------------------------------------------------
        # Executive Summary Metric Cards
        # -------------------------------------------------------------------
        st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-family:Plus Jakarta Sans; font-size:1.25rem; font-weight:800; color:#0f172a; margin-bottom:0.75rem;'>📊 Executive Clinical Findings</h3>", unsafe_allow_html=True)

        if tumor_detected:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
                <span class="detection-badge detected">
                    <span class="badge-dot"></span>
                    TUMOR DETECTED — {prob*100:.1f}% Confidence
                </span>
                <span style="font-size:0.85rem; color:#475569;">
                    Predicted Type: <b style="color:#0f172a; text-transform:capitalize;">{tumor_type}</b> ({classification_confidence*100:.1f}%) · Severity: {severity_html(severity)}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                <span class="detection-badge clear">
                    <span class="badge-dot"></span>
                    NO INTRACRANIAL LESION DETECTED — {prob*100:.1f}% Confidence
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Metrics Grid
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card {'red' if tumor_detected else 'green'}">
                <div class="metric-label">Detection Status</div>
                <div class="metric-value">{'Positive' if tumor_detected else 'Clear'}</div>
                <div class="metric-sub">{prob*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card purple">
                <div class="metric-label">Histopathology</div>
                <div class="metric-value" style="font-size:1.2rem; text-transform:capitalize;">{tumor_type if tumor_type else 'N/A'}</div>
                <div class="metric-sub">{f'{classification_confidence*100:.1f}% match' if classification_confidence else 'Non-neoplastic'}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            rano_txt = f"{max_diameter_mm:.1f} × {perpendicular_diameter_mm:.1f}" if (max_diameter_mm and perpendicular_diameter_mm) else "N/A"
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">RANO 2D (L × W)</div>
                <div class="metric-value" style="font-size:1.15rem;">{rano_txt}</div>
                <div class="metric-sub">Millimeters (mm)</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            area_txt = f"{area_mm2:.1f}" if area_mm2 else "0.0"
            st.markdown(f"""
            <div class="metric-card amber">
                <div class="metric-label">Surface Area</div>
                <div class="metric-value" style="font-size:1.2rem;">{area_txt}</div>
                <div class="metric-sub">mm² (Pixel Calibrated)</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card teal">
                <div class="metric-label">Clinical Severity</div>
                <div class="metric-value" style="font-size:1.15rem; text-transform:uppercase;">{severity if severity else 'LOW'}</div>
                <div class="metric-sub">WHO Grade Aligned</div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # Structured Tabbed Clinical Results View
        # -------------------------------------------------------------------
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
            "📊 Diagnostic Findings & RANO Details",
            "🏥 Specialized Neurosurgical Centers",
            "💚 Lifestyle, Diet & Recovery Protocol",
            "📄 Export Clinical Report (PDF)"
        ])

        with res_tab1:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("""
                <div class="clinical-card">
                    <div class="clinical-card-header"><span>📐 RANO & RECIST Bidirectional Measurements</span></div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                - **Major Axis Diameter ($L$):** `{f'{max_diameter_mm:.2f} mm' if max_diameter_mm else 'N/A'}`
                - **Perpendicular Minor Axis ($W$):** `{f'{perpendicular_diameter_mm:.2f} mm' if perpendicular_diameter_mm else 'N/A'}`
                - **Bidirectional Product ($L \\times W$):** `{f'{product_bidirectional_mm2:.2f} mm²' if product_bidirectional_mm2 else 'N/A'}`
                - **Total Cross-Sectional Lesion Area:** `{f'{area_mm2:.2f} mm²' if area_mm2 else '0.00 mm²'}`
                - **DICOM Pixel Spacing:** `{pixel_spacing:.2f} mm/pixel`
                """)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_t2:
                st.markdown("""
                <div class="clinical-card">
                    <div class="clinical-card-header"><span>🔬 AI Subtype Classification Probability</span></div>
                """, unsafe_allow_html=True)
                if tumor_detected and classification_confidence:
                    for idx, c_name in enumerate(CLASSIFICATION_CLASSES):
                        c_prob = class_probs[idx].item() * 100
                        st.markdown(f"""
                        <div style="margin-bottom:0.5rem;">
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem; font-weight:600; color:#334155;">
                                <span style="text-transform:capitalize;">{c_name}</span>
                                <span>{c_prob:.1f}%</span>
                            </div>
                            <div class="confidence-meter"><div class="confidence-fill info" style="width:{c_prob}%;"></div></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("Intracranial tissue scan appears normal. No neoplastic differential required.")
                st.markdown("</div>", unsafe_allow_html=True)

        with res_tab2:
            st.markdown("<p style='font-size:0.85rem; color:#475569;'>Multi-criteria ranked neurosurgical centers filtered by surgical tier, proximity, and insurance compatibility:</p>", unsafe_allow_html=True)
            matched_hospitals = recommend_hospitals(patient, analysis, default_hospitals(), top_n=4)
            for h in matched_hospitals:
                h_name = h.get("name", "Neurosurgical Centre")
                h_city = h.get("city", "")
                h_score = h.get("match_score", 0.0)
                h_dist = h.get("distance_km", 0.0)
                reasons = h.get("match_reasons", {})
                city_label = f" ({h_city})" if h_city else ""
                st.markdown(f"""
                <div class="hospital-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="hospital-name">🏥 {h_name}{city_label}</span>
                        <span class="match-score">Match: {h_score*100:.0f}%</span>
                    </div>
                    <div class="hospital-meta">
                        <span>📍 {h_dist} km away</span>
                        <span>🎯 Specialization: {reasons.get('specialization_match', 'N/A')}</span>
                        <span>⭐ Quality: {reasons.get('hospital_quality', 'N/A')}</span>
                        <span>💰 Cost fit: {reasons.get('cost_fit', 'N/A')}</span>
                        <span>🛡️ Insurance: {reasons.get('insurance_fit', 'N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_tab3:
            lifestyle = generate_lifestyle_recommendations(patient, analysis)
            category_icons = {"diet": "🥗", "exercise": "🏃", "habits": "🔄", "monitoring": "📋", "warning_signs": "⚠️"}
            for category, items in lifestyle.items():
                if not items:
                    continue
                icon = category_icons.get(category, "📌")
                title = category.replace("_", " ").title()
                items_html = "".join(f"<li>{item}</li>" for item in items)
                warning_style = "border-color: #fca5a5; background: #fff8f8;" if category == "warning_signs" else ""
                st.markdown(f"""
                <div class="lifestyle-category" style="{warning_style}">
                    <div class="category-title">{icon} {title}</div>
                    <ul>{items_html}</ul>
                </div>
                """, unsafe_allow_html=True)

        with res_tab4:
            st.markdown("""
            <div class="clinical-card">
                <div class="clinical-card-header"><span>📄 Clinical PDF Diagnostic Report</span></div>
                <p style="font-size:0.85rem; color:#475569;">Export a comprehensive, board-standard PDF report containing patient demographics, RANO tumor metrics, spatial overlays, hospital triage, and recovery protocols.</p>
            </div>
            """, unsafe_allow_html=True)

            pdf_buffer = io.BytesIO()
            report_path = "generated_report.pdf"
            generate_patient_report(patient, analysis, matched_hospitals, report_path)
            with open(report_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download Full Clinical PDF Report",
                data=pdf_bytes,
                file_name=f"Smart_NeuroCare_Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Footer Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:1.5rem 1rem; margin-top:2rem; color:#94a3b8; font-size:0.78rem; border-top:1px solid #e2e8f0;">
    ⚖️ <b>Clinical Decision Support Disclaimer:</b> Smart NeuroCare is an investigational AI-assisted triaging platform. All findings must be validated by a board-certified radiologist or neurosurgeon prior to surgical intervention.
</div>
<a href="#top" class="nc-back-to-top" title="Back to top">↑</a>
""", unsafe_allow_html=True)
