"""
Streamlit Web Application — Smart NeuroCare
Enterprise-Grade AI Neuro-Oncology Clinical Decision Support (CDS) Suite

State-Driven Clinical Workstation:
  - State 1: Scan & Patient Intake (Clean dual-path intake with case presets)
  - State 2: Comprehensive Diagnostic Results with clear '⬅ Back to Intake' & '🔄 New Scan' navigation

Differentiating feature: Longitudinal Treatment-Response Tracking. Every analyzed
scan is persisted as a "visit" for the entered Patient ID (patient_history.py,
SQLite). Once a patient has 2+ visits, treatment_response.py compares the
current measurement against baseline/previous/nadir and produces a
measurement-based, RANO-inspired response assessment — decision-support only,
not a diagnostic system, not clinically validated. See the "Treatment
Response" tab and its disclaimer for the exact scope of what this determines.
"""

import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import streamlit as st

from cnn_detection_model import TumorDetectionModel
from train_classification import TumorClassificationModel
from unet_segmentation import UNet, compute_tumor_measurements, find_tumor_circle
from hospital_recommendation import recommend_hospitals, default_hospitals, PatientContext
from patient_lifestyle import (
    PatientDetails,
    TumorAnalysisResult,
    generate_lifestyle_recommendations,
    generate_full_report,
)
from image_preprocessing import preprocess_mri
from patient_history import record_visit, get_visit_history
from treatment_response import (
    classify_response, VisitMeasurement, INSUFFICIENT_DATA, CR, PR, SD, PD,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart NeuroCare — Clinical Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Models
# ---------------------------------------------------------------------------
CLASSIFICATION_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SAMPLE_DIR = "sample_scans"
if not os.path.exists(SAMPLE_DIR):
    from create_sample_scans import generate_sample_mris
    generate_sample_mris(SAMPLE_DIR)

DETECTION_CKPT = "best_detection_model.pt"
CLASSIFICATION_CKPT = "best_classification_model.pt"
SEGMENTATION_CKPT = "best_segmentation_model.pt"
using_trained_weights = all(os.path.exists(p) for p in [DETECTION_CKPT, CLASSIFICATION_CKPT, SEGMENTATION_CKPT])
MODEL_VERSION = "trained-checkpoints-v1" if using_trained_weights else "demo-untrained-v0"


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

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Helper: draw red circle overlay
# ---------------------------------------------------------------------------
def draw_tumor_overlay(image: Image.Image, circle_info: dict) -> Image.Image:
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cx, cy, r = circle_info["center_x"], circle_info["center_y"], circle_info["radius"]
    overlay = img_bgr.copy()
    cv2.circle(overlay, (cx, cy), r + 3, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
    cv2.circle(img_bgr, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)

    tick_len = max(6, r // 5)
    color = (0, 0, 255)
    cv2.line(img_bgr, (cx - r, cy), (cx - r + tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx + r, cy), (cx + r - tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy - r), (cx, cy - r + tick_len), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy + r), (cx, cy + r - tick_len), color, 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Helper: adapt persisted visit rows into treatment_response's input type
# ---------------------------------------------------------------------------
def _history_to_measurements(history) -> list:
    return [
        VisitMeasurement(
            scan_date=v.scan_date,
            tumor_type=v.tumor_type,
            max_diameter_mm=v.max_diameter_mm,
            perpendicular_diameter_mm=v.perpendicular_diameter_mm,
            product_bidirectional_mm2=v.product_bidirectional_mm2,
            area_mm2=v.area_mm2,
            visit_id=v.visit_id,
        )
        for v in history
    ]


RESPONSE_BADGE_STYLE = {
    CR: ("#f0fdf4", "#bbf7d0", "#15803d"),
    PR: ("#f0fdf4", "#bbf7d0", "#15803d"),
    SD: ("#fffbeb", "#fde68a", "#b45309"),
    PD: ("#fef2f2", "#fecaca", "#dc2626"),
}

# ---------------------------------------------------------------------------
# Custom CSS — Clean Medical Workstation
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #f8fafc;
    --border-color: #e2e8f0;
    --accent-primary: #0284c7;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
}

.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }

.block-container {
    padding: 0.6rem 1rem 1.5rem !important;
    max-width: 1440px !important;
}

/* ── Top Navbar ── */
.clinical-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.clinical-navbar .brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.01em;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}

.status-pill.success { background: #f0fdf4; border: 1px solid #bbf7d0; color: #15803d; }
.status-pill.info { background: #f0f9ff; border: 1px solid #bae6fd; color: #0369a1; }
.status-pill.warn { background: #fffbeb; border: 1px solid #fde68a; color: #b45309; }

/* ── Panels ── */
.pacs-panel {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.pacs-panel-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 0.35rem;
}

/* ── Result Banners ── */
.result-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.85rem;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.result-banner.detected {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
}

.result-banner.clear {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
}

/* ── Metric Matrix ── */
.metric-matrix {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.45rem;
    margin-bottom: 0.65rem;
}

.matrix-tile {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 0.4rem 0.55rem;
    border-top: 3px solid #0284c7;
}

.matrix-tile.red { border-top-color: #dc2626; }
.matrix-tile.green { border-top-color: #16a34a; }
.matrix-tile.purple { border-top-color: #6366f1; }
.matrix-tile.blue { border-top-color: #0284c7; }
.matrix-tile.amber { border-top-color: #d97706; }
.matrix-tile.teal { border-top-color: #0d9488; }

.matrix-tile .t-label {
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    margin-bottom: 0.1rem;
}

.matrix-tile .t-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.98rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.15;
}

.matrix-tile .t-sub {
    font-size: 0.66rem;
    color: var(--text-secondary);
    margin-top: 0.1rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #0284c7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.45rem 1.1rem !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #0369a1 !important;
}

.stDownloadButton > button {
    background: #f0f9ff !important;
    color: #0369a1 !important;
    border: 1px solid #bae6fd !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    font-size: 0.82rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 30px;
    border-radius: 4px;
    padding: 0 10px;
    font-weight: 600;
    font-size: 0.78rem;
}

.stTabs [aria-selected="true"] {
    background-color: #f0f9ff !important;
    color: #0284c7 !important;
    border: 1px solid #bae6fd !important;
}

/* ── Response Assessment ── */
.response-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.85rem;
    border-radius: 6px;
    font-weight: 800;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}

.visit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    margin-bottom: 0.5rem;
}

.visit-table th, .visit-table td {
    border: 1px solid #e2e8f0;
    padding: 0.35rem 0.5rem;
    text-align: left;
}

.visit-table th {
    background: #f8fafc;
    font-weight: 700;
    color: #334155;
}

.demo-tag {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
    border-radius: 4px;
    padding: 0.05rem 0.35rem;
    font-size: 0.68rem;
    font-weight: 700;
    margin-left: 0.4rem;
}

.caveat-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    font-size: 0.76rem;
    color: #78350f;
    margin-bottom: 0.5rem;
}

.disclaimer-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    font-size: 0.72rem;
    color: #64748b;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: Patient Profile
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🗂️ Patient Case Record")

col_p1, col_p2, col_p3 = st.sidebar.columns(3)
if col_p1.button("👩 45F", help="Jane Doe (45F, Frontal)"):
    st.session_state["p_name"] = "Jane Doe"
    st.session_state["p_id"] = "P-10293"
    st.session_state["p_age"] = 45
    st.session_state["p_sex"] = "female"
    st.session_state["p_symptoms"] = ["headaches", "vision changes"]

if col_p2.button("👨 62M", help="Robert K. (62M, Seizures)"):
    st.session_state["p_name"] = "Robert King"
    st.session_state["p_id"] = "P-88412"
    st.session_state["p_age"] = 62
    st.session_state["p_sex"] = "male"
    st.session_state["p_symptoms"] = ["seizures", "memory issues"]

if col_p3.button("👦 28M", help="Alex M. (28M, Clear)"):
    st.session_state["p_name"] = "Alex Miller"
    st.session_state["p_id"] = "P-33291"
    st.session_state["p_age"] = 28
    st.session_state["p_sex"] = "male"
    st.session_state["p_symptoms"] = ["none"]

with st.sidebar.expander("🪪 Demographics & Vitals", expanded=True):
    name = st.text_input("Full Name", st.session_state.get("p_name", "Jane Doe"))
    patient_id = st.text_input("Patient ID", st.session_state.get("p_id", "P-10293"))
    age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("p_age", 45))
    sex = st.selectbox("Sex", ["female", "male", "other"], index=0 if st.session_state.get("p_sex") == "female" else 1)
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
    height_cm = st.number_input("Height (cm)", min_value=30.0, value=165.0)

bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm else 0
st.sidebar.caption(f"📋 {name} · {patient_id} · {age}y {sex} · BMI {bmi:.1f}")

with st.sidebar.expander("🏥 Clinical History & Symptoms"):
    existing_conditions = st.multiselect(
        "Comorbidities",
        ["diabetes", "hypertension", "heart disease", "asthma", "epilepsy", "none"],
        default=["none"],
    )
    family_history_cancer = st.checkbox("Family history of oncology")
    default_syms = st.session_state.get("p_symptoms", ["headaches"])
    symptoms = st.multiselect(
        "Reported Symptoms",
        ["headaches", "seizures", "vision changes", "balance issues", "nausea", "memory issues", "none"],
        default=[s for s in default_syms if s in ["headaches", "seizures", "vision changes", "balance issues", "nausea", "memory issues", "none"]] or ["none"],
    )

with st.sidebar.expander("📍 Location & Insurance"):
    budget = st.number_input("Max Budget (₹)", value=800000, step=50000)
    insurance = st.text_input("Insurance Provider", "StarHealth")
    lat = st.number_input("Latitude", value=12.9716, format="%.4f")
    lon = st.number_input("Longitude", value=77.5946, format="%.4f")
    pixel_spacing = st.number_input("DICOM Pixel Spacing (mm)", value=1.0, step=0.1)

visit_count = len(get_visit_history(patient_id)) if patient_id.strip() else 0
st.sidebar.caption(f"🕒 {visit_count} prior visit(s) on record for this Patient ID")


# ---------------------------------------------------------------------------
# View State Management (Intake vs Results)
# ---------------------------------------------------------------------------
if "view_state" not in st.session_state:
    st.session_state["view_state"] = "intake"

model_status_pill = (
    '<span class="status-pill success">🟢 Trained Checkpoints Loaded</span>'
    if using_trained_weights else
    '<span class="status-pill warn">⚠ Demo Weights — Not Clinically Meaningful</span>'
)

st.markdown(f"""
<div class="clinical-navbar">
    <div style="display:flex; align-items:center; gap:0.4rem;">
        <span style="font-size:1.2rem;">🧠</span>
        <span class="brand-title">Smart NeuroCare™</span>
        <span style="font-size:0.72rem; color:#0284c7; background:#f0f9ff; border:1px solid #bae6fd; padding:0.15rem 0.45rem; border-radius:4px; font-weight:600;">Clinical Decision Support v2.5</span>
    </div>
    <div style="display:flex; align-items:center; gap:0.4rem;">
        {model_status_pill}
        <span class="status-pill info">📐 DICOM Calibrated</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# STATE 1: SCAN & PATIENT INTAKE
# ---------------------------------------------------------------------------
if st.session_state["view_state"] == "intake":
    col_in_left, col_in_right = st.columns([1, 1])

    with col_in_left:
        st.markdown("""
        <div class="pacs-panel">
            <div class="pacs-panel-header">
                <span>📥 1. Select Brain MRI Scan Source</span>
                <span style="font-size:0.72rem; color:#64748b;">Axial T1/T2 & FLAIR</span>
            </div>
        """, unsafe_allow_html=True)

        intake_mode = st.radio(
            "Scan Source",
            ["🧪 Verified Clinical Demo Cases", "📤 Upload Custom DICOM/MRI"],
            horizontal=True,
            label_visibility="collapsed",
        )

        active_img = None
        active_img_name = None
        active_img_is_demo = False

        if intake_mode == "🧪 Verified Clinical Demo Cases":
            # Single source of truth for label <-> filename, so a label typo can
            # never desync from the lookup table (this caused a KeyError before).
            case_map = {
                "Case 1: Frontal Meningioma (Axial T1+C)": "meningioma_sample.png",
                "Case 2: Temporal High-Grade Glioma (Axial T2)": "glioma_sample.png",
                "Case 3: Sellar Pituitary Macroadenoma (Coronal T1)": "pituitary_sample.png",
                "Case 4: Healthy Brain Screening (Normal)": "healthy_normal_sample.png",
                "Case 5 (DEMO — Simulated Follow-up): Glioma, smaller vs. Case 2": "glioma_followup_smaller_demo.png",
                "Case 6 (DEMO — Simulated Follow-up): Glioma, larger vs. Case 2": "glioma_followup_larger_demo.png",
            }
            demo_case = st.selectbox("Clinical Demo Slices", list(case_map.keys()))
            if "DEMO" in demo_case:
                st.caption("⚠ SIMULATED FOLLOW-UP — a synthetically resized lesion for demonstrating longitudinal tracking only. Not a real patient scan.")
            f_name = case_map[demo_case]
            target_path = os.path.join(SAMPLE_DIR, f_name)
            if os.path.exists(target_path):
                active_img = Image.open(target_path).convert("RGB")
                active_img_name = f_name
                active_img_is_demo = "_demo" in f_name

        else:
            up_file = st.file_uploader("Upload MRI Slice", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            if up_file is not None:
                active_img = Image.open(up_file).convert("RGB")
                active_img_name = up_file.name
                active_img_is_demo = False

        st.markdown("</div>", unsafe_allow_html=True)

        if active_img is not None:
            st.session_state["loaded_image"] = active_img
            st.session_state["loaded_image_name"] = active_img_name
            st.session_state["loaded_image_is_demo"] = active_img_is_demo

            st.markdown(f"""
            <div class="pacs-panel">
                <div class="pacs-panel-header">
                    <span>🖼️ Active Scan Preview</span>
                    <span style="font-size:0.7rem; color:#0284c7; font-family:'JetBrains Mono';">{active_img_name}</span>
                </div>
            """, unsafe_allow_html=True)

            col_pr1, col_pr2 = st.columns(2)
            with col_pr1:
                enable_crop = st.checkbox("🧠 Auto-Crop Margins", value=True)
            with col_pr2:
                enable_clahe = st.checkbox("✨ CLAHE Contrast", value=True)

            processed_preview = preprocess_mri(active_img, auto_crop=enable_crop, enhance_contrast=enable_clahe)
            st.session_state["processed_image"] = processed_preview

            col_sub1, col_sub2 = st.columns([1, 1])
            with col_sub1:
                st.image(processed_preview, width=220)
            with col_sub2:
                st.markdown(f"""
                <div style="font-size:0.8rem; color:#334155; line-height:1.6; margin-top:0.5rem;">
                    <b>Resolution:</b> {processed_preview.size[0]}×{processed_preview.size[1]}px<br>
                    <b>Scale:</b> {pixel_spacing:.2f} mm/px<br>
                    <b>Enhancement:</b> {'Active' if (enable_crop or enable_clahe) else 'Raw'}
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔬 Execute Diagnostic Analysis ➔", use_container_width=True):
                    st.session_state["view_state"] = "results"
                    st.session_state["analysis_nonce"] = str(uuid.uuid4())
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    with col_in_right:
        st.markdown(f"""
        <div class="pacs-panel">
            <div class="pacs-panel-header">
                <span>📋 2. Patient Case Briefing & Protocol</span>
                <span class="status-pill info">ID: {patient_id}</span>
            </div>
            <div style="font-size:0.82rem; color:#334155; line-height:1.7; margin-bottom:0.75rem;">
                <b>Full Name:</b> {name} ({age}y, {sex.title()})<br>
                <b>BMI Index:</b> {bmi:.1f} ({'Normal range' if 18.5<=bmi<25 else 'Elevated range'})<br>
                <b>Insurance Provider:</b> {insurance} (Max Budget: ₹{budget:,.0f})<br>
                <b>Reported Symptoms:</b> {', '.join(symptoms) if symptoms else 'None'}<br>
                <b>Comorbidities:</b> {', '.join(existing_conditions) if existing_conditions else 'None'}<br>
                <b>Visit history:</b> {visit_count} prior visit(s) recorded for this Patient ID
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pacs-panel">
            <div class="pacs-panel-header">
                <span>⚡ Automated Triaging Capabilities</span>
                <span style="font-size:0.72rem; color:#64748b;">Latency: ~0.6s</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.78rem;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.55rem;">
                    <b>1. Binary Detection</b><br>
                    <span style="color:#64748b;">High-sensitivity CNN backbone flags neoplastic mass presence.</span>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.55rem;">
                    <b>2. Differential Typing</b><br>
                    <span style="color:#64748b;">Classifies Glioma, Meningioma, Pituitary, or Normal tissue.</span>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.55rem;">
                    <b>3. RANO 2D Sizing</b><br>
                    <span style="color:#64748b;">Measures major × minor diameters and lesion surface area in mm².</span>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.55rem;">
                    <b>4. Geospatial Routing</b><br>
                    <span style="color:#64748b;">Multi-criteria hospital matching scoring surgeon quality & insurance.</span>
                </div>
                <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:6px; padding:0.55rem; grid-column: span 2;">
                    <b>5. Longitudinal Response Tracking (NEW)</b><br>
                    <span style="color:#64748b;">Compares this Patient ID's measurements across visits against baseline / previous / nadir and produces a measurement-based, RANO-inspired response assessment — decision-support only.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# STATE 2: COMPREHENSIVE DIAGNOSTIC RESULTS
# ---------------------------------------------------------------------------
elif st.session_state["view_state"] == "results":
    processed_img = st.session_state.get("processed_image")
    img_name = st.session_state.get("loaded_image_name", "Scan.png")
    img_is_demo = st.session_state.get("loaded_image_is_demo", False)

    if processed_img is None:
        st.session_state["view_state"] = "intake"
        st.rerun()

    # Back Button Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns([2.5, 5, 2.5])
    with nav_col1:
        if st.button("⬅ Back to Scan Intake", use_container_width=True):
            st.session_state["view_state"] = "intake"
            st.rerun()
    with nav_col2:
        st.markdown(f"<div style='text-align:center; font-size:0.85rem; font-weight:700; color:#0f172a; padding-top:0.4rem;'>Patient: {name} · ID: {patient_id} · Scan: {img_name}</div>", unsafe_allow_html=True)
    with nav_col3:
        if st.button("🔄 New Case / Reset", use_container_width=True):
            st.session_state["view_state"] = "intake"
            st.session_state.pop("loaded_image", None)
            st.session_state.pop("loaded_image_name", None)
            st.rerun()

    # ----- Run Analysis Computations -----
    tensor = EVAL_TRANSFORMS(processed_img).unsqueeze(0)
    with torch.no_grad():
        logit = detection_model(tensor)
        prob = torch.sigmoid(logit).item()
    tumor_detected = prob > 0.5

    tumor_type = None
    classification_confidence = None
    class_probs = None
    max_diameter_mm = None
    perpendicular_diameter_mm = None
    product_bidirectional_mm2 = None
    area_mm2 = None
    severity = None
    overlay_path = None
    circle_info = None
    annotated = None

    if tumor_detected:
        with torch.no_grad():
            class_logits = classification_model(tensor)
            class_probs = torch.softmax(class_logits, dim=1)[0]
            pred_idx = class_probs.argmax().item()
            tumor_type = CLASSIFICATION_CLASSES[pred_idx]
            classification_confidence = class_probs[pred_idx].item()

        gray = processed_img.convert("L").resize((256, 256))
        seg_input = torch.tensor(np.array(gray), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            mask = segmentation_model(seg_input)[0, 0]

        measurements = compute_tumor_measurements(mask, pixel_spacing_mm=pixel_spacing)
        area_mm2 = measurements["area_mm2"]
        max_diameter_mm = measurements["max_diameter_mm"]
        perpendicular_diameter_mm = measurements.get("perpendicular_diameter_mm", 0.0)
        product_bidirectional_mm2 = measurements.get("product_bidirectional_mm2", 0.0)

        orig_w, orig_h = processed_img.size
        circle_info = find_tumor_circle(mask, orig_w, orig_h)

        if circle_info:
            annotated = draw_tumor_overlay(processed_img, circle_info)
            overlay_path = "demo_overlay.png"
            annotated.save(overlay_path)

        if area_mm2 is not None and area_mm2 > 0:
            if area_mm2 < 200: severity = "low"
            elif area_mm2 < 500: severity = "moderate"
            elif area_mm2 < 1000: severity = "high"
            else: severity = "critical"
        else:
            severity = "moderate"

    visit_scan_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    analysis = TumorAnalysisResult(
        tumor_detected=tumor_detected,
        detection_confidence=prob,
        tumor_type=tumor_type,
        classification_confidence=classification_confidence,
        tumor_area_mm2=area_mm2,
        max_diameter_mm=max_diameter_mm,
        perpendicular_diameter_mm=perpendicular_diameter_mm,
        product_bidirectional_mm2=product_bidirectional_mm2,
        severity_score=severity,
        segmentation_overlay_path=overlay_path,
        model_version=MODEL_VERSION,
        scan_date=visit_scan_date,
    )

    patient = PatientDetails(
        name=name,
        patient_id=patient_id,
        age=int(age),
        sex=sex,
        weight_kg=float(weight_kg),
        height_cm=float(height_cm),
        smoker=False,
        cigarettes_per_day=0,
        alcohol_use="none",
        physical_activity="moderate",
        existing_conditions=existing_conditions,
        family_history_cancer=family_history_cancer,
        symptoms=symptoms,
    )

    patient_ctx = PatientContext(
        latitude=float(lat), longitude=float(lon),
        tumor_type=tumor_type or "notumor",
        severity_score=severity or "low",
        max_budget=float(budget), insurance_provider=insurance,
    )

    # ----- Persist this analysis as one visit (exactly once per button click) -----
    current_nonce = st.session_state.get("analysis_nonce")
    if patient_id.strip() and current_nonce and st.session_state.get("last_recorded_nonce") != current_nonce:
        record_visit(
            patient_id=patient_id,
            scan_date=visit_scan_date,
            tumor_type=tumor_type,
            max_diameter_mm=max_diameter_mm,
            perpendicular_diameter_mm=perpendicular_diameter_mm,
            product_bidirectional_mm2=product_bidirectional_mm2,
            area_mm2=area_mm2,
            severity_score=severity,
            overlay_path=overlay_path,
            is_demo=img_is_demo,
        )
        st.session_state["last_recorded_nonce"] = current_nonce

    visit_history = get_visit_history(patient_id) if patient_id.strip() else []
    response_result = classify_response(_history_to_measurements(visit_history)) if visit_history else None

    # 1. Executive Result Alert Banner
    if tumor_detected:
        st.markdown(f"""
        <div class="result-banner detected">
            <span>🔴 INTRACRANIAL LESION DETECTED — {prob*100:.1f}% Conf · Subtype: <b style="text-transform:capitalize;">{tumor_type}</b> ({classification_confidence*100:.1f}%)</span>
            <span style="font-size:0.72rem; text-transform:uppercase; background:#b91c1c; color:#fff; padding:0.15rem 0.45rem; border-radius:4px;">Severity: {severity}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner clear">
            <span>🟢 NO INTRACRANIAL LESION DETECTED — {prob*100:.1f}% Conf</span>
            <span style="font-size:0.72rem; text-transform:uppercase; background:#15803d; color:#fff; padding:0.15rem 0.45rem; border-radius:4px;">Clear Screening</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. 5-Tile Metric Matrix
    rano_str = f"{max_diameter_mm:.1f}×{perpendicular_diameter_mm:.1f} mm" if (max_diameter_mm and perpendicular_diameter_mm) else "N/A"
    area_str = f"{area_mm2:.1f} mm²" if area_mm2 else "0.0 mm²"

    st.markdown(f"""
    <div class="metric-matrix">
        <div class="matrix-tile {'red' if tumor_detected else 'green'}">
            <div class="t-label">Status</div>
            <div class="t-val">{'Positive' if tumor_detected else 'Clear'}</div>
            <div class="t-sub">{prob*100:.1f}% conf</div>
        </div>
        <div class="matrix-tile purple">
            <div class="t-label">Histopathology</div>
            <div class="t-val" style="font-size:0.88rem; text-transform:capitalize;">{tumor_type if tumor_type else 'Normal'}</div>
            <div class="t-sub">{f'{classification_confidence*100:.0f}% match' if classification_confidence else 'Non-neoplastic'}</div>
        </div>
        <div class="matrix-tile blue">
            <div class="t-label">RANO 2D (L×W)</div>
            <div class="t-val" style="font-size:0.85rem;">{rano_str}</div>
            <div class="t-sub">Millimeters</div>
        </div>
        <div class="matrix-tile amber">
            <div class="t-label">Lesion Area</div>
            <div class="t-val" style="font-size:0.85rem;">{area_str}</div>
            <div class="t-sub">Calibrated</div>
        </div>
        <div class="matrix-tile teal">
            <div class="t-label">WHO Grade</div>
            <div class="t-val" style="font-size:0.88rem; text-transform:uppercase;">{severity if severity else 'LOW'}</div>
            <div class="t-sub">Severity Tier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Workstation Results Split
    col_res_img, col_res_tabs = st.columns([3.5, 6.5])

    with col_res_img:
        st.markdown("""
        <div class="pacs-panel">
            <div class="pacs-panel-header"><span>🖼️ Spatial Localization View</span></div>
        """, unsafe_allow_html=True)
        if tumor_detected and circle_info and annotated is not None:
            st.image(annotated, width=280, caption="Tumor Localized (Crosshair Red Ring Indicator)")
        else:
            st.image(processed_img, width=280, caption="Normal Brain Parenchyma (Clear)")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res_tabs:
        res_t1, res_t2, res_t3, res_t4, res_t5 = st.tabs([
            "📊 Quantitative Sizing",
            "🏥 Neurosurgical Routing",
            "💚 Care Guidance",
            "🕒 Treatment Response",
            "📄 Download PDF",
        ])

        with res_t1:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:0.55rem 0.75rem; font-size:0.8rem; color:#334155;">
                <b>RANO Major Axis ($L$):</b> <code>{f'{max_diameter_mm:.2f} mm' if max_diameter_mm else 'N/A'}</code> &nbsp;|&nbsp;
                <b>Minor Axis ($W$):</b> <code>{f'{perpendicular_diameter_mm:.2f} mm' if perpendicular_diameter_mm else 'N/A'}</code> &nbsp;|&nbsp;
                <b>Product ($L \\times W$):</b> <code>{f'{product_bidirectional_mm2:.2f} mm²' if product_bidirectional_mm2 else 'N/A'}</code><br>
                <b>Lesion Cross-Section:</b> <code>{f'{area_mm2:.2f} mm²' if area_mm2 else '0.00 mm²'}</code> &nbsp;|&nbsp;
                <b>DICOM Calibration:</b> <code>{pixel_spacing:.2f} mm/px</code>
            </div>
            """, unsafe_allow_html=True)

            if tumor_detected and classification_confidence:
                st.markdown("<p style='font-size:0.76rem; font-weight:700; color:#0f172a; margin:0.4rem 0 0.2rem;'>Differential Subtype Probabilities:</p>", unsafe_allow_html=True)
                for idx, c_name in enumerate(CLASSIFICATION_CLASSES):
                    c_prob = class_probs[idx].item() * 100
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; margin-bottom:0.15rem;">
                        <span style="text-transform:capitalize; width:85px;">{c_name}</span>
                        <div style="flex:1; height:5px; background:#e2e8f0; border-radius:3px; margin:0 6px; overflow:hidden;">
                            <div style="height:100%; width:{c_prob}%; background:#0284c7;"></div>
                        </div>
                        <span style="width:36px; text-align:right; font-weight:600;">{c_prob:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

        with res_t2:
            matched_hospitals = recommend_hospitals(patient_ctx, default_hospitals(), top_k=3)
            for h in matched_hospitals:
                h_name = h.get("name", "Neurosurgical Centre")
                h_city = h.get("city", "")
                h_score = h.get("match_score", 0.0)
                h_dist = h.get("distance_km", 0.0)
                reasons = h.get("match_reasons", {})
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:0.45rem 0.65rem; margin-bottom:0.35rem;">
                    <div style="display:flex; justify-content:space-between; font-weight:700; font-size:0.82rem; color:#0f172a;">
                        <span>🏥 {h_name} ({h_city})</span>
                        <span style="color:#0284c7;">{h_score*100:.0f}% Match</span>
                    </div>
                    <div style="font-size:0.72rem; color:#64748b; margin-top:0.15rem;">
                        📍 {h_dist} km · ⭐ Quality: {reasons.get('hospital_quality', 'N/A')} · 💰 Cost: {reasons.get('cost_fit', 'N/A')} · 🛡️ Insurance: {reasons.get('insurance_fit', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_t3:
            lifestyle = generate_lifestyle_recommendations(patient, analysis)
            for category, items in list(lifestyle.items())[:3]:
                if not items: continue
                title = category.replace("_", " ").title()
                items_html = "".join(f"<li>{item}</li>" for item in items[:2])
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:0.45rem 0.65rem; margin-bottom:0.35rem;">
                    <div style="font-weight:700; font-size:0.78rem; color:#0f172a; margin-bottom:0.15rem;">📌 {title}</div>
                    <ul style="margin:0; padding-left:1.1rem; font-size:0.78rem; color:#334155;">{items_html}</ul>
                </div>
                """, unsafe_allow_html=True)

        with res_t4:
            if response_result is None or response_result.category == INSUFFICIENT_DATA:
                st.markdown("""
                <div class="pacs-panel" style="text-align:center; padding:1.5rem;">
                    <p style="font-size:0.9rem; font-weight:700; color:#0f172a;">Baseline established.</p>
                    <p style="font-size:0.8rem; color:#64748b;">A longitudinal response assessment will become available after a follow-up scan for this same Patient ID.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                ra = response_result
                bg, border, fg = RESPONSE_BADGE_STYLE.get(ra.category, ("#f8fafc", "#e2e8f0", "#334155"))
                st.markdown(
                    f'<span class="response-badge" style="background:{bg}; border:1px solid {border}; color:{fg};">{ra.category}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(ra.assessment_label)

                def _fmt_pct(v):
                    return f"{v:+.1f}%" if v is not None else "N/A"

                def _fmt_visit(v):
                    if v is None:
                        return "N/A"
                    val, _ = (v.product_bidirectional_mm2, "product") if v.product_bidirectional_mm2 else (v.area_mm2, "area")
                    return f"{v.scan_date} — {val:.1f} mm²" if val is not None else f"{v.scan_date} — no measurable tumor"

                st.markdown(f"""
                <table class="visit-table">
                    <tr><th>Reference</th><th>Visit</th><th>% Change vs. Current</th></tr>
                    <tr><td>Baseline (first visit)</td><td>{_fmt_visit(ra.baseline)}</td><td>{_fmt_pct(ra.pct_change_from_baseline)}</td></tr>
                    <tr><td>Previous visit</td><td>{_fmt_visit(ra.previous)}</td><td>{_fmt_pct(ra.pct_change_from_previous)}</td></tr>
                    <tr><td>Nadir (smallest prior)</td><td>{_fmt_visit(ra.nadir)}</td><td>{_fmt_pct(ra.pct_change_from_nadir)}</td></tr>
                    <tr><td>Current visit</td><td>{_fmt_visit(ra.current)}</td><td>—</td></tr>
                </table>
                """, unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:0.78rem; color:#334155;'><b>Rationale:</b> {ra.rationale}</p>", unsafe_allow_html=True)

                if ra.caveats:
                    caveats_html = "".join(f"<li>{c}</li>" for c in ra.caveats)
                    st.markdown(f'<div class="caveat-box"><b>⚠ Caveats:</b><ul style="margin:0.25rem 0 0 1rem;">{caveats_html}</ul></div>', unsafe_allow_html=True)

                # Trend chart across all visits (not just the ones used as references)
                chart_rows = {
                    v.scan_date: (v.product_bidirectional_mm2 if v.product_bidirectional_mm2 else (v.area_mm2 or 0.0))
                    for v in visit_history
                }
                if len(chart_rows) >= 2:
                    st.caption("Measurement trend across visits (mm²):")
                    st.line_chart(chart_rows)

                st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#0f172a; margin-top:0.5rem;'>Visit history for this Patient ID:</p>", unsafe_allow_html=True)
                rows = "".join(
                    f"<tr><td>{v.scan_date}</td><td style='text-transform:capitalize;'>{v.tumor_type or 'N/A'}</td>"
                    f"<td>{f'{v.product_bidirectional_mm2:.1f} mm²' if v.product_bidirectional_mm2 else 'N/A'}</td>"
                    f"<td>{'<span class=\"demo-tag\">DEMO / SIMULATED</span>' if v.is_demo else 'Real intake'}</td></tr>"
                    for v in visit_history
                )
                st.markdown(f"""
                <table class="visit-table">
                    <tr><th>Date</th><th>Tumor Type</th><th>Product (L×W)</th><th>Source</th></tr>
                    {rows}
                </table>
                <div class="disclaimer-box">{ra.disclaimer}</div>
                """, unsafe_allow_html=True)

        with res_t5:
            report_path = "generated_report.pdf"
            pdf_response_assessment = response_result if (response_result and response_result.category != INSUFFICIENT_DATA) else None
            generate_full_report(
                patient, analysis,
                matched_hospitals if "matched_hospitals" in dir() else [],
                report_path,
                response_assessment=pdf_response_assessment,
            )
            with open(report_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download Clinical PDF Diagnostic Report",
                data=pdf_bytes,
                file_name=f"Smart_NeuroCare_Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            if pdf_response_assessment:
                st.caption("This PDF includes a Treatment Response Assessment section, computed from the same result shown in the Treatment Response tab.")

# ---------------------------------------------------------------------------
# Footer Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:0.6rem 0; margin-top:1rem; color:#94a3b8; font-size:0.7rem; border-top:1px solid #e2e8f0;">
    ⚖️ <b>Clinical Decision Support Disclaimer:</b> Smart NeuroCare is an investigational, non-diagnostic AI-assisted triage and monitoring aid.
    It has not been clinically validated. All findings — including the Treatment Response Assessment — require review and confirmation by a
    licensed radiologist or oncologist and do not substitute for professional medical judgment.
</div>
""", unsafe_allow_html=True)
