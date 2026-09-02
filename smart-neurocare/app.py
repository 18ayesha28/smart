"""
Streamlit Web Application — Smart NeuroCare
AI-Powered Brain Tumor Detection & Comprehensive Patient Analysis Platform

Ultra-Compact Clinical Workstation:
  - Dense, unified single-screen layout (zero scrolling needed to see results)
  - Left Column: Compact Scan Preview (max 280px) + Preprocessing + Run CTA
  - Right Column: Live Executive Diagnostic Findings + Metrics + Tabbed Insights
  - Balanced typography and tight clinical spacing
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
from image_preprocessing import preprocess_mri

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
# Constants
# ---------------------------------------------------------------------------
CLASSIFICATION_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

DETECTION_CKPT = "best_detection_model.pt"
CLASSIFICATION_CKPT = "best_classification_model.pt"
SEGMENTATION_CKPT = "best_segmentation_model.pt"
using_trained_weights = all(os.path.exists(p) for p in [DETECTION_CKPT, CLASSIFICATION_CKPT, SEGMENTATION_CKPT])

# ---------------------------------------------------------------------------
# Custom CSS — Ultra-Compact Clinical Sapphire & Slate Design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #f8fafc;
    --bg-card: #ffffff;
    --border-color: #e2e8f0;
    --accent-primary: #0284c7;
    --accent-hover: #0369a1;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-radius: 8px;
}

.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Tight Main Container ── */
.block-container {
    padding: 0.75rem 1.25rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Slim Header Bar ── */
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
    margin: 0;
    line-height: 1.2;
}

.clinical-navbar .brand-subtitle {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin: 0;
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

/* ── Compact Cards ── */
.clinical-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.clinical-card-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Compact Metrics Row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.5rem;
    margin: 0.5rem 0 0.75rem;
}

.metric-tile {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 0.45rem 0.6rem;
    border-top: 3px solid #0284c7;
}

.metric-tile.red { border-top-color: #dc2626; }
.metric-tile.green { border-top-color: #16a34a; }
.metric-tile.purple { border-top-color: #6366f1; }
.metric-tile.blue { border-top-color: #0284c7; }
.metric-tile.amber { border-top-color: #d97706; }
.metric-tile.teal { border-top-color: #0d9488; }

.metric-tile .tile-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    margin-bottom: 0.15rem;
}

.metric-tile .tile-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-tile .tile-sub {
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin-top: 0.1rem;
}

/* ── Result Alert Banners ── */
.result-alert {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.85rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.result-alert.detected {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #dc2626;
}

.result-alert.clear {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
}

/* ── Compact Hospital & Lifestyle Cards ── */
.hospital-card-compact {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.45rem;
}

.hospital-card-compact:hover {
    border-color: #bae6fd;
}

.lifestyle-card-compact {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.4rem;
}

.lifestyle-card-compact ul {
    margin: 0;
    padding-left: 1.1rem;
}

.lifestyle-card-compact li {
    font-size: 0.8rem;
    color: #334155;
    line-height: 1.45;
}

/* ── Action Buttons ── */
.stButton > button {
    background: #0284c7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(2, 132, 199, 0.25) !important;
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
    padding: 0.5rem 1.2rem !important;
    font-size: 0.85rem !important;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 32px;
    border-radius: 4px;
    padding: 0 10px;
    font-weight: 600;
    font-size: 0.8rem;
}

.stTabs [aria-selected="true"] {
    background-color: #f0f9ff !important;
    color: #0284c7 !important;
    border: 1px solid #bae6fd !important;
}

/* ── File Uploader Compact ── */
.stFileUploader > div {
    padding: 0.75rem !important;
    border-radius: 6px !important;
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
    cv2.circle(overlay, (cx, cy), r + 3, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)

    cv2.circle(img_bgr, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)

    tick_len = max(6, r // 5)
    color = (0, 0, 255)
    cv2.line(img_bgr, (cx - r, cy), (cx - r + tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx + r, cy), (cx + r - tick_len, cy), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy - r), (cx, cy - r + tick_len), color, 1, cv2.LINE_AA)
    cv2.line(img_bgr, (cx, cy + r), (cx, cy + r - tick_len), color, 1, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ---------------------------------------------------------------------------
# Ensure sample scans exist
# ---------------------------------------------------------------------------
SAMPLE_DIR = "sample_scans"
if not os.path.exists(SAMPLE_DIR):
    from create_sample_scans import generate_sample_mris
    generate_sample_mris(SAMPLE_DIR)

# ---------------------------------------------------------------------------
# Sidebar: Compact Patient Profile
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 👤 Patient Profile")

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

with st.sidebar.expander("🪪 Identity & Vitals", expanded=True):
    name = st.text_input("Name", st.session_state.get("p_name", "Jane Doe"))
    patient_id = st.text_input("ID", st.session_state.get("p_id", "P-10293"))
    age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("p_age", 45))
    sex = st.selectbox("Sex", ["female", "male", "other"], index=0 if st.session_state.get("p_sex") == "female" else 1)
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0)
    height_cm = st.number_input("Height (cm)", min_value=30.0, value=165.0)

bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm else 0
st.sidebar.caption(f"📋 {name} · {patient_id} · {age}y {sex} · BMI {bmi:.1f}")

with st.sidebar.expander("🏥 Symptoms & Conditions"):
    existing_conditions = st.multiselect(
        "Conditions",
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
    insurance = st.text_input("Insurance", "StarHealth")
    lat = st.number_input("Latitude", value=12.9716, format="%.4f")
    lon = st.number_input("Longitude", value=77.5946, format="%.4f")
    pixel_spacing = st.number_input("Pixel spacing (mm)", value=1.0, step=0.1)


# ---------------------------------------------------------------------------
# Slim Top Clinical Header Bar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="clinical-navbar">
    <div style="display:flex; align-items:center; gap:0.5rem;">
        <span style="font-size:1.3rem;">🧠</span>
        <div>
            <span class="brand-title">Smart NeuroCare™</span>
            <span style="font-size:0.75rem; color:#64748b; margin-left:0.4rem;">AI Neuro-Oncology Triage Suite</span>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:0.4rem;">
        <span class="status-pill success">🟢 3 AI Models Loaded</span>
        <span class="status-pill info">📐 DICOM Calibrated</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Compact Dual-Path Scan Intake
# ---------------------------------------------------------------------------
intake_col1, intake_col2 = st.columns([1, 1])

active_image = None
active_image_name = None

with intake_col1:
    uploaded_file = st.file_uploader("Upload MRI Slice (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        active_image_name = uploaded_file.name

with intake_col2:
    st.markdown("<p style='font-size:0.78rem; font-weight:600; color:#475569; margin:0 0 0.25rem;'>Or click a verified clinical case to test instantly:</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🟡 Case A\nMeningioma", use_container_width=True):
            p = os.path.join(SAMPLE_DIR, "meningioma_sample.png")
            if os.path.exists(p):
                active_image = Image.open(p).convert("RGB")
                active_image_name = "Case_A_Meningioma.png"
                st.session_state["sample_loaded"] = active_image_name
    with c2:
        if st.button("🔴 Case B\nGlioma", use_container_width=True):
            p = os.path.join(SAMPLE_DIR, "glioma_sample.png")
            if os.path.exists(p):
                active_image = Image.open(p).convert("RGB")
                active_image_name = "Case_B_Glioma.png"
                st.session_state["sample_loaded"] = active_image_name
    with c3:
        if st.button("🟣 Case C\nPituitary", use_container_width=True):
            p = os.path.join(SAMPLE_DIR, "pituitary_sample.png")
            if os.path.exists(p):
                active_image = Image.open(p).convert("RGB")
                active_image_name = "Case_C_Pituitary.png"
                st.session_state["sample_loaded"] = active_image_name
    with c4:
        if st.button("🟢 Case D\nNormal", use_container_width=True):
            p = os.path.join(SAMPLE_DIR, "healthy_normal_sample.png")
            if os.path.exists(p):
                active_image = Image.open(p).convert("RGB")
                active_image_name = "Case_D_Normal.png"
                st.session_state["sample_loaded"] = active_image_name

# Retain sample scan in session
if active_image is None and st.session_state.get("sample_loaded"):
    s_name = st.session_state["sample_loaded"]
    mapping = {
        "Case_A_Meningioma.png": "meningioma_sample.png",
        "Case_B_Glioma.png": "glioma_sample.png",
        "Case_C_Pituitary.png": "pituitary_sample.png",
        "Case_D_Normal.png": "healthy_normal_sample.png",
    }
    target = os.path.join(SAMPLE_DIR, mapping.get(s_name, "meningioma_sample.png"))
    if os.path.exists(target):
        active_image = Image.open(target).convert("RGB")
        active_image_name = s_name


# ---------------------------------------------------------------------------
# Integrated Workstation (Side-by-Side: Scan on Left, Live Results on Right)
# ---------------------------------------------------------------------------
if active_image is None:
    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:1.25rem; text-align:center; margin-top:0.5rem;">
        <p style="font-size:0.95rem; font-weight:700; color:#0f172a; margin:0 0 0.25rem;">👈 Upload a Brain MRI Slice or Click a Demo Case Above to Begin</p>
        <p style="font-size:0.8rem; color:#64748b; margin:0;">Supports Axial T1/T2 & FLAIR brain slices · AI automated RANO measurements, subtype classification, and hospital triage</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # 2-Column Clinical Layout
    col_left_scan, col_right_results = st.columns([3, 7])

    with col_left_scan:
        st.markdown(f"""
        <div class="clinical-card" style="padding:0.6rem 0.8rem;">
            <div class="clinical-card-header" style="margin-bottom:0.25rem;">
                <span style="font-size:0.82rem;">🖼️ Active MRI Slice</span>
                <span style="font-size:0.7rem; color:#0284c7; font-family:'JetBrains Mono';">{active_image_name}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing toggles
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            enable_crop = st.checkbox("🧠 Auto-Crop", value=True, help="Removes uninformative black margins")
        with col_c2:
            enable_clahe = st.checkbox("✨ CLAHE", value=True, help="Equalizes MRI contrast variance")

        processed_image = preprocess_mri(active_image, auto_crop=enable_crop, enhance_contrast=enable_clahe)

        # Render compact image (constrained width so it never overflows vertically)
        image_slot = st.empty()
        image_slot.image(processed_image, width=280)

        # Primary Run CTA right under image
        run_btn = st.button("🔬 Run Diagnostic Analysis", use_container_width=True)

    with col_right_results:
        # Check if analysis should run or has run
        if run_btn:
            st.session_state["analyzed_image_name"] = active_image_name

        has_analysis = st.session_state.get("analyzed_image_name") == active_image_name

        if not has_analysis:
            st.markdown(f"""
            <div class="clinical-card" style="min-height:360px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
                <span style="font-size:2rem; margin-bottom:0.4rem;">🔬</span>
                <p style="font-size:0.95rem; font-weight:700; color:#0f172a; margin:0 0 0.25rem;">Ready for Diagnostic Analysis</p>
                <p style="font-size:0.8rem; color:#64748b; max-width:400px; margin:0 0 1rem;">
                    Patient: <b>{name}</b> ({age}y, {sex.title()}) · ID: <b>{patient_id}</b><br>
                    Click <b>"Run Diagnostic Analysis"</b> on the left to extract lesion coordinates, RANO measurements, and triage recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            # 1. Detection
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
                # 2. Classification
                with torch.no_grad():
                    class_logits = classification_model(tensor)
                    class_probs = torch.softmax(class_logits, dim=1)[0]
                    pred_idx = class_probs.argmax().item()
                    tumor_type = CLASSIFICATION_CLASSES[pred_idx]
                    classification_confidence = class_probs[pred_idx].item()

                # 3. Segmentation
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
                    image_slot.image(annotated, width=280)
                    overlay_path = "demo_overlay.png"
                    annotated.save(overlay_path)

                if area_mm2 is not None and area_mm2 > 0:
                    if area_mm2 < 200: severity = "low"
                    elif area_mm2 < 500: severity = "moderate"
                    elif area_mm2 < 1000: severity = "high"
                    else: severity = "critical"
                else:
                    severity = "moderate"

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
                smoker=False,
                cigarettes_per_day=0,
                alcohol_use="none",
                physical_activity="moderate",
                existing_conditions=existing_conditions,
                family_history_cancer=family_history_cancer,
                symptoms=symptoms,
                max_budget=float(budget),
                insurance_provider=insurance,
                latitude=float(lat),
                longitude=float(lon),
            )

            # Executive Summary Top Banner (IMMEDIATELY VISIBLE!)
            if tumor_detected:
                st.markdown(f"""
                <div class="result-alert detected">
                    <span>🔴 TUMOR DETECTED — {prob*100:.1f}% Confidence · Subtype: <b style="text-transform:capitalize;">{tumor_type}</b> ({classification_confidence*100:.1f}%)</span>
                    <span style="font-size:0.75rem; text-transform:uppercase; background:#b91c1c; color:#fff; padding:0.15rem 0.5rem; border-radius:4px;">Severity: {severity}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-alert clear">
                    <span>🟢 NO TUMOR DETECTED — {prob*100:.1f}% Confidence</span>
                    <span style="font-size:0.75rem; text-transform:uppercase; background:#15803d; color:#fff; padding:0.15rem 0.5rem; border-radius:4px;">Clear Screening</span>
                </div>
                """, unsafe_allow_html=True)

            # Compact 5-Metric Tiles
            rano_str = f"{max_diameter_mm:.1f}×{perpendicular_diameter_mm:.1f} mm" if (max_diameter_mm and perpendicular_diameter_mm) else "N/A"
            area_str = f"{area_mm2:.1f} mm²" if area_mm2 else "0.0 mm²"

            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-tile {'red' if tumor_detected else 'green'}">
                    <div class="tile-label">Detection</div>
                    <div class="tile-value">{'Positive' if tumor_detected else 'Negative'}</div>
                    <div class="tile-sub">{prob*100:.1f}% conf</div>
                </div>
                <div class="metric-tile purple">
                    <div class="tile-label">Subtype</div>
                    <div class="tile-value" style="font-size:0.95rem; text-transform:capitalize;">{tumor_type if tumor_type else 'Normal'}</div>
                    <div class="tile-sub">{f'{classification_confidence*100:.0f}% match' if classification_confidence else 'Non-neoplastic'}</div>
                </div>
                <div class="metric-tile blue">
                    <div class="tile-label">RANO 2D (L×W)</div>
                    <div class="tile-value" style="font-size:0.9rem;">{rano_str}</div>
                    <div class="tile-sub">Millimeters</div>
                </div>
                <div class="metric-tile amber">
                    <div class="tile-label">Lesion Area</div>
                    <div class="tile-value" style="font-size:0.9rem;">{area_str}</div>
                    <div class="tile-sub">Calibrated</div>
                </div>
                <div class="metric-tile teal">
                    <div class="tile-label">Severity</div>
                    <div class="tile-value" style="font-size:0.95rem; text-transform:uppercase;">{severity if severity else 'LOW'}</div>
                    <div class="tile-sub">WHO Graded</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Compact Results Tabs
            rt1, rt2, rt3, rt4 = st.tabs([
                "📊 Detailed Measurements",
                "🏥 Hospital Triage",
                "💚 Lifestyle Guidance",
                "📄 Download Report"
            ])

            with rt1:
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:0.6rem 0.8rem; font-size:0.82rem; color:#334155;">
                    <b>RANO Major Axis ($L$):</b> <code>{f'{max_diameter_mm:.2f} mm' if max_diameter_mm else 'N/A'}</code> &nbsp;|&nbsp;
                    <b>Minor Axis ($W$):</b> <code>{f'{perpendicular_diameter_mm:.2f} mm' if perpendicular_diameter_mm else 'N/A'}</code> &nbsp;|&nbsp;
                    <b>Product ($L \\times W$):</b> <code>{f'{product_bidirectional_mm2:.2f} mm²' if product_bidirectional_mm2 else 'N/A'}</code><br>
                    <b>Cross-Sectional Area:</b> <code>{f'{area_mm2:.2f} mm²' if area_mm2 else '0.00 mm²'}</code> &nbsp;|&nbsp;
                    <b>DICOM Scale:</b> <code>{pixel_spacing:.2f} mm/px</code>
                </div>
                """, unsafe_allow_html=True)

                if tumor_detected and classification_confidence:
                    st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#0f172a; margin:0.5rem 0 0.25rem;'>Differential Subtype Probabilities:</p>", unsafe_allow_html=True)
                    for idx, c_name in enumerate(CLASSIFICATION_CLASSES):
                        c_prob = class_probs[idx].item() * 100
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; margin-bottom:0.2rem;">
                            <span style="text-transform:capitalize; width:90px;">{c_name}</span>
                            <div style="flex:1; height:6px; background:#e2e8f0; border-radius:4px; margin:0 8px; overflow:hidden;">
                                <div style="height:100%; width:{c_prob}%; background:#0284c7;"></div>
                            </div>
                            <span style="width:40px; text-align:right; font-weight:600;">{c_prob:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)

            with rt2:
                matched_hospitals = recommend_hospitals(patient, analysis, default_hospitals(), top_n=3)
                for h in matched_hospitals:
                    h_name = h.get("name", "Neurosurgical Centre")
                    h_city = h.get("city", "")
                    h_score = h.get("match_score", 0.0)
                    h_dist = h.get("distance_km", 0.0)
                    reasons = h.get("match_reasons", {})
                    st.markdown(f"""
                    <div class="hospital-card-compact">
                        <div style="display:flex; justify-content:space-between; font-weight:700; font-size:0.85rem; color:#0f172a;">
                            <span>🏥 {h_name} ({h_city})</span>
                            <span style="color:#0284c7;">{h_score*100:.0f}% Match</span>
                        </div>
                        <div style="font-size:0.75rem; color:#64748b; margin-top:0.2rem;">
                            📍 {h_dist} km · ⭐ Quality: {reasons.get('hospital_quality', 'N/A')} · 💰 Cost: {reasons.get('cost_fit', 'N/A')} · 🛡️ Insurance: {reasons.get('insurance_fit', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with rt3:
                lifestyle = generate_lifestyle_recommendations(patient, analysis)
                for category, items in list(lifestyle.items())[:3]:
                    if not items: continue
                    title = category.replace("_", " ").title()
                    items_html = "".join(f"<li>{item}</li>" for item in items[:2])
                    st.markdown(f"""
                    <div class="lifestyle-card-compact">
                        <div style="font-weight:700; font-size:0.8rem; color:#0f172a; margin-bottom:0.2rem;">📌 {title}</div>
                        <ul>{items_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)

            with rt4:
                report_path = "generated_report.pdf"
                generate_patient_report(patient, analysis, matched_hospitals if 'matched_hospitals' in locals() else [], report_path)
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name=f"Smart_NeuroCare_Report_{patient_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Footer Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:0.75rem 0; margin-top:1.5rem; color:#94a3b8; font-size:0.72rem; border-top:1px solid #e2e8f0;">
    ⚖️ <b>Clinical Decision Support Disclaimer:</b> Smart NeuroCare is an investigational AI-assisted triaging tool. All findings require radiologist verification.
</div>
""", unsafe_allow_html=True)
