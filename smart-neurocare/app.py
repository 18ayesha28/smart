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
from image_preprocessing import preprocess_mri, crop_brain_contour, apply_clahe_enhancement


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
# Custom CSS — Premium Clinical Light Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables — Clinical Light Theme ── */
:root {
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #fcfdfe;
    --glass-border: #e2e8f0;
    --glass-glow: rgba(37, 99, 235, 0.04);
    --accent-primary: #2563eb;
    --accent-gradient: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 50%, #db2777 100%);
    --accent-gradient-soft: linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(124,58,237,0.08) 50%, rgba(219,39,119,0.08) 100%);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --severity-low: #16a34a;
    --severity-moderate: #d97706;
    --severity-high: #dc2626;
    --severity-critical: #b91c1c;
    --border-radius: 16px;
    --card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
    --card-shadow-hover: 0 16px 36px -4px rgba(37, 99, 235, 0.08), 0 4px 12px -2px rgba(0, 0, 0, 0.03);
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Overrides ── */
.stApp {
    background-color: var(--bg-primary) !important;
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
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.02) !important;
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
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div > input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div:focus-within,
section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within,
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

section[data-testid="stSidebar"] .stCheckbox label span {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
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
    font-size: 3.2rem;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 4px 16px rgba(37,99,235,0.25));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
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
    line-height: 1.15 !important;
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
    font-weight: 600;
}

.status-banner.success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #15803d;
}

.status-banner.warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #b45309;
}

.status-banner .status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse-dot 2s infinite;
}

.status-banner.success .status-dot { background: #22c55e; }
.status-banner.warning .status-dot { background: #f59e0b; }

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(37,99,235,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(37,99,235,0); }
}

/* ── Glass/White Card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    transition: var(--transition);
    box-shadow: var(--card-shadow);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.4), transparent);
}

.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: #cbd5e1;
    box-shadow: var(--card-shadow-hover);
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
    background: #ffffff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: var(--border-radius) !important;
    transition: var(--transition) !important;
    padding: 2rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
}

.stFileUploader > div:hover {
    border-color: var(--accent-primary) !important;
    background: #f8fafc !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.08) !important;
}

.stFileUploader label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em;
    transition: var(--transition) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.4) !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #eff6ff 0%, #ede9fe 100%) !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.08) !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #dbeafe 0%, #ddd6fe 100%) !important;
    border-color: #93c5fd !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.16) !important;
}

/* ── Metric Cards Grid ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.25rem 0;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: var(--transition);
    box-shadow: var(--card-shadow);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    border-radius: 4px 4px 0 0;
}

.metric-card.blue::before { background: linear-gradient(90deg, #2563eb, #3b82f6); }
.metric-card.purple::before { background: linear-gradient(90deg, #7c3aed, #8b5cf6); }
.metric-card.green::before { background: linear-gradient(90deg, #16a34a, #22c55e); }
.metric-card.amber::before { background: linear-gradient(90deg, #d97706, #f59e0b); }
.metric-card.red::before { background: linear-gradient(90deg, #dc2626, #ef4444); }
.metric-card.pink::before { background: linear-gradient(90deg, #db2777, #ec4899); }

.metric-card:hover {
    border-color: #cbd5e1;
    box-shadow: var(--card-shadow-hover);
    transform: translateY(-2px);
}

.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 700;
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
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.detection-badge.detected .badge-dot {
    background: #ef4444;
    box-shadow: 0 0 10px rgba(239,68,68,0.4);
    animation: pulse-dot 1.5s infinite;
}

.detection-badge.clear .badge-dot {
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34,197,94,0.4);
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
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.severity-badge.moderate {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}

.severity-badge.high {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

.severity-badge.critical {
    background: #450a0a;
    color: #fee2e2;
    border: 1px solid #991b1b;
}

/* ── MRI Image Container ── */
.mri-container {
    position: relative;
    border-radius: var(--border-radius);
    overflow: hidden;
    border: 1px solid #e2e8f0;
    background: #0f172a;
    box-shadow: var(--card-shadow);
}

.mri-container img {
    width: 100%;
    display: block;
}

/* ── Tumor Details Overlay ── */
.tumor-details-overlay {
    background: #ffffff;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    margin-top: 0.85rem;
    box-shadow: 0 8px 24px -4px rgba(239, 68, 68, 0.08), 0 2px 6px rgba(0, 0, 0, 0.02);
}

.tumor-details-overlay .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #f1f5f9;
}

.tumor-details-overlay .detail-row:last-child {
    border-bottom: none;
}

.tumor-details-overlay .detail-label {
    color: var(--text-secondary);
    font-size: 0.84rem;
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
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 2rem 0;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
}

.streamlit-expanderHeader:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
}

.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
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
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
    box-shadow: var(--card-shadow) !important;
}

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    padding: 1.25rem 1rem;
    margin-top: 2rem;
    color: var(--text-muted);
    font-size: 0.8rem;
    font-style: italic;
    border-top: 1px solid #e2e8f0;
}

/* ── Hospital Card ── */
.hospital-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: var(--transition);
    box-shadow: var(--card-shadow);
}

.hospital-card:hover {
    border-color: #93c5fd;
    box-shadow: var(--card-shadow-hover);
    transform: translateY(-2px);
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
    color: #1d4ed8;
}

/* ── Lifestyle Category ── */
.lifestyle-category {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: var(--transition);
    box-shadow: var(--card-shadow);
}

.lifestyle-category:hover {
    border-color: #cbd5e1;
    box-shadow: var(--card-shadow-hover);
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

/* ── Progress Stepper ── */
.nc-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 0 0 1.75rem;
    flex-wrap: wrap;
}

.nc-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem 0.4rem 0.5rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    transition: var(--transition);
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
    background: #e2e8f0;
    color: var(--text-muted);
    flex-shrink: 0;
}

.nc-step.active {
    background: var(--accent-gradient-soft);
    border-color: #93c5fd;
    color: var(--accent-primary);
}

.nc-step.active .nc-step-num {
    background: var(--accent-primary);
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
    width: 20px;
    height: 2px;
    background: #e2e8f0;
    flex-shrink: 0;
}

/* ── Confidence Meter ── */
.confidence-meter {
    width: 100%;
    height: 8px;
    border-radius: 8px;
    background: #e2e8f0;
    overflow: hidden;
    margin-top: 0.5rem;
}

.confidence-meter .confidence-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

.confidence-meter .confidence-fill.danger { background: linear-gradient(90deg, #dc2626, #ef4444); }
.confidence-meter .confidence-fill.safe { background: linear-gradient(90deg, #16a34a, #22c55e); }
.confidence-meter .confidence-fill.info { background: linear-gradient(90deg, #2563eb, #7c3aed); }

/* ── Upload Info Chip ── */
.upload-info-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.6rem;
}

/* ── Patient Snapshot ── */
.patient-snapshot {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin: 0.5rem 0 1rem;
    line-height: 1.6;
}

/* ── Back to Top ── */
.nc-back-to-top {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--accent-gradient);
    color: #ffffff !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    text-decoration: none !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.35);
    z-index: 999;
    transition: var(--transition);
}

.nc-back-to-top:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(37,99,235,0.45);
}

/* ── Accessibility: visible focus states ── */
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 3px solid rgba(37,99,235,0.5) !important;
    outline-offset: 2px !important;
}

/* ── Responsive tweaks ── */
@media (max-width: 640px) {
    .hero-header h1 { font-size: 2rem !important; }
    .hero-header .hero-icon { font-size: 2.2rem; }
    .hero-header .hero-subtitle { font-size: 0.9rem; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .metrics-grid { grid-template-columns: 1fr 1fr !important; }
    .nc-step .nc-step-label { display: none; }
    .nc-back-to-top { width: 38px; height: 38px; bottom: 16px; right: 16px; font-size: 1rem; }
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
<div id="nc-top"></div>
<div class="hero-header">
    <div class="hero-icon">🧠</div>
    <h1>Smart NeuroCare</h1>
    <div class="hero-subtitle">AI-Powered Brain Tumor Detection & Analysis Platform</div>
</div>
<a href="#nc-top" class="nc-back-to-top" title="Back to top">↑</a>
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
st.sidebar.caption("Fill in the sections below, then upload a scan to run analysis.")

with st.sidebar.expander("🪪 Identity & Vitals", expanded=True):
    name = st.text_input("Full name", "Jane Doe")
    patient_id = st.text_input("Patient ID", "P-10293")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    sex = st.selectbox("Sex", ["female", "male", "other"])
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
    symptoms = st.multiselect(
        "Current symptoms",
        ["headaches", "seizures", "vision changes", "balance issues", "nausea", "memory issues", "none"],
        default=["none"],
    )

with st.sidebar.expander("📍 Location & Insurance"):
    budget = st.number_input("Max budget (₹)", value=800000, step=50000)
    insurance = st.text_input("Insurance provider", "StarHealth")
    lat = st.number_input("Latitude", value=12.9716, format="%.4f")
    lon = st.number_input("Longitude", value=77.5946, format="%.4f")

with st.sidebar.expander("⚙️ Advanced / Technical Settings"):
    pixel_spacing = st.number_input(
        "Pixel spacing (mm)", value=1.0, step=0.1,
        help="Real-world millimeters per pixel, from the scan's DICOM metadata. Used to convert pixel measurements into clinical mm / mm² units.",
    )


# ---------------------------------------------------------------------------
# Main: Upload + Analysis
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header"><span class="section-icon">📤</span> Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a brain MRI image (PNG / JPEG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
st.caption("Accepted formats: PNG, JPG, JPEG · Axial T1/T2 brain MRI slices work best · Max recommended size 10 MB")


def render_stepper(stage: int) -> None:
    """Purely visual progress indicator. stage: 0=upload, 1=configure/analyze, 2=results reviewed."""
    labels = ["Upload Scan", "Configure & Analyze", "Review Results"]
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
            f'<span class="nc-step-label">{label}</span></div>'
        )
        if i < len(labels) - 1:
            parts.append('<div class="nc-step-connector"></div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


nc_stage = 0
if uploaded_file is not None:
    nc_stage = 2 if st.session_state.get("nc_analysis_done_for") == uploaded_file.name else 1
render_stepper(nc_stage)

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")

    # Image enhancement options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_crop = st.checkbox("🧠 AI Brain Auto-Crop (Remove Dark Margins)", value=True, help="Removes uninformative black background to focus network receptive fields on neurological tissue.")
    with col_opt2:
        enable_clahe = st.checkbox("✨ CLAHE Contrast Normalization", value=True, help="Equalizes intra-scanner illumination differences (1.5T vs 3.0T MRI variance) and sharpens lesion contours.")

    # Apply preprocessing if enabled
    image = preprocess_mri(raw_image, auto_crop=enable_crop, enhance_contrast=enable_clahe)

    # Center column for the image
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_center:
        # Show uploaded/enhanced image initially
        image_placeholder = st.empty()
        caption_text = "Uploaded MRI Scan (Enhanced with Auto-Crop & CLAHE)" if (enable_crop or enable_clahe) else "Uploaded MRI Scan"
        image_placeholder.image(image, caption=caption_text, use_column_width=True)
        st.markdown(f"""
        <div style="text-align:center;">
            <span class="upload-info-chip">📁 {uploaded_file.name} · {image.size[0]}×{image.size[1]}px</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔬 Run Analysis", use_container_width=True):
        st.session_state["nc_analysis_done_for"] = uploaded_file.name
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
                <div style="max-width:420px; margin: 0.6rem auto 0;">
                    <div class="confidence-meter"><div class="confidence-fill danger" style="width:{prob*100:.1f}%;"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; margin: 1rem 0;">
                <span class="detection-badge clear">
                    <span class="badge-dot"></span>
                    NO TUMOR DETECTED — {prob*100:.1f}% confidence
                </span>
                <div style="max-width:420px; margin: 0.6rem auto 0;">
                    <div class="confidence-meter"><div class="confidence-fill safe" style="width:{prob*100:.1f}%;"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
                perpendicular_diameter_mm = measurements.get("perpendicular_diameter_mm", 0.0)
                product_bidirectional_mm2 = measurements.get("product_bidirectional_mm2", 0.0)

                # Find tumor circle for overlay
                orig_w, orig_h = image.size
                circle_info = find_tumor_circle(mask, orig_w, orig_h)

            # Draw overlay and display
            with col_center:
                if circle_info:
                    annotated = draw_tumor_overlay(image, circle_info)
                    image_placeholder.image(annotated, caption="Tumor Detected — Hover below for details", use_column_width=True)

                    # Save overlay for PDF
                    overlay_path = "demo_overlay.png"
                    annotated.save(overlay_path)
                else:
                    image_placeholder.image(image, caption="Tumor detected (spatial localization unavailable)", use_column_width=True)

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
            rano_display = f"{max_diameter_mm:.1f} mm × {perpendicular_diameter_mm:.1f} mm" if (max_diameter_mm and perpendicular_diameter_mm) else f"{max_diameter_mm:.1f} mm" if max_diameter_mm else "N/A"
            classification_meter_html = ""
            if classification_confidence:
                classification_meter_html = (
                    '<div style="padding: 0 0 0.6rem;"><div class="confidence-meter">'
                    f'<div class="confidence-fill info" style="width:{classification_confidence*100:.1f}%;"></div>'
                    "</div></div>"
                )
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
                {classification_meter_html}
                <div class="detail-row">
                    <span class="detail-label">RANO Bidirectional Dimensions (L × W)</span>
                    <span class="detail-value">{rano_display}</span>
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
            perpendicular_diameter_mm=perpendicular_diameter_mm,
            product_bidirectional_mm2=product_bidirectional_mm2,
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
            is_warning = "border-color: #fca5a5; background: #fff8f8;" if category == "warning_signs" else ""

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
