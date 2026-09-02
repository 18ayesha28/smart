# Smart NeuroCare

🧠 **Smart NeuroCare** is an AI-powered Brain Tumor Detection, Segmentation, Severity Scoring, and Intelligent Hospital Recommendation System.

## Overview

Smart NeuroCare provides an end-to-end clinical decision-support pipeline:
- **MRI Analysis & Detection**: Convolutional Neural Networks (CNN) for brain tumor classification and detection.
- **U-Net Segmentation**: Precise lesion boundary delineation and measurement metrics.
- **Severity Assessment**: Calculates severity scores, diameter, volume estimation, and risk profile.
- **Intelligent Hospital Recommendation**: Matches patients to specialized neurosurgery and oncology centers based on severity, location, and facilities.
- **Clinical Report Generator**: Automated structured PDF reports for medical review.
- **Interactive Streamlit Dashboard**: 2026-grade glassmorphic medical interface.

---

## Project Structure

```
smart/
├── .gitignore
├── README.md
└── smart-neurocare/
    ├── app.py                      # Main Streamlit Dashboard Application
    ├── cnn_detection_model.py      # CNN Tumor Detection Model Architecture
    ├── train_classification.py     # Classification Model & Training
    ├── unet_segmentation.py        # U-Net Segmentation & Measurement logic
    ├── train_segmentation.py       # U-Net Training Script
    ├── hospital_recommendation.py  # Hospital Scoring & Recommendation System
    ├── patient_lifestyle.py        # Patient Profiling & Lifestyle Guidance
    ├── report_generator.py         # Automated Medical PDF Report Generator
    ├── api_scans.py                # Scan API utilities
    ├── prepare_data.py             # Data preprocessing pipeline
    ├── requirements.txt            # Python dependencies
    ├── README_WINDOWS.md           # Setup and execution guide for Windows
    ├── TRAINING_GUIDE.md           # Guide for training custom models
    └── smart_neurocare_design.md   # Architectural & Design Documentation
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or 3.11 installed and added to PATH.

### 2. Setup Virtual Environment
```powershell
cd smart-neurocare
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Dashboard
```powershell
streamlit run app.py
```

---

## Documentation
- Detailed Windows Setup Guide: [README_WINDOWS.md](smart-neurocare/README_WINDOWS.md)
- Model Training Instructions: [TRAINING_GUIDE.md](smart-neurocare/TRAINING_GUIDE.md)
- System Architecture Design: [smart_neurocare_design.md](smart-neurocare/smart_neurocare_design.md)

---

## Disclaimer
*All AI outputs (detection, classification, segmentation, severity scores) are decision-support tools only and do not replace professional medical diagnosis or clinical judgment.*
