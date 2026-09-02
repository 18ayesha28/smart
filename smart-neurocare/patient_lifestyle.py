"""
Smart NeuroCare — Patient Profile, Lifestyle Recommendation Engine, and
Expanded Medical Report Generator.

Extends the original report_generator.py to:
  - Capture deep patient details (age, weight, height, habits, symptoms, family history)
  - Generate personalized lifestyle guidance (diet, exercise, habit changes)
  - Include recommended hospitals directly inside the PDF report
  - Keep the mandatory medical disclaimer

This module is rule-based and transparent by design — for healthcare-adjacent
lifestyle guidance, patients and clinicians should be able to see exactly
which factor drove which recommendation, not a black-box output.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, ListFlowable, ListItem
)
from reportlab.lib import colors

DISCLAIMER_TEXT = (
    "This report was generated with AI-assisted analysis and is intended as a "
    "decision-support tool only. It is NOT a substitute for professional medical "
    "diagnosis, treatment, or advice. Lifestyle suggestions are general wellness "
    "guidance, not a personalized clinical treatment plan. Please consult a "
    "licensed physician, oncologist, or dietitian before making any medical or "
    "lifestyle decisions."
)


# ---------------------------------------------------------------------------
# Patient profile
# ---------------------------------------------------------------------------
@dataclass
class PatientDetails:
    name: str
    patient_id: str
    age: int
    sex: str                       # "male", "female", "other"
    weight_kg: float
    height_cm: float
    smoker: bool = False
    cigarettes_per_day: int = 0
    alcohol_use: str = "none"      # "none", "occasional", "regular", "heavy"
    physical_activity: str = "moderate"  # "sedentary", "light", "moderate", "active"
    existing_conditions: list = field(default_factory=list)   # e.g. ["diabetes", "hypertension"]
    family_history_cancer: bool = False
    symptoms: list = field(default_factory=list)  # e.g. ["headaches", "seizures", "vision changes"]

    @property
    def bmi(self) -> float:
        h_m = self.height_cm / 100.0
        return round(self.weight_kg / (h_m ** 2), 1) if h_m > 0 else 0.0

    @property
    def bmi_category(self) -> str:
        b = self.bmi
        if b < 18.5:
            return "underweight"
        elif b < 25:
            return "normal"
        elif b < 30:
            return "overweight"
        return "obese"


@dataclass
class TumorAnalysisResult:
    tumor_detected: bool
    detection_confidence: float
    tumor_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    tumor_area_mm2: Optional[float] = None
    tumor_volume_mm3: Optional[float] = None
    max_diameter_mm: Optional[float] = None
    perpendicular_diameter_mm: Optional[float] = None
    product_bidirectional_mm2: Optional[float] = None
    severity_score: Optional[str] = None       # low, moderate, high, critical
    segmentation_overlay_path: Optional[str] = None
    model_version: str = "unspecified"
    scan_date: str = ""


# ---------------------------------------------------------------------------
# Lifestyle recommendation engine (rule-based, explainable)
# ---------------------------------------------------------------------------
def generate_lifestyle_recommendations(patient: PatientDetails, analysis: TumorAnalysisResult) -> dict:
    """
    Returns a dict of {category: [reasoned recommendations]}. Every recommendation
    is tied to a specific factor in the patient's profile so it's explainable
    rather than generic boilerplate.
    """
    recs = {"diet": [], "exercise": [], "habits": [], "monitoring": [], "warning_signs": []}

    # --- Diet ---
    recs["diet"].append(
        "Favor a balanced, anti-inflammatory diet: vegetables, fruits, whole grains, "
        "lean protein, and healthy fats (olive oil, nuts, fatty fish)."
    )
    if patient.bmi_category in ("overweight", "obese"):
        recs["diet"].append(
            f"BMI is {patient.bmi} ({patient.bmi_category}). A gradual, physician-guided "
            "calorie reduction and increased fiber intake can support a healthier weight "
            "range, which also matters for surgical/anesthesia risk if treatment involves surgery."
        )
    elif patient.bmi_category == "underweight":
        recs["diet"].append(
            f"BMI is {patient.bmi} (underweight). Adequate calorie and protein intake is "
            "important for recovery capacity, especially if surgery or chemotherapy is planned — "
            "discuss a nutrition plan with a clinical dietitian."
        )
    if "diabetes" in [c.lower() for c in patient.existing_conditions]:
        recs["diet"].append(
            "Existing diabetes noted — coordinate any dietary changes with an endocrinologist, "
            "since steroid medications sometimes used in brain tumor treatment can affect blood sugar."
        )

    # --- Exercise ---
    if patient.physical_activity == "sedentary":
        recs["exercise"].append(
            "Current activity is sedentary. Low-impact daily walking (15–20 min) is safe and "
            "promotes circulation, mood, and sleep — cleared by your physician first."
        )
    else:
        recs["exercise"].append(
            "Maintain moderate physical activity (walking, light swimming, stretching), avoiding "
            "heavy straining, contact sports, or activities with head-trauma risk."
        )

    # --- Habits ---
    if patient.smoker:
        recs["habits"].append(
            f"Active smoking ({patient.cigarettes_per_day} cig/day). Smoking cessation "
            "substantially improves tissue oxygenation, lowers surgical complication rates, and "
            "improves response to radiation therapy. Consult your doctor for cessation support."
        )
    if patient.alcohol_use in ("regular", "heavy"):
        recs["habits"].append(
            f"Alcohol intake ({patient.alcohol_use}). Alcohol may interact with neuro-oncology "
            "medications (including anti-epileptics like levetiracetam) and worsen cerebral edema. "
            "Minimizing or eliminating alcohol is strongly advised."
        )

    # --- Monitoring & symptoms ---
    if "seizures" in patient.symptoms:
        recs["monitoring"].append(
            "History of seizures: strict adherence to prescribed anti-epileptic medications (AEDs) "
            "is critical. Avoid driving or operating machinery until cleared by a neurologist."
        )
    if "headaches" in patient.symptoms:
        recs["monitoring"].append(
            "Track headache patterns (worse in the morning, exacerbated by coughing/bending) "
            "— these can reflect changes in intracranial pressure and should be reported to your care team."
        )

    # --- Warning signs ---
    recs["warning_signs"].append(
        "Seek IMMEDIATE emergency care if you experience: sudden severe headache, new or worsening "
        "seizures, acute vision loss, limb weakness, difficulty speaking, or sudden confusion."
    )

    return recs


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------
def build_summary_text(analysis: TumorAnalysisResult) -> str:
    if not analysis.tumor_detected:
        return (
            f"No tumor was detected in the submitted MRI scan "
            f"(model confidence: {analysis.detection_confidence * 100:.1f}%). "
            "Routine follow-up is recommended as advised by your physician."
        )
    parts = [f"A tumor was detected with {analysis.detection_confidence * 100:.1f}% model confidence."]
    if analysis.tumor_type:
        parts.append(
            f"Classified as {analysis.tumor_type}"
            + (f" ({analysis.classification_confidence * 100:.1f}% confidence)."
               if analysis.classification_confidence else ".")
        )
    if analysis.max_diameter_mm:
        if analysis.perpendicular_diameter_mm:
            parts.append(
                f"RANO/RECIST Bidirectional Dimensions: {analysis.max_diameter_mm:.1f} mm (major) × "
                f"{analysis.perpendicular_diameter_mm:.1f} mm (perpendicular minor)."
            )
        else:
            parts.append(f"Estimated maximum diameter: {analysis.max_diameter_mm:.1f} mm.")
    if analysis.tumor_area_mm2:
        parts.append(f"Estimated cross-sectional area: {analysis.tumor_area_mm2:.1f} mm².")
    if analysis.tumor_volume_mm3:
        parts.append(f"Estimated volume: {analysis.tumor_volume_mm3:.1f} mm³.")
    if analysis.severity_score:
        parts.append(f"AI-assigned severity level: {analysis.severity_score.upper()}.")
    return " ".join(parts)


def generate_full_report(
    patient: PatientDetails,
    analysis: TumorAnalysisResult,
    recommended_hospitals: list,   # list of dicts from hospital_recommendation.recommend_hospitals
    output_path: str,
):
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey))

    story = []
    story.append(Paragraph("Smart NeuroCare — AI-Assisted Brain Tumor Report", styles["TitleCenter"]))
    story.append(Spacer(1, 8 * mm))

    # --- Patient info table ---
    patient_table_data = [
        ["Name", patient.name, "Patient ID", patient.patient_id],
        ["Age", str(patient.age), "Sex", patient.sex],
        ["Weight", f"{patient.weight_kg} kg", "Height", f"{patient.height_cm} cm"],
        ["BMI", f"{patient.bmi} ({patient.bmi_category})", "Report Date",
         datetime.now().strftime("%Y-%m-%d")],
        ["Smoker", "Yes" if patient.smoker else "No", "Alcohol Use", patient.alcohol_use],
        ["Activity Level", patient.physical_activity, "Family Hx Cancer",
         "Yes" if patient.family_history_cancer else "No"],
    ]
    patient_table = Table(patient_table_data, colWidths=[30 * mm, 55 * mm, 30 * mm, 55 * mm])
    patient_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 4 * mm))

    if patient.existing_conditions:
        story.append(Paragraph(f"<b>Existing conditions:</b> {', '.join(patient.existing_conditions)}", styles["Normal"]))
    if patient.symptoms:
        story.append(Paragraph(f"<b>Reported symptoms:</b> {', '.join(patient.symptoms)}", styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    # --- Findings ---
    story.append(Paragraph("Findings", styles["Heading2"]))
    story.append(Paragraph(build_summary_text(analysis), styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    if analysis.segmentation_overlay_path:
        story.append(Paragraph("Segmentation Overlay", styles["Heading2"]))
        story.append(Image(analysis.segmentation_overlay_path, width=90 * mm, height=90 * mm))
        story.append(Spacer(1, 6 * mm))

    # --- Recommended hospitals ---
    if recommended_hospitals:
        story.append(Paragraph("Recommended Hospitals & Specialists", styles["Heading2"]))
        hosp_data = [["Hospital", "Match Score", "Distance", "Why recommended"]]
        for h in recommended_hospitals:
            reasons = ", ".join(f"{k}: {v}" for k, v in h["match_reasons"].items())
            hosp_data.append([h["name"], f"{h['match_score']:.2f}", f"{h['distance_km']} km", reasons])
        hosp_table = Table(hosp_data, colWidths=[35 * mm, 22 * mm, 22 * mm, 91 * mm])
        hosp_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(hosp_table)
        story.append(Spacer(1, 6 * mm))

    # --- Lifestyle recommendations ---
    lifestyle = generate_lifestyle_recommendations(patient, analysis)
    story.append(Paragraph("Personalized Lifestyle & Recovery Guidance", styles["Heading2"]))
    for category, items in lifestyle.items():
        if category == "warning_signs" or not items:
            continue
        story.append(Paragraph(category.replace("_", " ").title(), styles["Heading3"]))
        story.append(ListFlowable(
            [ListItem(Paragraph(item, styles["Normal"])) for item in items],
            bulletType="bullet",
        ))
        story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Seek Urgent Care If You Experience:", styles["Heading3"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(item, styles["Normal"])) for item in lifestyle["warning_signs"]],
        bulletType="bullet",
    ))

    # --- Disclaimer ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(DISCLAIMER_TEXT, styles["Disclaimer"]))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    patient = PatientDetails(
        name="Jane Doe", patient_id="P-10293", age=45, sex="female",
        weight_kg=78, height_cm=165, smoker=True, cigarettes_per_day=10,
        alcohol_use="regular", physical_activity="sedentary",
        existing_conditions=["hypertension"], family_history_cancer=True,
        symptoms=["headaches", "blurred vision"],
    )
    analysis = TumorAnalysisResult(
        tumor_detected=True, detection_confidence=0.94, tumor_type="glioma",
        classification_confidence=0.87, tumor_area_mm2=452.3, tumor_volume_mm3=8760.5,
        max_diameter_mm=28.4, severity_score="high", model_version="demo-v0",
        scan_date="2026-07-01",
    )
    hospitals = [
        {"name": "NeuroCare Institute", "match_score": 0.91, "distance_km": 12.4,
         "match_reasons": {"specialization_match": 1.0, "hospital_quality": 0.9}},
    ]
    generate_full_report(patient, analysis, hospitals, "sample_full_report.pdf")
    print("Generated sample_full_report.pdf")
