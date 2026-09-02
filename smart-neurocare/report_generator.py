"""
Smart NeuroCare — Automated Medical Report Generator

Compiles detection/classification/segmentation/severity results into a
structured report and renders it to PDF using reportlab.

pip install reportlab
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib import colors


DISCLAIMER_TEXT = (
    "This report was generated with AI-assisted analysis and is intended as a "
    "decision-support tool only. It is NOT a substitute for professional medical "
    "diagnosis, treatment, or advice. Please consult a licensed radiologist or "
    "neurologist to confirm all findings before making any medical decisions."
)


@dataclass
class TumorAnalysisResult:
    patient_name: str
    patient_id: str
    scan_date: str
    tumor_detected: bool
    detection_confidence: float
    tumor_type: Optional[str]
    classification_confidence: Optional[float]
    tumor_area_mm2: Optional[float]
    tumor_volume_mm3: Optional[float]
    max_diameter_mm: Optional[float]
    severity_score: Optional[str]
    segmentation_overlay_path: Optional[str]  # path to overlay image (mask on MRI)
    model_version: str


def build_summary_text(result: TumorAnalysisResult) -> str:
    if not result.tumor_detected:
        return (
            f"No tumor was detected in the submitted MRI scan "
            f"(model confidence: {result.detection_confidence * 100:.1f}%). "
            "Routine follow-up is recommended as advised by your physician."
        )

    parts = [
        f"A tumor was detected with {result.detection_confidence * 100:.1f}% model confidence."
    ]
    if result.tumor_type:
        parts.append(
            f"The tumor was classified as **{result.tumor_type}** "
            f"({result.classification_confidence * 100:.1f}% confidence)."
        )
    if result.max_diameter_mm:
        parts.append(f"Estimated maximum diameter: {result.max_diameter_mm:.1f} mm.")
    if result.tumor_area_mm2:
        parts.append(f"Estimated cross-sectional area: {result.tumor_area_mm2:.1f} mm².")
    if result.tumor_volume_mm3:
        parts.append(f"Estimated volume: {result.tumor_volume_mm3:.1f} mm³.")
    if result.severity_score:
        parts.append(f"AI-assigned severity level: **{result.severity_score.upper()}**.")

    return " ".join(parts)


def generate_pdf_report(result: TumorAnalysisResult, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Disclaimer", parent=styles["Normal"],
                               fontSize=8, textColor=colors.grey))

    story = []

    story.append(Paragraph("Smart NeuroCare — AI-Assisted Brain Tumor Report", styles["TitleCenter"]))
    story.append(Spacer(1, 10 * mm))

    # Patient info table
    patient_table_data = [
        ["Patient Name", result.patient_name],
        ["Patient ID", result.patient_id],
        ["Scan Date", result.scan_date],
        ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Model Version", result.model_version],
    ]
    patient_table = Table(patient_table_data, colWidths=[60 * mm, 100 * mm])
    patient_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 8 * mm))

    # Findings
    story.append(Paragraph("Findings", styles["Heading2"]))
    story.append(Paragraph(build_summary_text(result), styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    # Segmentation overlay image
    if result.segmentation_overlay_path:
        story.append(Paragraph("Segmentation Overlay", styles["Heading2"]))
        story.append(Image(result.segmentation_overlay_path, width=100 * mm, height=100 * mm))
        story.append(Spacer(1, 6 * mm))

    # Structured metrics table
    if result.tumor_detected:
        story.append(Paragraph("Detailed Metrics", styles["Heading2"]))
        metrics_data = [["Metric", "Value"]]
        metrics_data.append(["Tumor Type", result.tumor_type or "N/A"])
        metrics_data.append(["Detection Confidence", f"{result.detection_confidence * 100:.1f}%"])
        if result.classification_confidence:
            metrics_data.append(["Classification Confidence", f"{result.classification_confidence * 100:.1f}%"])
        if result.max_diameter_mm:
            metrics_data.append(["Max Diameter", f"{result.max_diameter_mm:.1f} mm"])
        if result.tumor_area_mm2:
            metrics_data.append(["Cross-sectional Area", f"{result.tumor_area_mm2:.1f} mm²"])
        if result.tumor_volume_mm3:
            metrics_data.append(["Volume", f"{result.tumor_volume_mm3:.1f} mm³"])
        if result.severity_score:
            metrics_data.append(["Severity", result.severity_score.upper()])

        metrics_table = Table(metrics_data, colWidths=[80 * mm, 80 * mm])
        metrics_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 8 * mm))

    # Disclaimer (mandatory)
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(DISCLAIMER_TEXT, styles["Disclaimer"]))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    result = TumorAnalysisResult(
        patient_name="Jane Doe",
        patient_id="P-10293",
        scan_date="2026-07-01",
        tumor_detected=True,
        detection_confidence=0.94,
        tumor_type="glioma",
        classification_confidence=0.87,
        tumor_area_mm2=452.3,
        tumor_volume_mm3=8760.5,
        max_diameter_mm=28.4,
        severity_score="high",
        segmentation_overlay_path=None,  # provide a real path in production
        model_version="detection-v2.3 / segmentation-v1.8",
    )
    generate_pdf_report(result, "sample_report.pdf")
    print("Report generated: sample_report.pdf")
