"""
Smart NeuroCare — Longitudinal Treatment-Response Assessment

Deterministic, rule-based comparison of tumor measurements across visits for
the same patient. This module contains NO machine learning, NO Streamlit, and
NO database code — it is pure logic over plain data, so it can be unit tested
and audited line by line independently of the UI or persistence layer.

Clinical basis
--------------
The category thresholds are adapted from two published, non-proprietary
response-assessment protocols:

  - RANO (Response Assessment in Neuro-Oncology) for high-grade glioma,
    Wen PY et al., "Updated Response Assessment Criteria for High-Grade
    Gliomas", J Clin Oncol. 2010;28(11):1963-1972.
    Uses the product of the two largest perpendicular diameters, compared
    against the NADIR (smallest measurement on study, not the previous scan).

  - RECIST 1.1 (Response Evaluation Criteria in Solid Tumors),
    Eisenhauer EA et al., Eur J Cancer. 2009;45(2):228-247.
    Uses the single longest diameter, compared against baseline (for partial
    response) and nadir (for progression).

IMPORTANT — what this module does NOT and CANNOT determine
------------------------------------------------------------
Full RANO/RECIST response assessment also requires contrast-enhancement
characteristics, corticosteroid dose stability, and clinical/neurological
status. This system has access to none of those — only 2D bidirectional
size measurements derived from a single MRI slice's segmentation mask.
For that reason every result produced here is labeled a
"Measurement-based response assessment (RANO-inspired)", never a definitive
clinical RANO/RECIST category, and every result carries an explicit list of
caveats. This distinction is deliberate and must not be papered over in the
UI or PDF layer.
"""

from dataclasses import dataclass, field
from typing import Optional, List

# ---------------------------------------------------------------------------
# Thresholds (from the published criteria cited above)
# ---------------------------------------------------------------------------
PD_THRESHOLD_VS_NADIR_PCT = 25.0   # RANO: >=25% increase in product of perpendiculars vs nadir => Progressive Disease
PR_THRESHOLD_VS_BASELINE_PCT = -50.0  # RANO: >=50% decrease in product of perpendiculars vs baseline => Partial Response

CR = "Complete Response (CR)"
PR = "Partial Response (PR)"
SD = "Stable Disease (SD)"
PD = "Progressive Disease (PD)"
INSUFFICIENT_DATA = "Insufficient Data"

ASSESSMENT_LABEL = "Measurement-based response assessment (RANO-inspired)"

DECISION_SUPPORT_DISCLAIMER = (
    "This is an automated, measurement-based comparison of tumor size across visits, "
    "not a definitive clinical RANO/RECIST determination. It is decision-support "
    "only, is not a diagnostic system, has not been clinically validated, and does "
    "not substitute for assessment by a radiologist or oncologist, who additionally "
    "weigh contrast enhancement, corticosteroid dose, and clinical/neurological status."
)

# Segmentation was trained ONLY on the LGG (low-grade glioma) MRI Segmentation
# dataset (mateuszbuda/lgg-mri-segmentation). Classification covers a broader
# set of tumor types trained on a different, unrelated dataset. Any tumor type
# outside the segmentation model's training domain gets an explicit caveat.
SEGMENTATION_TRAINING_DOMAIN = {"glioma"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class VisitMeasurement:
    """One visit's worth of measurement data. Plain data, no behavior."""
    scan_date: str
    tumor_type: Optional[str] = None
    max_diameter_mm: Optional[float] = None
    perpendicular_diameter_mm: Optional[float] = None
    product_bidirectional_mm2: Optional[float] = None
    area_mm2: Optional[float] = None
    visit_id: Optional[int] = None


@dataclass
class ResponseAssessment:
    category: str
    assessment_label: str
    metric_used: str                       # "product_bidirectional_mm2" or "area_mm2" fallback
    current: Optional[VisitMeasurement]
    baseline: Optional[VisitMeasurement]
    previous: Optional[VisitMeasurement]
    nadir: Optional[VisitMeasurement]
    pct_change_from_baseline: Optional[float]
    pct_change_from_previous: Optional[float]
    pct_change_from_nadir: Optional[float]
    rationale: str
    caveats: List[str] = field(default_factory=list)
    disclaimer: str = DECISION_SUPPORT_DISCLAIMER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _effective_measurement(visit: VisitMeasurement) -> "tuple[Optional[float], str]":
    """
    Returns (value, metric_name). Prefers the RANO metric (product of
    perpendicular diameters); falls back to cross-sectional area if the
    product wasn't computed for that visit; returns (None, "none") if the
    visit has no measurable tumor at all (e.g. detection was negative).
    """
    if visit.product_bidirectional_mm2 is not None and visit.product_bidirectional_mm2 > 0:
        return visit.product_bidirectional_mm2, "product_bidirectional_mm2"
    if visit.area_mm2 is not None and visit.area_mm2 > 0:
        return visit.area_mm2, "area_mm2"
    return None, "none"


def _pct_change(current: Optional[float], reference: Optional[float]) -> Optional[float]:
    """Safe percentage change. Returns None if it cannot be computed (missing
    or zero reference) rather than raising or silently returning 0/inf."""
    if current is None or reference is None or reference == 0:
        return None
    return (current - reference) / reference * 100.0


def _domain_caveats(current: VisitMeasurement, history: List[VisitMeasurement]) -> List[str]:
    caveats = []
    types_seen = {v.tumor_type for v in history + [current] if v.tumor_type}
    out_of_domain = types_seen - SEGMENTATION_TRAINING_DOMAIN
    if out_of_domain:
        caveats.append(
            "Segmentation/measurement model was trained only on the LGG "
            "(low-grade glioma) MRI Segmentation dataset. Measurement reliability "
            f"has not been established for: {', '.join(sorted(out_of_domain))}. "
            "Interpret size comparisons for these tumor types with caution."
        )
    if len(types_seen) > 1:
        caveats.append(
            "Classified tumor type differs across visits ("
            f"{', '.join(sorted(types_seen))}). Size comparison across a type "
            "change should be interpreted cautiously — consult the treating physician."
        )
    return caveats


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def classify_response(history: List[VisitMeasurement]) -> ResponseAssessment:
    """
    Given a patient's visit history in ascending scan_date order (oldest
    first, current/latest last), classify the treatment-response trajectory.

    Definitions (deliberately kept distinct, per RANO):
      - baseline: the FIRST visit on record.
      - previous: the visit immediately before the current one.
      - nadir:    the SMALLEST measurement among all visits BEFORE the
                   current one (not necessarily baseline or previous).

    Returns an Insufficient Data assessment if there are fewer than 2 visits,
    or if the current visit has no measurable tumor data comparable to any
    prior visit's data in a way that would let a category be assigned.
    """
    if not history:
        return ResponseAssessment(
            category=INSUFFICIENT_DATA,
            assessment_label=ASSESSMENT_LABEL,
            metric_used="none",
            current=None, baseline=None, previous=None, nadir=None,
            pct_change_from_baseline=None, pct_change_from_previous=None, pct_change_from_nadir=None,
            rationale="No visits recorded for this patient.",
        )

    current = history[-1]
    prior_visits = history[:-1]

    if not prior_visits:
        return ResponseAssessment(
            category=INSUFFICIENT_DATA,
            assessment_label=ASSESSMENT_LABEL,
            metric_used="none",
            current=current, baseline=None, previous=None, nadir=None,
            pct_change_from_baseline=None, pct_change_from_previous=None, pct_change_from_nadir=None,
            rationale=(
                "Baseline established. A longitudinal response assessment will "
                "become available after a follow-up scan."
            ),
        )

    baseline = prior_visits[0]
    previous = prior_visits[-1]

    prior_with_values = [(_effective_measurement(v)[0], v) for v in prior_visits]
    prior_with_values = [(val, v) for val, v in prior_with_values if val is not None]
    nadir = min(prior_with_values, key=lambda pair: pair[0])[1] if prior_with_values else None

    current_val, metric_name = _effective_measurement(current)
    baseline_val, _ = _effective_measurement(baseline)
    previous_val, _ = _effective_measurement(previous)
    nadir_val, _ = _effective_measurement(nadir) if nadir is not None else (None, "none")

    pct_from_baseline = _pct_change(current_val, baseline_val)
    pct_from_previous = _pct_change(current_val, previous_val)
    pct_from_nadir = _pct_change(current_val, nadir_val)

    caveats = _domain_caveats(current, prior_visits)

    # --- No measurable tumor now ---
    # (nadir_val is None exactly when NO prior visit had a measurable value,
    # since baseline is itself one of the candidates nadir is computed from.)
    if current_val is None:
        if nadir_val is None:
            category = SD
            rationale = (
                "No measurable tumor at baseline, prior visits, or the current visit. "
                "No size change to assess."
            )
        else:
            category = CR
            rationale = (
                "No measurable tumor detected on the current scan, following prior "
                "visit(s) with measurable disease. Note: absence of detection on this "
                "screening model does not rule out microscopic or non-enhancing "
                "residual disease — clinical correlation is required before treating "
                "this as a confirmed complete response."
            )
        return ResponseAssessment(
            category=category, assessment_label=ASSESSMENT_LABEL, metric_used=metric_name,
            current=current, baseline=baseline, previous=previous, nadir=nadir,
            pct_change_from_baseline=pct_from_baseline, pct_change_from_previous=pct_from_previous,
            pct_change_from_nadir=pct_from_nadir, rationale=rationale, caveats=caveats,
        )

    # --- New measurable lesion where none existed before ---
    if nadir_val is None:
        category = PD
        rationale = (
            f"A new measurable lesion ({current_val:.1f} mm²) is present where no prior "
            "visit had measurable tumor. Per RANO, a new measurable lesion is classified "
            "as Progressive Disease."
        )
        return ResponseAssessment(
            category=category, assessment_label=ASSESSMENT_LABEL, metric_used=metric_name,
            current=current, baseline=baseline, previous=previous, nadir=nadir,
            pct_change_from_baseline=None, pct_change_from_previous=None, pct_change_from_nadir=None,
            rationale=rationale, caveats=caveats,
        )

    # --- Standard case: compare against nadir (for progression) and baseline (for response) ---
    if pct_from_nadir is not None and pct_from_nadir >= PD_THRESHOLD_VS_NADIR_PCT:
        category = PD
        rationale = (
            f"Current measurement ({current_val:.1f} mm²) is {pct_from_nadir:+.1f}% vs. the "
            f"nadir ({nadir_val:.1f} mm² on {nadir.scan_date}), meeting the RANO threshold "
            f"of ≥25% increase from nadir for Progressive Disease."
        )
    elif pct_from_baseline is not None and pct_from_baseline <= PR_THRESHOLD_VS_BASELINE_PCT:
        category = PR
        rationale = (
            f"Current measurement ({current_val:.1f} mm²) is {pct_from_baseline:+.1f}% vs. "
            f"baseline ({baseline_val:.1f} mm² on {baseline.scan_date}), meeting the RANO "
            f"threshold of ≥50% decrease from baseline for Partial Response."
        )
    else:
        category = SD
        rationale = (
            f"Current measurement ({current_val:.1f} mm²) is {pct_from_baseline:+.1f}% vs. "
            f"baseline and {pct_from_nadir:+.1f}% vs. nadir ({nadir_val:.1f} mm² on "
            f"{nadir.scan_date}) — neither the Progressive Disease threshold (≥25% "
            "vs. nadir) nor the Partial Response threshold (≥50% vs. baseline) is met."
        )

    return ResponseAssessment(
        category=category, assessment_label=ASSESSMENT_LABEL, metric_used=metric_name,
        current=current, baseline=baseline, previous=previous, nadir=nadir,
        pct_change_from_baseline=pct_from_baseline, pct_change_from_previous=pct_from_previous,
        pct_change_from_nadir=pct_from_nadir, rationale=rationale, caveats=caveats,
    )


if __name__ == "__main__":
    demo_history = [
        VisitMeasurement(scan_date="2026-01-01", tumor_type="glioma", product_bidirectional_mm2=300.0),
        VisitMeasurement(scan_date="2026-03-01", tumor_type="glioma", product_bidirectional_mm2=150.0),
        VisitMeasurement(scan_date="2026-05-01", tumor_type="glioma", product_bidirectional_mm2=220.0),
        VisitMeasurement(scan_date="2026-07-01", tumor_type="glioma", product_bidirectional_mm2=180.0),
    ]
    result = classify_response(demo_history)
    print(result.category)
    print(result.rationale)
    print("vs baseline:", result.pct_change_from_baseline)
    print("vs previous:", result.pct_change_from_previous)
    print("vs nadir:   ", result.pct_change_from_nadir)
