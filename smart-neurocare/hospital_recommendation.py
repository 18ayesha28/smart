"""
Smart NeuroCare — Smart Hospital & Specialist Recommendation Engine

A weighted multi-factor scoring system combining:
  - Medical factors: tumor type/severity match to hospital specialization
  - Patient factors: distance from patient, budget fit, insurance acceptance
  - Hospital factors: rating, success rate, specialist experience

Transparent, explainable rule-based clinical triage engine.
"""

from dataclasses import dataclass, field
from math import radians, sin, cos, sqrt, atan2
from typing import Optional


@dataclass
class PatientContext:
    latitude: float
    longitude: float
    tumor_type: str = "glioma"
    severity_score: str = "moderate"  # low, moderate, high, critical
    max_budget: Optional[float] = None
    insurance_provider: Optional[str] = None


@dataclass
class Hospital:
    hospital_id: str
    name: str
    latitude: float
    longitude: float
    specializations: list          # e.g. ["neuro-oncology", "pediatric neurosurgery"]
    rating: float                  # 0-5
    success_rate: float            # 0-100
    avg_cost_min: float
    avg_cost_max: float
    city: str = "Bengaluru"
    accepted_insurance: list = field(default_factory=list)


TUMOR_TYPE_TO_SPECIALIZATION = {
    "glioma": ["neuro-oncology", "neurosurgery"],
    "meningioma": ["neurosurgery", "neuro-oncology"],
    "pituitary": ["endocrine neurosurgery", "neurosurgery"],
    "notumor": ["general surgery", "neurosurgery"],
}

SEVERITY_WEIGHT_BOOST = {
    "low": 0.0,
    "moderate": 0.05,
    "high": 0.10,
    "critical": 0.15,
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def specialization_match_score(hospital: Hospital, tumor_type: Optional[str]) -> float:
    if not tumor_type:
        return 0.5
    relevant_specs = TUMOR_TYPE_TO_SPECIALIZATION.get(tumor_type.lower(), ["neurosurgery"])
    if not relevant_specs:
        return 0.5
    matches = sum(1 for spec in hospital.specializations if spec.lower() in relevant_specs)
    return min(matches / len(relevant_specs), 1.0)


def distance_score(distance_km: float, max_reasonable_km: float = 800.0) -> float:
    return max(0.1, 1.0 - (distance_km / max_reasonable_km))


def cost_fit_score(hospital: Hospital, max_budget: Optional[float]) -> float:
    if max_budget is None or max_budget <= 0:
        return 0.7
    if hospital.avg_cost_min <= max_budget:
        if hospital.avg_cost_max <= max_budget:
            return 1.0
        return 0.75
    return 0.2


def insurance_score(hospital: Hospital, insurance_provider: Optional[str]) -> float:
    if not insurance_provider or insurance_provider.lower() in ["none", ""]:
        return 0.5
    for ins in hospital.accepted_insurance:
        if insurance_provider.lower() in ins.lower() or ins.lower() in insurance_provider.lower():
            return 1.0
    return 0.25


def default_hospitals() -> list[Hospital]:
    return [
        Hospital(
            "h1", "NIMHANS — National Institute of Mental Health & Neurosciences",
            12.9372, 77.5906, ["neurosurgery", "neuro-oncology", "endocrine neurosurgery"],
            rating=4.9, success_rate=93.5, avg_cost_min=200000, avg_cost_max=450000,
            city="Bengaluru", accepted_insurance=["StarHealth", "HDFC Ergo", "ICICI Lombard", "Max Bupa", "Ayushman Bharat"]
        ),
        Hospital(
            "h2", "Apollo Proton Cancer & Neuroscience Centre",
            12.9606, 80.2443, ["neuro-oncology", "neurosurgery", "radiosurgery"],
            rating=4.8, success_rate=91.0, avg_cost_min=600000, avg_cost_max=950000,
            city="Chennai", accepted_insurance=["StarHealth", "HDFC Ergo", "Tata AIG", "Care Health"]
        ),
        Hospital(
            "h3", "Manipal Comprehensive Cancer Institute",
            12.9584, 77.6496, ["neurosurgery", "neuro-oncology"],
            rating=4.7, success_rate=88.0, avg_cost_min=450000, avg_cost_max=750000,
            city="Bengaluru", accepted_insurance=["StarHealth", "HDFC Ergo", "ICICI Lombard", "Bajaj Allianz"]
        ),
        Hospital(
            "h4", "Tata Memorial Hospital — ACTREC Cancer Centre",
            19.0069, 72.8427, ["neuro-oncology", "endocrine neurosurgery", "neurosurgery"],
            rating=4.9, success_rate=94.0, avg_cost_min=250000, avg_cost_max=500000,
            city="Mumbai", accepted_insurance=["StarHealth", "HDFC Ergo", "ICICI Lombard", "Ayushman Bharat", "CGHS"]
        ),
        Hospital(
            "h5", "Fortis Memorial Research Institute — Brain & Spine Centre",
            28.4595, 77.0266, ["neurosurgery", "neuro-oncology", "endocrine neurosurgery"],
            rating=4.7, success_rate=89.5, avg_cost_min=550000, avg_cost_max=850000,
            city="Gurugram / Delhi NCR", accepted_insurance=["StarHealth", "Max Bupa", "HDFC Ergo", "Care Health"]
        ),
        Hospital(
            "h6", "Max Super Speciality Hospital — Institute of Neurosciences",
            28.5284, 77.2147, ["neurosurgery", "neuro-oncology"],
            rating=4.6, success_rate=87.0, avg_cost_min=500000, avg_cost_max=800000,
            city="New Delhi", accepted_insurance=["StarHealth", "ICICI Lombard", "Bajaj Allianz", "Tata AIG"]
        ),
    ]


def recommend_hospitals(
    patient: PatientContext,
    hospitals: list[Hospital],
    top_k: int = 5,
) -> list[dict]:
    """
    Score and rank hospitals for a given patient context.

    Single, explicit signature by design: patient/tumor context is always a
    PatientContext, the hospital pool is always passed explicitly (e.g.
    default_hospitals()), and top_k controls how many results come back.
    """
    severity_boost = SEVERITY_WEIGHT_BOOST.get(str(patient.severity_score).lower(), 0.0)
    weights = {
        "specialization": 0.35 + severity_boost,
        "quality": 0.25,
        "distance": max(0.05, 0.20 - (severity_boost / 2)),
        "cost": max(0.05, 0.10 - (severity_boost / 4)),
        "insurance": max(0.05, 0.10 - (severity_boost / 4)),
    }

    results = []
    for hospital in hospitals:
        distance_km = haversine_distance_km(patient.latitude, patient.longitude, hospital.latitude, hospital.longitude)
        spec_score = specialization_match_score(hospital, patient.tumor_type)
        quality_score = (hospital.rating / 5.0) * 0.5 + (hospital.success_rate / 100.0) * 0.5
        dist_score = distance_score(distance_km)
        cost_score = cost_fit_score(hospital, patient.max_budget)
        ins_score = insurance_score(hospital, patient.insurance_provider)

        final_score = (
            spec_score * weights["specialization"]
            + quality_score * weights["quality"]
            + dist_score * weights["distance"]
            + cost_score * weights["cost"]
            + ins_score * weights["insurance"]
        )

        results.append({
            "hospital_id": hospital.hospital_id,
            "name": hospital.name,
            "city": getattr(hospital, "city", "Bengaluru"),
            "match_score": round(final_score, 4),
            "distance_km": round(distance_km, 1),
            "match_reasons": {
                "specialization_match": round(spec_score, 2),
                "hospital_quality": round(quality_score, 2),
                "distance_fit": round(dist_score, 2),
                "cost_fit": round(cost_score, 2),
                "insurance_fit": round(ins_score, 2),
            },
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    p = PatientContext(latitude=12.9716, longitude=77.5946, tumor_type="glioma")
    print(recommend_hospitals(p, default_hospitals()))
