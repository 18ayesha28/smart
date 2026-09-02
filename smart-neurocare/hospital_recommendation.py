"""
Smart NeuroCare — Smart Hospital & Specialist Recommendation Engine

A weighted multi-factor scoring system combining:
  - Medical factors: tumor type/severity match to hospital specialization
  - Patient factors: distance from patient, budget fit, insurance acceptance
  - Hospital factors: rating, success rate, specialist experience

This is a transparent, explainable rule-based scorer (recommended over a
black-box model for healthcare recommendations, since patients/doctors
need to understand *why* a hospital was recommended).
"""

from dataclasses import dataclass, field
from math import radians, sin, cos, sqrt, atan2
from typing import Optional


@dataclass
class PatientContext:
    latitude: float
    longitude: float
    tumor_type: str
    severity_score: str  # low, moderate, high, critical
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
    accepted_insurance: list = field(default_factory=list)


TUMOR_TYPE_TO_SPECIALIZATION = {
    "glioma": ["neuro-oncology", "neurosurgery"],
    "meningioma": ["neurosurgery", "neuro-oncology"],
    "pituitary": ["endocrine neurosurgery", "neurosurgery"],
}

SEVERITY_WEIGHT_BOOST = {
    "low": 0.0,
    "moderate": 0.05,
    "high": 0.10,
    "critical": 0.15,  # for critical cases, weight hospital success rate/specialization more heavily
}


def haversine_distance_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def specialization_match_score(hospital: Hospital, tumor_type: str) -> float:
    relevant_specs = TUMOR_TYPE_TO_SPECIALIZATION.get(tumor_type.lower(), [])
    if not relevant_specs:
        return 0.5  # unknown tumor type - neutral score
    matches = sum(1 for spec in hospital.specializations if spec.lower() in relevant_specs)
    return min(matches / len(relevant_specs), 1.0)


def distance_score(distance_km: float, max_reasonable_km: float = 300.0) -> float:
    """Closer is better; score decays linearly, floors at 0."""
    return max(0.0, 1.0 - (distance_km / max_reasonable_km))


def cost_fit_score(hospital: Hospital, max_budget: Optional[float]) -> float:
    if max_budget is None:
        return 0.7  # neutral if patient hasn't specified a budget
    if hospital.avg_cost_min <= max_budget:
        # Fully affordable if even the max estimated cost fits; partial credit otherwise
        if hospital.avg_cost_max <= max_budget:
            return 1.0
        return 0.6
    return 0.1  # even minimum cost exceeds budget


def insurance_score(hospital: Hospital, insurance_provider: Optional[str]) -> float:
    if not insurance_provider:
        return 0.5
    return 1.0 if insurance_provider in hospital.accepted_insurance else 0.2


def recommend_hospitals(patient: PatientContext, hospitals: list[Hospital], top_k: int = 5):
    """
    Weighted scoring:
      specialization_match: 0.35 (+severity boost)
      hospital_quality (rating + success_rate): 0.25
      distance: 0.20
      cost_fit: 0.10
      insurance: 0.10
    """
    severity_boost = SEVERITY_WEIGHT_BOOST.get(patient.severity_score.lower(), 0.0)
    weights = {
        "specialization": 0.35 + severity_boost,
        "quality": 0.25,
        "distance": 0.20 - (severity_boost / 2),  # for critical cases, distance matters slightly less
        "cost": 0.10 - (severity_boost / 4),
        "insurance": 0.10 - (severity_boost / 4),
    }

    results = []
    for hospital in hospitals:
        distance_km = haversine_distance_km(
            patient.latitude, patient.longitude, hospital.latitude, hospital.longitude
        )

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
    patient = PatientContext(
        latitude=12.9716, longitude=77.5946,  # Bengaluru
        tumor_type="glioma",
        severity_score="high",
        max_budget=800000,
        insurance_provider="StarHealth",
    )

    hospitals = [
        Hospital("h1", "NeuroCare Institute", 12.9352, 77.6245,
                 ["neuro-oncology", "neurosurgery"], rating=4.7, success_rate=88,
                 avg_cost_min=500000, avg_cost_max=750000,
                 accepted_insurance=["StarHealth", "HDFC Ergo"]),
        Hospital("h2", "City General Hospital", 13.0827, 80.2707,
                 ["general surgery"], rating=4.0, success_rate=70,
                 avg_cost_min=300000, avg_cost_max=500000,
                 accepted_insurance=["ICICI Lombard"]),
    ]

    for rec in recommend_hospitals(patient, hospitals):
        print(rec)
