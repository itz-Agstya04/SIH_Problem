"""Transparent prototype rules for extracting safety evidence and demo triage scores."""

from __future__ import annotations

import re
from typing import Any

MODEL_VERSION = "demo-rules-0.1"

# Provisional labels for the prototype only. Replace with OIL-approved references.
LIFE_SAVING_RULES: dict[str, tuple[str, ...]] = {
    "Energy Isolation": ("loto", "lockout", "lock out", "tagout", "tag out", "isolation", "energized", "energised", "live panel"),
    "Line of Fire": ("line of fire", "suspended load", "struck by", "pinch point", "between equipment"),
    "Working at Height": ("work at height", "working at height", "scaffold", "ladder", "fall from height"),
    "Permit to Work": ("permit to work", "work permit", "confined space", "permit not"),
    "Driving Safety": ("driving", "vehicle", "seat belt", "seatbelt", "reversing", "road"),
    "Hot Work": ("hot work", "welding", "grinding", "flame", "spark"),
}

TAXONOMY: dict[str, dict[str, tuple[str, ...]]] = {
    "activity": {
        "Maintenance": ("maintenance", "repair", "servicing"),
        "Electrical work": ("electrical work", "electrical panel", "breaker", "switchgear"),
        "Lifting operation": ("lifting", "crane", "hoist", "suspended load"),
        "Work at height": ("work at height", "scaffold", "ladder"),
        "Driving": ("driving", "vehicle", "reversing"),
        "Excavation": ("excavation", "trenching", "excavator"),
        "Hot work": ("hot work", "welding", "grinding"),
    },
    "hazard": {
        "Unexpected energization": ("unexpected energization", "unexpected energisation", "energized", "energised", "live panel", "backfeed"),
        "Dropped or suspended load": ("suspended load", "dropped load", "load fell", "lifting"),
        "Fall from height": ("fall from height", "fell from", "working at height", "scaffold", "ladder"),
        "Vehicle interaction": ("vehicle", "reversing", "mobile equipment", "traffic"),
        "Hazardous atmosphere": ("confined space", "gas detected", "oxygen level", "hazardous atmosphere"),
        "Fire or ignition": ("ignition", "fire", "spark", "welding", "hot work"),
        "Stored energy release": ("pressurized", "pressurised", "pressure release", "stored energy"),
    },
    "energy": {
        "Electrical energy": ("electrical", "energized", "energised", "live panel", "breaker", "voltage"),
        "Gravity / kinetic energy": ("suspended load", "dropped load", "fall from", "vehicle", "moving equipment"),
        "Pressure energy": ("pressurized", "pressurised", "pressure release", "hydraulic", "pneumatic"),
        "Thermal / ignition energy": ("hot work", "welding", "grinding", "hot surface", "spark"),
        "Chemical energy": ("chemical", "gas release", "flammable", "toxic"),
    },
    "barrier": {
        "Energy isolation (LOTO)": ("loto", "lockout", "lock out", "tagout", "tag out", "isolation"),
        "Permit to work": ("permit to work", "work permit", "permit"),
        "Guarding": ("guard", "barrier", "barricade", "guardrail"),
        "Personal protective equipment": ("ppe", "safety glasses", "face shield", "harness"),
        "Gas testing": ("gas test", "gas testing", "gas detector", "atmosphere test"),
    },
    "barrier_failure": {
        "Verification not performed": ("not verified", "verification not performed", "failed to verify", "verification was not done"),
        "Barrier missing or absent": ("no loto", "no isolation", "without a permit", "guard was missing", "barrier was absent"),
        "Barrier incorrectly applied": ("wrong breaker", "wrong isolation point", "incorrectly isolated", "incorrect isolation"),
        "Barrier bypassed": ("bypassed", "bypass", "defeated", "overridden"),
        "Barrier failed or inadequate": ("barrier failed", "guard failed", "inadequate barrier", "failed barrier"),
    },
    "exposure": {
        "Worker in hazard zone": ("worker entered", "technician entered", "entered the area", "in the area", "exposed", "near miss", "near-miss"),
        "Person near moving equipment": ("near the vehicle", "in the line of fire", "beneath the load", "under the load"),
    },
    "potential_consequence": {
        "Serious electrical injury": ("electric shock", "electrocut", "electrical injury", "energized panel"),
        "Fatal or serious fall": ("fall from height", "fell from", "dropped from scaffold"),
        "Struck-by or crush injury": ("struck by", "crushed", "suspended load", "pinch point"),
        "Fire or explosion injury": ("explosion", "flash fire", "fire", "ignition"),
        "Toxic exposure": ("toxic", "gas exposure", "inhalation", "chemical exposure"),
    },
}


def _find_evidence(text: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in sorted(phrases, key=len, reverse=True):
        match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def estimate_bayesian_sif_probability(
    factors: dict[str, list[dict[str, str]]] | None = None,
    activity: str = "",
) -> dict[str, Any]:
    """Estimate a Bayesian-style posterior from extracted safety-factor evidence."""
    factors = factors or {}
    hazard = bool(factors.get("hazard"))
    energy = bool(factors.get("energy"))
    barrier = bool(factors.get("barrier"))
    barrier_failure = bool(factors.get("barrier_failure"))
    exposure = bool(factors.get("exposure"))
    consequence = bool(factors.get("potential_consequence"))

    probability = 0.08
    probability += 0.14 * int(hazard)
    probability += 0.12 * int(energy)
    probability += 0.12 * int(barrier)
    probability += 0.18 * int(barrier_failure)
    probability += 0.18 * int(exposure)
    probability += 0.10 * int(consequence)
    if barrier_failure and not barrier:
        probability += 0.05
    activity_key = (activity or "").strip().casefold()
    if activity_key in {"maintenance", "electrical work", "lifting operation", "hot work", "excavation", "driving", "work at height"}:
        probability += 0.05

    probability = min(max(probability, 0.0), 0.98)
    return {
        "probability": round(probability, 4),
        "prediction": "sif_potential" if probability >= 0.5 else "not_sif_potential",
        "model_version": "bayesian-factor-rules-0.1",
    }


def analyze_report(description: str) -> dict[str, Any]:
    """Extract prototype concepts with spans and calculate a transparent screening score."""
    raw = description.strip()
    normalized = re.sub(r"\s+", " ", raw)
    factors: dict[str, list[dict[str, str]]] = {}

    for category, concepts in TAXONOMY.items():
        matches = []
        for concept, phrases in concepts.items():
            evidence = _find_evidence(normalized, phrases)
            if evidence:
                matches.append({"concept": concept, "evidence": evidence})
        factors[category] = matches

    rule_scores: list[dict[str, Any]] = []
    for rule, phrases in LIFE_SAVING_RULES.items():
        evidence = _find_evidence(normalized, phrases)
        rule_scores.append({"rule": rule, "score": 0.88 if evidence else 0.02, "evidence": evidence})
    rule_scores.sort(key=lambda item: item["score"], reverse=True)

    points = 0.08
    points += 0.10 if factors["hazard"] else 0.0
    points += 0.12 if factors["energy"] else 0.0
    points += 0.16 if factors["barrier"] else 0.0
    points += 0.24 if factors["barrier_failure"] else 0.0
    points += 0.18 if factors["exposure"] else 0.0
    points += 0.12 if factors["potential_consequence"] else 0.0
    score = round(min(points, 0.96), 2)

    evidence = [
        {"category": category, **factor}
        for category, items in factors.items()
        for factor in items
    ]
    bayesian = estimate_bayesian_sif_probability(factors, activity="")
    return {
        "normalized_description": normalized,
        "factors": factors,
        "rule_scores": rule_scores,
        "evidence": evidence,
        "sif_score": score,
        "bayesian_probability": bayesian["probability"],
        "bayesian_prediction": bayesian["prediction"],
        "bayesian_model_version": bayesian["model_version"],
        "model_version": MODEL_VERSION,
        "score_notice": "Illustrative demo screening score. Neither trained nor calibrated, and not approved for operational safety decisions.",
    }


def derive_patterns(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group reviewed reports by extracted barrier-failure concepts."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        factors = report.get("factors", {})
        failures = factors.get("barrier_failure", [])
        keys = [factor["concept"] for factor in failures] or ["No barrier failure extracted"]
        for key in keys:
            groups.setdefault(key, []).append(report)

    result = []
    for name, members in groups.items():
        if len(members) < 2:
            continue
        scores = [float(member.get("sif_score", 0)) for member in members]
        result.append({
            "name": name,
            "count": len(members),
            "mean_screening_score": round(sum(scores) / len(scores), 2),
            "sites": sorted({member.get("site") or "Unknown" for member in members}),
            "activities": sorted({member.get("activity") or "Unknown" for member in members}),
            "report_ids": [member["report_id"] for member in members],
        })
    return sorted(result, key=lambda item: item["count"], reverse=True)
