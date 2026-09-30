from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml import METADATA_PATH, MODEL_PATH, predict_bayesian_score

DATA_PATH = ROOT / "External_dataset.json"
OUTPUT_PATH = ROOT / "data" / "external_dataset_summary.json"

FATALITY_KEYWORDS = (
    "killed",
    "fatal",
    "died",
    "death",
    "injuries",
    "hospitalized",
    "fatality",
    "amputated",
    "crushed",
    "electrocuted",
    "brain",
    "traumatic",
)


def _iter_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _extract_text(record: dict[str, Any]) -> str:
    report = record.get("accident_report", {})
    texts = report.get("text", [])
    if isinstance(texts, str):
        return texts
    return " ".join(str(item) for item in texts if item)


def _accident_type(record: dict[str, Any]) -> str:
    value = record.get("accident_type", {}).get("value") or ["unknown"]
    return str(value[0])


def _has_fatality_signal(text: str) -> bool:
    lowered = text.casefold()
    return any(keyword in lowered for keyword in FATALITY_KEYWORDS)


def _supervised_status() -> dict[str, Any]:
    metadata = {}
    if METADATA_PATH.is_file():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    return {
        "trained": MODEL_PATH.is_file() and METADATA_PATH.is_file(),
        "prediction_ready": bool(metrics.get("f1", 0) > 0 and metrics.get("roc_auc", 0.5) > 0.5),
        "model_version": metadata.get("model_version"),
        "metrics": metrics,
        "message": "Blocked: supervised model needs reviewed data with skill above a no-skill baseline before it can be used for external validation.",
    }


def evaluate_external_dataset() -> dict[str, Any]:
    records = _iter_records(DATA_PATH)
    accident_types: Counter[str] = Counter()
    fatality_like = 0
    bayesian_predictions: Counter[str] = Counter()
    bayesian_scores: list[float] = []
    bayesian_by_type: dict[str, list[float]] = {}

    for record in records:
        text = _extract_text(record)
        accident_type = _accident_type(record)
        accident_types[accident_type] += 1
        if _has_fatality_signal(text):
            fatality_like += 1

        result = predict_bayesian_score(text)
        bayesian_scores.append(float(result["model_score"]))
        bayesian_predictions[result["prediction"]] += 1
        bayesian_by_type.setdefault(accident_type, []).append(float(result["model_score"]))

    score_summary = {
        "total_records": len(records),
        "fatality_like_records": fatality_like,
        "fatality_like_fraction": round(fatality_like / len(records), 4) if records else 0.0,
        "bayesian_high_risk_count": bayesian_predictions.get("sif_potential", 0),
        "bayesian_high_risk_fraction": round(bayesian_predictions.get("sif_potential", 0) / len(records), 4) if records else 0.0,
        "bayesian_mean_score": round(sum(bayesian_scores) / len(bayesian_scores), 4) if bayesian_scores else 0.0,
        "bayesian_min_score": round(min(bayesian_scores), 4) if bayesian_scores else 0.0,
        "bayesian_max_score": round(max(bayesian_scores), 4) if bayesian_scores else 0.0,
        "predictive_distribution": {key: int(value) for key, value in sorted(bayesian_predictions.items())},
    }

    top_types = {key: int(value) for key, value in accident_types.most_common(10)}
    by_type_risk = {
        key: {
            "count": len(values),
            "mean_score": round(sum(values) / len(values), 4),
            "high_risk_count": sum(1 for value in values if value >= 0.5),
            "high_risk_fraction": round(sum(1 for value in values if value >= 0.5) / len(values), 4),
        }
        for key, values in sorted(bayesian_by_type.items())
    }

    summary = {
        "source": str(DATA_PATH),
        "total_records": len(records),
        "accident_type_counts": top_types,
        "fatality_signal": {
            "count": fatality_like,
            "fraction": round(fatality_like / len(records), 4) if records else 0.0,
        },
        "bayesian_model": {
            "model_name": "pgmpy-discrete-bayesian-sif",
            "threshold": 0.5,
            "average_score": score_summary["bayesian_mean_score"],
            "min_score": score_summary["bayesian_min_score"],
            "max_score": score_summary["bayesian_max_score"],
            "high_risk_count": score_summary["bayesian_high_risk_count"],
            "high_risk_fraction": score_summary["bayesian_high_risk_fraction"],
            "prediction_distribution": score_summary["predictive_distribution"],
            "by_accident_type": by_type_risk,
        },
        "supervised_model": _supervised_status(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = evaluate_external_dataset()
    print(json.dumps({
        "total_records": result["total_records"],
        "fatality_like_records": result["fatality_signal"]["count"],
        "bayesian_high_risk_count": result["bayesian_model"]["high_risk_count"],
        "bayesian_high_risk_fraction": result["bayesian_model"]["high_risk_fraction"],
        "bayesian_mean_score": result["bayesian_model"]["average_score"],
        "supervised_prediction_ready": result["supervised_model"]["prediction_ready"],
    }, indent=2))
