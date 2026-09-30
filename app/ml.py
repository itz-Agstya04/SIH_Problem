"""Train and serve the reviewed-label SIF screening model."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

try:
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:  # pragma: no cover - pgmpy is optional until project install completes.
    BayesianEstimator = None  # type: ignore[assignment]
    VariableElimination = None  # type: ignore[assignment]
    DiscreteBayesianNetwork = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "sif_text_model.joblib"
METADATA_PATH = MODEL_DIR / "sif_text_model.json"
BAYESIAN_MODEL_PATH = MODEL_DIR / "sif_bayesian_model.joblib"
BAYESIAN_METADATA_PATH = MODEL_DIR / "sif_bayesian_model.json"
MINIMUM_TOTAL_LABELS = 30
MINIMUM_PER_CLASS = 10
MODEL_NAME = "tfidf-logistic-sif"
BAYESIAN_MODEL_NAME = "pgmpy-discrete-bayesian-sif"


class ModelQualityError(ValueError):
    """Raised when a saved model has no demonstrated signal in its validation."""


def build_feature_text(
    description: str,
    activity: str = "",
    report_type: str = "",
    factors: dict[str, list[dict[str, str]]] | None = None,
) -> str:
    """Combine narrative with extracted concepts, excluding site-specific identifiers."""
    parts = [description]
    if activity and activity.casefold() != "unknown":
        parts.append(f"activity {activity}")
    if report_type and report_type.casefold() != "unknown":
        parts.append(f"report type {report_type}")
    for category, concepts in (factors or {}).items():
        parts.extend(f"{category.replace('_', ' ')} {item['concept']}" for item in concepts)
    return " ".join(parts)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )


def evaluate_cross_validated(texts: list[str], labels: list[int]) -> dict[str, Any]:
    class_counts = {int(label): labels.count(label) for label in set(labels)}
    if len(class_counts) != 2:
        raise ValueError("Training data must contain both SIF and non-SIF labels")
    folds = min(5, min(class_counts.values()))
    if folds < 2:
        raise ValueError("At least two examples of each label are required for cross-validation")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_predict(make_pipeline(), texts, labels, cv=splitter, method="predict_proba")[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return {
        "evaluation": "stratified cross-validation; random folds, not temporal or site holdout",
        "folds": folds,
        "sample_count": len(labels),
        "class_counts": {"not_sif_potential": class_counts.get(0, 0), "sif_potential": class_counts.get(1, 0)},
        "threshold": 0.5,
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "average_precision": round(float(average_precision_score(labels, scores)), 4),
        "brier_score": round(float(brier_score_loss(labels, scores)), 4),
        "score_interpretation": "Model score is not calibrated and must not be interpreted as a probability.",
    }


def _normalise_yes_no(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"yes", "true", "1", "sif_potential"}:
            return "yes"
        if text in {"no", "false", "0", "not_sif_potential"}:
            return "no"
        if text in {"unknown", "null", "none", ""}:
            return "unknown"
        return "yes" if text else "no"
    return "yes" if bool(value) else "no"


def build_bayesian_feature_row(
    description: str,
    activity: str = "",
    report_type: str = "",
    factors: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Convert a report into the discrete variables the Bayesian layer expects."""
    lowered = (description or "").casefold()
    record: dict[str, Any] = {
        "activity": (activity or "unknown").strip() or "unknown",
        "hazard": "no",
        "energy": "no",
        "barrier_failure": "no",
        "exposure": "no",
        "consequence": "no",
        "sif_label": "unknown",
    }
    if factors:
        for category, key in (("hazard", "hazard"), ("energy", "energy"), ("barrier_failure", "barrier_failure"), ("exposure", "exposure"), ("potential_consequence", "consequence")):
            if factors.get(category):
                record[key] = "yes"
    if any(token in lowered for token in ("energized", "energised", "live panel", "shock", "electrical", "fire", "gas", "explosion")):
        record["hazard"] = "yes"
    if any(token in lowered for token in ("loto", "lockout", "isolation", "energized", "live panel", "shock", "electrical")):
        record["energy"] = "yes"
    if any(token in lowered for token in ("not verified", "verification not performed", "wrong breaker", "bypassed", "barrier failed", "guardrail failed", "no permit")):
        record["barrier_failure"] = "yes"
    if any(token in lowered for token in ("entered the area", "worker", "exposed", "line of fire", "beneath", "under the load", "near miss")):
        record["exposure"] = "yes"
    if any(token in lowered for token in ("serious", "fatal", "injury", "explosion", "fire", "toxic", "fall from height")):
        record["consequence"] = "yes"
    return record


def train_bayesian_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Train a discrete Bayesian network from extracted safety-factor rows."""
    if DiscreteBayesianNetwork is None or BayesianEstimator is None or VariableElimination is None:
        raise ImportError("pgmpy is required for the Bayesian layer. Install it with pip install pgmpy.")
    if not rows:
        raise ValueError("At least one report row is required to train the Bayesian layer")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        safe_row = dict(row)
        safe_row["activity"] = str(safe_row.get("activity") or "unknown").strip() or "unknown"
        safe_row["hazard"] = _normalise_yes_no(safe_row.get("hazard", "no"))
        safe_row["energy"] = _normalise_yes_no(safe_row.get("energy", "no"))
        safe_row["barrier_failure"] = _normalise_yes_no(safe_row.get("barrier_failure", "no"))
        safe_row["exposure"] = _normalise_yes_no(safe_row.get("exposure", "no"))
        label = safe_row.get("sif_label") or safe_row.get("human_label") or safe_row.get("label") or "unknown"
        if isinstance(label, str) and label.casefold() == "sif_potential":
            safe_row["sif_label"] = "sif_potential"
        elif isinstance(label, str) and label.casefold() == "not_sif_potential":
            safe_row["sif_label"] = "not_sif_potential"
        elif label in (0, 1):
            safe_row["sif_label"] = "sif_potential" if int(label) == 1 else "not_sif_potential"
        else:
            safe_row["sif_label"] = "unknown"
        normalized_rows.append(safe_row)

    frame = pd.DataFrame(normalized_rows)
    required = ["activity", "hazard", "energy", "barrier_failure", "exposure", "sif_label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training rows are missing required Bayesian columns: {missing}")

    if frame["sif_label"].eq("unknown").any():
        frame = frame[frame["sif_label"] != "unknown"]
    if frame.empty:
        raise ValueError("Training rows must include explicit SIF labels")

    state_names = {
        "activity": sorted(frame["activity"].dropna().astype(str).unique().tolist()),
        "hazard": ["no", "yes"],
        "energy": ["no", "yes"],
        "barrier_failure": ["no", "yes"],
        "exposure": ["no", "yes"],
        "sif_label": ["not_sif_potential", "sif_potential"],
    }
    model = DiscreteBayesianNetwork([
        ("activity", "hazard"),
        ("hazard", "energy"),
        ("energy", "barrier_failure"),
        ("barrier_failure", "exposure"),
        ("hazard", "sif_label"),
        ("exposure", "sif_label"),
    ])
    estimator = BayesianEstimator(model, frame, state_names=state_names)
    model.add_cpds(*estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=5))
    model_version = f"{BAYESIAN_MODEL_NAME}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    payload = {"model": model, "state_names": state_names, "model_name": BAYESIAN_MODEL_NAME, "model_version": model_version}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_DIR, suffix=".joblib", delete=False) as temporary:
        temporary_model = Path(temporary.name)
    temporary_metadata = temporary_model.with_suffix(".json")
    try:
        joblib.dump(payload, temporary_model)
        temporary_metadata.write_text(json.dumps({"model_name": BAYESIAN_MODEL_NAME, "model_version": model_version, "trained_at": datetime.now(UTC).isoformat(), "state_names": state_names}, indent=2), encoding="utf-8")
        os.replace(temporary_model, BAYESIAN_MODEL_PATH)
        os.replace(temporary_metadata, BAYESIAN_METADATA_PATH)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return payload


def predict_bayesian_score(data: dict[str, Any] | str, model: Any | None = None) -> dict[str, Any]:
    """Predict SIF probability using the discrete Bayesian network."""
    if DiscreteBayesianNetwork is None or BayesianEstimator is None or VariableElimination is None:
        raise ImportError("pgmpy is required for the Bayesian layer. Install it with pip install pgmpy.")
    if model is None:
        if not BAYESIAN_MODEL_PATH.is_file() or not BAYESIAN_METADATA_PATH.is_file():
            raise FileNotFoundError("No trained Bayesian SIF model is available")
        payload = joblib.load(BAYESIAN_MODEL_PATH)
        if isinstance(payload, dict):
            model = payload["model"]
            model_version = payload.get("model_version", "bayesian-untrained")
        else:
            model = payload
            model_version = "bayesian-untrained"
    else:
        model_version = "bayesian-direct"

    if isinstance(data, str):
        row = build_bayesian_feature_row(data)
    elif isinstance(data, dict):
        row = build_bayesian_feature_row(
            str(data.get("raw_description") or data.get("description") or data.get("narrative") or ""),
            activity=str(data.get("activity") or "unknown"),
            report_type=str(data.get("report_type") or "unknown"),
            factors=data.get("factors") or {},
        )
        row["hazard"] = _normalise_yes_no(data.get("hazard", row["hazard"]))
        row["energy"] = _normalise_yes_no(data.get("energy", row["energy"]))
        row["barrier_failure"] = _normalise_yes_no(data.get("barrier_failure", row["barrier_failure"]))
        row["exposure"] = _normalise_yes_no(data.get("exposure", row["exposure"]))
        row["consequence"] = _normalise_yes_no(data.get("consequence", row["consequence"]))
    else:
        raise TypeError("Bayesian input must be a narrative string or report dictionary")

    evidence = {key: value for key, value in {
        "activity": row.get("activity", "unknown"),
        "hazard": row.get("hazard", "no"),
        "energy": row.get("energy", "no"),
        "barrier_failure": row.get("barrier_failure", "no"),
        "exposure": row.get("exposure", "no"),
    }.items() if value not in {"unknown", None, ""}}
    query = VariableElimination(model).query(["sif_label"], evidence=evidence, show_progress=False)
    label_states = list(model.states["sif_label"])
    probabilities = {state: float(value) for state, value in zip(label_states, query.values, strict=True)}
    score = probabilities.get("sif_potential", max(probabilities.values(), default=0.0))
    if BAYESIAN_METADATA_PATH.is_file():
        metadata = json.loads(BAYESIAN_METADATA_PATH.read_text(encoding="utf-8"))
        model_version = metadata.get("model_version", model_version)
    return {
        "model_name": BAYESIAN_MODEL_NAME,
        "model_version": model_version,
        "model_score": round(float(score), 4),
        "threshold": 0.5,
        "prediction": "sif_potential" if score >= 0.5 else "not_sif_potential",
        "calibrated": False,
        "notice": "Bayesian layer estimates conditional risk from observed safety factors; it is an analytical support model, not a causal proof.",
    }


def get_bayesian_model_status() -> dict[str, Any]:
    trained = BAYESIAN_MODEL_PATH.is_file() and BAYESIAN_METADATA_PATH.is_file()
    metadata = json.loads(BAYESIAN_METADATA_PATH.read_text(encoding="utf-8")) if trained else None
    return {
        "model_name": BAYESIAN_MODEL_NAME,
        "trained": trained,
        "model_version": metadata["model_version"] if metadata else None,
        "trained_at": metadata.get("trained_at") if metadata else None,
        "message": "Bayesian layer ready" if trained else "Bayesian layer not trained yet",
    }


def train_and_save(texts: list[str], labels: list[int]) -> dict[str, Any]:
    if len(texts) != len(labels):
        raise ValueError("Each training text must have exactly one label")
    if len(labels) < MINIMUM_TOTAL_LABELS:
        raise ValueError(f"At least {MINIMUM_TOTAL_LABELS} reviewed real reports are required")
    class_counts = {int(label): labels.count(label) for label in set(labels)}
    if len(class_counts) != 2 or min(class_counts.values(), default=0) < MINIMUM_PER_CLASS:
        raise ValueError(f"At least {MINIMUM_PER_CLASS} examples of each SIF label are required")

    metrics = evaluate_cross_validated(texts, labels)
    estimator = make_pipeline()
    estimator.fit(texts, labels)
    digest_input = "\n".join(f"{label}\t{text}" for text, label in zip(texts, labels, strict=True))
    training_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    model_version = f"{MODEL_NAME}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{training_digest}"
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_sample_count": len(labels),
        "class_counts": metrics["class_counts"],
        "metrics": metrics,
        "calibrated": False,
        "training_data_digest": training_digest,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_DIR, suffix=".joblib", delete=False) as temporary:
        temporary_model = Path(temporary.name)
    temporary_metadata = temporary_model.with_suffix(".json")
    try:
        joblib.dump(estimator, temporary_model)
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_model, MODEL_PATH)
        os.replace(temporary_metadata, METADATA_PATH)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return metadata


def get_model_status(labeled_count: int, class_counts: dict[str, int]) -> dict[str, Any]:
    ready_to_train = labeled_count >= MINIMUM_TOTAL_LABELS and all(
        class_counts.get(label, 0) >= MINIMUM_PER_CLASS
        for label in ("sif_potential", "not_sif_potential")
    )
    trained = MODEL_PATH.is_file() and METADATA_PATH.is_file()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if trained else None
    metrics = metadata["metrics"] if metadata else None
    has_basic_signal = bool(metrics and metrics["f1"] > 0 and metrics["roc_auc"] > 0.5)
    prediction_ready = trained and has_basic_signal
    if not trained:
        message = "Not trained: add reviewed, non-demo reports for both SIF labels."
    elif not has_basic_signal:
        message = "Saved model blocked: cross-validation did not show skill above a no-skill baseline."
    else:
        message = "Model has basic cross-validation signal; expert validation is still required."
    return {
        "model_name": MODEL_NAME,
        "trained": trained,
        "prediction_ready": prediction_ready,
        "ready_to_train": ready_to_train,
        "eligible_labeled_reports": labeled_count,
        "required_total": MINIMUM_TOTAL_LABELS,
        "required_per_class": MINIMUM_PER_CLASS,
        "class_counts": class_counts,
        "model_version": metadata["model_version"] if metadata else None,
        "trained_at": metadata["trained_at"] if metadata else None,
        "metrics": metrics,
        "calibrated": False,
        "message": message,
    }


def predict_score(text: str) -> dict[str, Any]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("No trained SIF model is available")
    estimator = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = metadata.get("metrics", {})
    if metrics.get("f1", 0) <= 0 or metrics.get("roc_auc", 0.5) <= 0.5:
        raise ModelQualityError("Saved model is blocked because validation did not show skill above a no-skill baseline")
    score = float(estimator.predict_proba([text])[0][1])
    return {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "model_score": round(score, 4),
        "threshold": 0.5,
        "prediction": "sif_potential" if score >= 0.5 else "not_sif_potential",
        "calibrated": False,
        "notice": "Uncalibrated model score for expert triage; not a risk probability or safety decision.",
    }
