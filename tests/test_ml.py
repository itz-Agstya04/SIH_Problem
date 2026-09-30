import pytest

from app import ml
from app.ml import evaluate_cross_validated, train_and_save


def test_cross_validation_evaluates_two_class_safety_texts():
    texts = [
        f"worker exposed energized panel and isolation failed example {index}"
        if index % 2
        else f"routine housekeeping observation with no exposure example {index}"
        for index in range(40)
    ]
    labels = [index % 2 for index in range(40)]

    metrics = evaluate_cross_validated(texts, labels)

    assert metrics["folds"] == 5
    assert metrics["sample_count"] == 40
    assert 0 <= metrics["f1"] <= 1
    assert "not calibrated" in metrics["score_interpretation"]


def test_training_requires_minimum_reviewed_dataset():
    with pytest.raises(ValueError, match="At least 30"):
        train_and_save(["text"] * 4, [0, 0, 1, 1])


def test_training_requires_both_labels_in_sufficient_numbers():
    with pytest.raises(ValueError, match="examples of each"):
        train_and_save([f"report {index}" for index in range(30)], [0] * 21 + [1] * 9)


def test_train_save_and_predict_uses_temporary_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "sif.joblib")
    monkeypatch.setattr(ml, "METADATA_PATH", tmp_path / "sif.json")
    texts = [
        f"worker exposed energized panel and isolation failed sample {index}"
        if index % 2
        else f"routine housekeeping with no exposure sample {index}"
        for index in range(40)
    ]
    labels = [index % 2 for index in range(40)]

    metadata = ml.train_and_save(texts, labels)
    prediction = ml.predict_score("worker exposed energized panel and isolation failed")

    assert metadata["model_version"].startswith("tfidf-logistic-sif-")
    assert metadata["metrics"]["folds"] == 5
    assert prediction["model_version"] == metadata["model_version"]
    assert prediction["prediction"] == "sif_potential"
    assert prediction["calibrated"] is False
