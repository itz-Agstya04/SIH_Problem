import pytest

from app.ml import build_bayesian_feature_row, predict_bayesian_score, train_bayesian_model


@pytest.fixture
def event_rows():
    return [
        {
            "activity": "Maintenance",
            "hazard": "yes",
            "energy": "yes",
            "barrier_failure": "yes",
            "exposure": "yes",
            "sif_label": "sif_potential",
        },
        {
            "activity": "Maintenance",
            "hazard": "yes",
            "energy": "yes",
            "barrier_failure": "yes",
            "exposure": "yes",
            "sif_label": "sif_potential",
        },
        {
            "activity": "Administration",
            "hazard": "no",
            "energy": "no",
            "barrier_failure": "no",
            "exposure": "no",
            "sif_label": "not_sif_potential",
        },
        {
            "activity": "Administration",
            "hazard": "no",
            "energy": "no",
            "barrier_failure": "no",
            "exposure": "no",
            "sif_label": "not_sif_potential",
        },
    ]


def test_build_bayesian_feature_row_extracts_boolean_risk_signals():
    row = build_bayesian_feature_row(
        "Technician entered the area while the panel was energized and LOTO was not verified.",
        activity="Maintenance",
        factors={
            "hazard": [{"concept": "Unexpected energization"}],
            "energy": [{"concept": "Electrical energy"}],
            "barrier_failure": [{"concept": "Verification not performed"}],
            "exposure": [{"concept": "Worker in hazard zone"}],
        },
    )

    assert row["activity"] == "Maintenance"
    assert row["hazard"] == "yes"
    assert row["energy"] == "yes"
    assert row["barrier_failure"] == "yes"
    assert row["exposure"] == "yes"
    assert row["sif_label"] == "unknown"


def test_train_and_predict_bayesian_model(event_rows):
    training = train_bayesian_model(event_rows)
    result = predict_bayesian_score(
        {
            "activity": "Maintenance",
            "hazard": "yes",
            "energy": "yes",
            "barrier_failure": "yes",
            "exposure": "yes",
        },
        model=training["model"],
    )

    assert result["prediction"] == "sif_potential"
    assert 0.0 <= result["model_score"] <= 1.0
    assert "bayesian" in result["model_name"].casefold()
