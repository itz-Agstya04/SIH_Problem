from app.engine import analyze_report, derive_patterns


def test_analysis_keeps_evidence_and_detects_isolation_failure():
    result = analyze_report(
        "During maintenance, a technician entered the area while the panel was energized. LOTO was not verified."
    )

    assert result["factors"]["barrier_failure"][0]["concept"] == "Verification not performed"
    assert result["factors"]["barrier_failure"][0]["evidence"] == "not verified"
    assert result["factors"]["energy"][0]["concept"] == "Electrical energy"
    assert result["sif_score"] >= 0.5
    assert "neither trained nor calibrated" in result["score_notice"].casefold()


def test_analysis_returns_unknown_factors_for_unmatched_narrative():
    result = analyze_report("Routine administrative update completed.")

    assert all(not factors for factors in result["factors"].values())
    assert result["sif_score"] == 0.08


def test_patterns_group_repeated_barrier_failure():
    reports = [
        {"report_id": "A", "site": "North", "activity": "Maintenance", "sif_score": 0.8, "factors": {"barrier_failure": [{"concept": "Verification not performed"}]}},
        {"report_id": "B", "site": "South", "activity": "Electrical work", "sif_score": 0.6, "factors": {"barrier_failure": [{"concept": "Verification not performed"}]}},
    ]

    result = derive_patterns(reports)

    assert result[0]["name"] == "Verification not performed"
    assert result[0]["count"] == 2
    assert result[0]["report_ids"] == ["A", "B"]
