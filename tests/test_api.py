from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.database import ReviewEvent, SessionLocal
from app.main import app


def test_health_seed_review_history_and_csv_import():
    report_id = f"TEST-{uuid.uuid4().hex[:12]}"
    csv_report_id = f"CSV-{uuid.uuid4().hex[:12]}"
    client = TestClient(app)
    client.__enter__()
    try:
        assert client.get("/api/health").json()["status"] == "ok"
        model_status = client.get("/api/ml/status")
        assert model_status.status_code == 200
        assert model_status.json()["eligible_labeled_reports"] == 0
        assert model_status.json()["ready_to_train"] is False
        assert client.post("/api/ml/train").status_code == 409
        prediction = client.get("/api/ml/predict/DEMO-001")
        assert prediction.status_code == (200 if model_status.json()["prediction_ready"] else 409)

        seeded = client.get("/api/reports/DEMO-001")
        assert seeded.status_code == 200
        assert seeded.json()["is_demo"] is True
        assert seeded.json()["evidence"]

        created = client.post(
            "/api/import",
            files={
                "file": (
                    "reports.csv",
                    "report_id,timestamp,site,activity,report_type,raw_description\n"
                    f'{report_id},2026-06-30,Test Site,Maintenance,Near miss,"LOTO was not verified near energized equipment."\n'
                    f'{csv_report_id},2026-06-30,Test Site,Maintenance,Near miss,"Worker entered while electrical panel was energized."\n',
                    "text/csv",
                )
            },
        )
        assert created.status_code == 200
        assert created.json() == {"created": 2, "skipped": 0}

        reviewed = client.put(
            f"/api/reports/{report_id}/review",
            json={
                "review_status": "reviewed",
                "human_label": "sif_potential",
                "review_note": "Confirmed by HSE review.",
                "reviewer": "test-reviewer",
            },
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["review_status"] == "reviewed"

        detail = client.get(f"/api/reports/{report_id}").json()
        assert detail["review_history"][0]["reviewer"] == "test-reviewer"
        assert detail["review_history"][0]["new_label"] == "sif_potential"
        assert detail["review_history"][0]["review_note"] == "Confirmed by HSE review."
        assert client.get("/api/summary").json()["total_reports"] >= 14
    finally:
        with SessionLocal() as session:
            session.query(ReviewEvent).filter(ReviewEvent.report_id == report_id).delete()
            session.query(ReviewEvent).filter(ReviewEvent.report_id == csv_report_id).delete()
            from app.database import Report

            session.query(Report).filter(Report.report_id.in_([report_id, csv_report_id])).delete(synchronize_session=False)
            session.commit()
        client.close()
