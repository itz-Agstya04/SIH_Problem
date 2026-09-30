from __future__ import annotations

import csv
import io
import uuid
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Report, ReviewEvent, SessionLocal, initialize_database
from app.engine import MODEL_VERSION, analyze_report, derive_patterns
from app.ml import (
    MINIMUM_PER_CLASS,
    MINIMUM_TOTAL_LABELS,
    ModelQualityError,
    build_bayesian_feature_row,
    build_feature_text,
    get_bayesian_model_status,
    get_model_status,
    predict_bayesian_score,
    predict_score,
    train_and_save,
    train_bayesian_model,
)

APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

DEMO_REPORTS = [
    {"report_id": "DEMO-001", "timestamp": "2026-01-12", "site": "Duliajan", "location": "Unit 4", "activity": "Maintenance", "report_type": "Near miss", "raw_description": "During maintenance, the technician entered the area while the electrical panel was still energized. LOTO was not verified."},
    {"report_id": "DEMO-002", "timestamp": "2026-01-19", "site": "Duliajan", "location": "Pump station", "activity": "Maintenance", "report_type": "Unsafe condition", "raw_description": "A wrong breaker was isolated before repair work and the isolation point was not verified."},
    {"report_id": "DEMO-003", "timestamp": "2026-02-03", "site": "Moran", "location": "Well pad 12", "activity": "Maintenance", "report_type": "Near miss", "raw_description": "The worker entered the area while equipment remained energized; verification was not performed after lockout."},
    {"report_id": "DEMO-004", "timestamp": "2026-02-17", "site": "Moran", "location": "Drilling floor", "activity": "Lifting operation", "report_type": "Unsafe act", "raw_description": "A worker stood beneath a suspended load during crane lifting. The barricade was missing."},
    {"report_id": "DEMO-005", "timestamp": "2026-03-06", "site": "Duliajan", "location": "Workshop", "activity": "Hot work", "report_type": "Unsafe condition", "raw_description": "Welding started without a work permit and sparks reached nearby flammable material."},
    {"report_id": "DEMO-006", "timestamp": "2026-03-21", "site": "Naharkatiya", "location": "Access road", "activity": "Driving", "report_type": "Near miss", "raw_description": "A reversing vehicle came close to a worker in the line of fire. The spotter was not present."},
    {"report_id": "DEMO-007", "timestamp": "2026-04-02", "site": "Moran", "location": "Separator area", "activity": "Inspection", "report_type": "Unsafe condition", "raw_description": "Gas testing detected a hazardous atmosphere before a worker entered a confined space."},
    {"report_id": "DEMO-008", "timestamp": "2026-04-16", "site": "Naharkatiya", "location": "Pipe rack", "activity": "Work at height", "report_type": "Near miss", "raw_description": "A worker on a scaffold was exposed to a fall hazard after the guardrail failed."},
    {"report_id": "DEMO-009", "timestamp": "2026-05-11", "site": "Duliajan", "location": "Electrical substation", "activity": "Electrical work", "report_type": "Incident", "raw_description": "An electrical worker received a shock while working on a live panel; lockout was bypassed."},
    {"report_id": "DEMO-010", "timestamp": "2026-05-23", "site": "Moran", "location": "Pipeline corridor", "activity": "Excavation", "report_type": "Unsafe act", "raw_description": "Excavation began without a permit to work and exposed an underground line."},
    {"report_id": "DEMO-011", "timestamp": "2026-06-07", "site": "Naharkatiya", "location": "Compressor shed", "activity": "Maintenance", "report_type": "Near miss", "raw_description": "Pressure was released from a pressurized hydraulic line near a technician; the isolation barrier was inadequate."},
    {"report_id": "DEMO-012", "timestamp": "2026-06-26", "site": "Duliajan", "location": "Tank farm", "activity": "Hot work", "report_type": "Unsafe condition", "raw_description": "Grinding produced sparks near a flammable vapor source. The hot work permit was not checked."},
]


class ReviewPayload(BaseModel):
    review_status: Literal["reviewed", "needs_review"]
    human_label: Literal["sif_potential", "not_sif_potential", "uncertain"]
    review_note: str = Field(default="", max_length=2000)
    reviewer: str = Field(default="local-user", max_length=120)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    with SessionLocal() as session:
        if session.scalar(select(Report.report_id).limit(1)) is None:
            for item in DEMO_REPORTS:
                _store_report(session, item, is_demo=True)
            session.commit()
    yield


app = FastAPI(title="OIL SIF Precursor Detection", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def get_session():
    with SessionLocal() as session:
        yield session


def _store_report(session: Session, item: dict, is_demo: bool = False) -> Report:
    description = str(item.get("raw_description") or item.get("description") or item.get("narrative") or "").strip()
    if not description:
        raise ValueError("Report narrative is required")
    analysis = analyze_report(description)
    report = Report(
        report_id=str(item.get("report_id") or f"RPT-{uuid.uuid4().hex[:10].upper()}"),
        timestamp=str(item.get("timestamp") or item.get("date") or ""),
        site=str(item.get("site") or "Unknown"),
        location=str(item.get("location") or "Unknown"),
        activity=str(item.get("activity") or "Unknown"),
        report_type=str(item.get("report_type") or "Observation"),
        raw_description=description,
        normalized_description=analysis["normalized_description"],
        factors=analysis["factors"],
        rule_scores=analysis["rule_scores"],
        evidence=analysis["evidence"],
        sif_score=analysis["sif_score"],
        model_version=analysis["model_version"],
        review_status="needs_review" if analysis["sif_score"] >= 0.5 else "unreviewed",
        is_demo=is_demo,
    )
    session.add(report)
    return report


def _serialize(report: Report) -> dict:
    return {
        "report_id": report.report_id,
        "timestamp": report.timestamp,
        "site": report.site,
        "location": report.location,
        "activity": report.activity,
        "report_type": report.report_type,
        "raw_description": report.raw_description,
        "normalized_description": report.normalized_description,
        "factors": report.factors,
        "rule_scores": report.rule_scores,
        "evidence": report.evidence,
        "sif_score": report.sif_score,
        "score_notice": "Illustrative demo screening score. Neither trained nor calibrated, and not approved for operational safety decisions.",
        "model_version": report.model_version,
        "review_status": report.review_status,
        "human_label": report.human_label,
        "review_note": report.review_note,
        "is_demo": report.is_demo,
    }


@app.get("/")
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "mode": "illustrative prototype"}


@app.get("/api/reports")
def list_reports(
    site: str | None = None,
    activity: str | None = None,
    report_type: str | None = None,
    review_status: str | None = None,
    q: str | None = None,
    session: Session = Depends(get_session),
):
    statement = select(Report).order_by(Report.timestamp.desc(), Report.report_id)
    if site:
        statement = statement.where(Report.site == site)
    if activity:
        statement = statement.where(Report.activity == activity)
    if report_type:
        statement = statement.where(Report.report_type == report_type)
    if review_status:
        statement = statement.where(Report.review_status == review_status)
    reports = [_serialize(report) for report in session.scalars(statement).all()]
    if q:
        query = q.casefold()
        reports = [report for report in reports if query in report["raw_description"].casefold() or query in report["report_id"].casefold()]
    return {"items": reports, "total": len(reports)}


@app.get("/api/summary")
def summary(session: Session = Depends(get_session)):
    reports = [_serialize(report) for report in session.scalars(select(Report)).all()]
    count = len(reports)
    sites: dict[str, list[float]] = defaultdict(list)
    activities: dict[str, list[float]] = defaultdict(list)
    monthly: dict[str, list[float]] = defaultdict(list)
    rule_counts: Counter = Counter()
    for report in reports:
        sites[report["site"]].append(report["sif_score"])
        activities[report["activity"]].append(report["sif_score"])
        month = report["timestamp"][:7] if len(report["timestamp"]) >= 7 else "Unknown"
        monthly[month].append(report["sif_score"])
        for rule in report["rule_scores"]:
            if rule["evidence"]:
                rule_counts[rule["rule"]] += 1
    mean = lambda values: round(sum(values) / len(values), 3) if values else 0
    needs_review = sum(report["review_status"] == "needs_review" for report in reports)
    return {
        "total_reports": count,
        "mean_screening_score": mean([report["sif_score"] for report in reports]),
        "high_screening_count": sum(report["sif_score"] >= 0.5 for report in reports),
        "needs_review": needs_review,
        "site_density": [{"name": key, "count": len(values), "score": mean(values)} for key, values in sorted(sites.items(), key=lambda pair: mean(pair[1]), reverse=True)],
        "activity_density": [{"name": key, "count": len(values), "score": mean(values)} for key, values in sorted(activities.items(), key=lambda pair: mean(pair[1]), reverse=True)],
        "monthly": [{"month": key, "count": len(values), "score": mean(values)} for key, values in sorted(monthly.items())],
        "life_saving_rules": [{"name": key, "count": value} for key, value in rule_counts.most_common()],
        "score_notice": "Illustrative demo screening score. Neither trained nor calibrated, and not approved for operational safety decisions.",
    }


@app.get("/api/patterns")
def patterns(session: Session = Depends(get_session)):
    reports = [_serialize(report) for report in session.scalars(select(Report)).all()]
    return {"items": derive_patterns(reports), "method": "Exact extracted barrier-failure grouping; prototype only."}


@app.get("/api/reports/{report_id}")
def get_report(report_id: str, session: Session = Depends(get_session)):
    report = session.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    result = _serialize(report)
    history = session.scalars(
        select(ReviewEvent)
        .where(ReviewEvent.report_id == report_id)
        .order_by(ReviewEvent.created_at.desc())
    ).all()
    result["review_history"] = [
        {
            "reviewer": event.reviewer,
            "previous_status": event.previous_status,
            "previous_label": event.previous_label,
            "new_status": event.new_status,
            "new_label": event.new_label,
            "review_note": event.review_note,
            "created_at": event.created_at.isoformat() if event.created_at else "",
        }
        for event in history
    ]
    return result


@app.put("/api/reports/{report_id}/review")
def review_report(report_id: str, payload: ReviewPayload, session: Session = Depends(get_session)):
    report = session.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    session.add(ReviewEvent(
        report_id=report.report_id,
        reviewer=payload.reviewer.strip() or "local-user",
        previous_status=report.review_status,
        previous_label=report.human_label,
        new_status=payload.review_status,
        new_label=payload.human_label,
        review_note=payload.review_note.strip(),
    ))
    report.review_status = payload.review_status
    report.human_label = payload.human_label
    report.review_note = payload.review_note.strip()
    session.commit()
    session.refresh(report)
    return _serialize(report)


@app.post("/api/import")
async def import_csv(file: UploadFile = File(...), session: Session = Depends(get_session)):
    if not file.filename or not file.filename.casefold().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 10 MB limit")
    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        fields = {field.strip().casefold() for field in reader.fieldnames if field}
        if not fields.intersection({"raw_description", "description", "narrative"}):
            raise ValueError("CSV needs a raw_description, description, or narrative column")
        created = 0
        skipped = 0
        for row in reader:
            item = {str(key).strip().casefold(): value for key, value in row.items() if key}
            description = item.get("raw_description") or item.get("description") or item.get("narrative") or ""
            if not description.strip():
                skipped += 1
                continue
            report_id = (item.get("report_id") or f"CSV-{uuid.uuid4().hex[:10].upper()}").strip()
            if session.get(Report, report_id):
                skipped += 1
                continue
            _store_report(session, {**item, "report_id": report_id})
            created += 1
        session.commit()
        return {"created": created, "skipped": skipped}
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/filters")
def filters(session: Session = Depends(get_session)):
    reports = session.scalars(select(Report)).all()
    return {
        "sites": sorted({report.site for report in reports}),
        "activities": sorted({report.activity for report in reports}),
        "report_types": sorted({report.report_type for report in reports}),
        "review_statuses": ["needs_review", "unreviewed", "reviewed"],
    }


def _eligible_ml_rows(session: Session) -> list[Report]:
    return list(session.scalars(
        select(Report).where(
            Report.is_demo.is_(False),
            Report.review_status == "reviewed",
            Report.human_label.in_(["sif_potential", "not_sif_potential"]),
        )
    ).all())


@app.get("/api/ml/status")
def ml_status(session: Session = Depends(get_session)):
    eligible = _eligible_ml_rows(session)
    class_counts = {
        label: sum(report.human_label == label for report in eligible)
        for label in ("sif_potential", "not_sif_potential")
    }
    return get_model_status(len(eligible), class_counts)


@app.post("/api/ml/train")
def train_sif_model(session: Session = Depends(get_session)):
    eligible = _eligible_ml_rows(session)
    labels = [1 if report.human_label == "sif_potential" else 0 for report in eligible]
    class_counts = {
        label: labels.count(value)
        for label, value in (("sif_potential", 1), ("not_sif_potential", 0))
    }
    if len(labels) < MINIMUM_TOTAL_LABELS or any(count < MINIMUM_PER_CLASS for count in class_counts.values()):
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Not enough reviewed real reports to train the model.",
                "eligible_labeled_reports": len(labels),
                "required_total": MINIMUM_TOTAL_LABELS,
                "class_counts": class_counts,
                "required_per_class": MINIMUM_PER_CLASS,
            },
        )
    texts = [
        build_feature_text(report.raw_description, report.activity, report.report_type, report.factors)
        for report in eligible
    ]
    try:
        return train_and_save(texts, labels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/ml/predict/{report_id}")
def predict_report(report_id: str, session: Session = Depends(get_session)):
    report = session.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    text = build_feature_text(report.raw_description, report.activity, report.report_type, report.factors)
    try:
        result = predict_score(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelQualityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"report_id": report.report_id, **result, "evidence": report.evidence}


@app.get("/api/bayesian/status")
@app.get("/api/bayes/status")
def bayesian_status():
    return get_bayesian_model_status()


@app.post("/api/bayesian/train")
@app.post("/api/bayes/train")
def train_bayesian(session: Session = Depends(get_session)):
    eligible = _eligible_ml_rows(session)
    rows = []
    for report in eligible:
        row = build_bayesian_feature_row(report.raw_description, report.activity, report.report_type, report.factors)
        row["sif_label"] = "sif_potential" if report.human_label == "sif_potential" else "not_sif_potential"
        rows.append(row)
    if len(rows) < 2:
        raise HTTPException(status_code=409, detail="At least two labeled reports are required to train the Bayesian layer.")
    try:
        return train_bayesian_model(rows)
    except (ImportError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/bayesian/predict/{report_id}")
@app.get("/api/bayes/predict/{report_id}")
def bayesian_predict_report(report_id: str, session: Session = Depends(get_session)):
    report = session.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        result = predict_bayesian_score({
            "raw_description": report.raw_description,
            "activity": report.activity,
            "report_type": report.report_type,
            "factors": report.factors,
        })
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"report_id": report.report_id, **result, "evidence": report.evidence}
