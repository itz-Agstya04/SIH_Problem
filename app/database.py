from __future__ import annotations

import os
from pathlib import Path

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.sql import func

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'sif_reports.sqlite3').as_posix()}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    timestamp: Mapped[str] = mapped_column(String(40), default="")
    site: Mapped[str] = mapped_column(String(160), default="Unknown")
    location: Mapped[str] = mapped_column(String(200), default="Unknown")
    activity: Mapped[str] = mapped_column(String(160), default="Unknown")
    report_type: Mapped[str] = mapped_column(String(80), default="Observation")
    raw_description: Mapped[str] = mapped_column(Text)
    normalized_description: Mapped[str] = mapped_column(Text)
    factors: Mapped[dict] = mapped_column(JSON)
    rule_scores: Mapped[list] = mapped_column(JSON)
    evidence: Mapped[list] = mapped_column(JSON)
    sif_score: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(80))
    review_status: Mapped[str] = mapped_column(String(40), default="needs_review")
    human_label: Mapped[str | None] = mapped_column(String(40), nullable=True)
    review_note: Mapped[str] = mapped_column(Text, default="")
    is_demo: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewEvent(Base):
    __tablename__ = "review_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.report_id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(120))
    previous_status: Mapped[str] = mapped_column(String(40))
    previous_label: Mapped[str | None] = mapped_column(String(40), nullable=True)
    new_status: Mapped[str] = mapped_column(String(40))
    new_label: Mapped[str] = mapped_column(String(40))
    review_note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
