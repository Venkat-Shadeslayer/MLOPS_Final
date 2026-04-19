"""
Persistence layer for predictions + ground truth.

Schema:
  predictions(
    id uuid primary key,
    created_at timestamp,
    model_version text,
    model_family text,
    input_features jsonb,
    predicted_aqi float,
    actual_aqi float null,
    feedback_at timestamp null,
    latency_ms float
  )

This table drives:
  - Grafana dashboard (rolling RMSE from predicted vs actual)
  - Drift detection (Day 5)
  - Retraining trigger (Day 5)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, String, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config import db_config
from src.utils.logging import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String, nullable=False)
    model_family = Column(String, nullable=False)
    input_features = Column(JSONB, nullable=False)
    predicted_aqi = Column(Float, nullable=False)
    actual_aqi = Column(Float, nullable=True)
    feedback_at = Column(DateTime, nullable=True)
    latency_ms = Column(Float, nullable=False)


_engine = None
_SessionLocal = None


def init_db() -> None:
    """Create engine + tables. Idempotent."""
    global _engine, _SessionLocal
    if _engine is not None:
        return
    log.info("Initializing predictions DB at %s", db_config.host)
    _engine = create_engine(db_config.url, pool_pre_ping=True)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def insert_prediction(
    model_version: str,
    model_family: str,
    input_features: dict,
    predicted_aqi: float,
    latency_ms: float,
) -> str:
    with get_session() as s:
        row = Prediction(
            model_version=model_version,
            model_family=model_family,
            input_features=input_features,
            predicted_aqi=predicted_aqi,
            latency_ms=latency_ms,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return str(row.id)


def record_ground_truth(prediction_id: str, actual_aqi: float) -> bool:
    """Update a past prediction with its actual AQI. Returns True if found."""
    with get_session() as s:
        row = s.get(Prediction, uuid.UUID(prediction_id))
        if row is None:
            return False
        row.actual_aqi = actual_aqi
        row.feedback_at = datetime.utcnow()
        s.commit()
        return True


def rolling_rmse(window_hours: int = 24) -> Optional[float]:
    """Compute RMSE over predictions in the window that have ground truth."""
    sql = text(
        """
        SELECT SQRT(AVG(POWER(predicted_aqi - actual_aqi, 2)))
        FROM predictions
        WHERE actual_aqi IS NOT NULL
          AND feedback_at > NOW() - (:hours || ' hours')::interval
        """
    )
    with get_session() as s:
        result = s.execute(sql, {"hours": window_hours}).scalar()
        return float(result) if result is not None else None