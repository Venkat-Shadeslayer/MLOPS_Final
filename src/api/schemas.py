"""
Pydantic schemas for API request/response validation.

FastAPI auto-generates OpenAPI docs at /docs from these.
Pydantic v2 gives strict type enforcement at request boundary — invalid
payloads return 422 with structured error messages before touching
business logic.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PollutantReading(BaseModel):
    """Single city-day pollutant observation. All fields optional because
    real-world sensors report partial data."""
    city: str = Field(..., description="City name, e.g. 'Delhi'")
    date: str = Field(..., description="ISO date YYYY-MM-DD")
    pm25: Optional[float] = Field(None, alias="PM2.5")
    pm10: Optional[float] = Field(None, alias="PM10")
    no: Optional[float] = Field(None, alias="NO")
    no2: Optional[float] = Field(None, alias="NO2")
    nox: Optional[float] = Field(None, alias="NOx")
    nh3: Optional[float] = Field(None, alias="NH3")
    co: Optional[float] = Field(None, alias="CO")
    so2: Optional[float] = Field(None, alias="SO2")
    o3: Optional[float] = Field(None, alias="O3")
    benzene: Optional[float] = Field(None, alias="Benzene")
    toluene: Optional[float] = Field(None, alias="Toluene")
    xylene: Optional[float] = Field(None, alias="Xylene")

    model_config = {"populate_by_name": True}


class PredictionRequest(BaseModel):
    """A prediction request. Includes recent history for lag features."""
    reading: PollutantReading
    # Optional historical readings for the same city, for lag/rolling features
    history: list[PollutantReading] = Field(
        default_factory=list,
        description="Recent past readings for this city (last 14 days ideally)",
    )


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_aqi: float
    model_version: str
    model_stage: str
    timestamp: datetime
    latency_ms: float


class GroundTruthSubmission(BaseModel):
    """Reported actual AQI for a past prediction — closes the feedback loop."""
    prediction_id: str
    actual_aqi: float


class HealthResponse(BaseModel):
    status: str
    service: str = "aqi-api"


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    detail: Optional[str] = None