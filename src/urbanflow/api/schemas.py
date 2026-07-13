from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded", "unavailable"]
ComponentStatus = Literal["available", "unconfigured", "unavailable"]


class ComponentHealth(BaseModel):
    status: ComponentStatus


class HealthResult(BaseModel):
    status: HealthStatus
    service: str
    version: str
    generated_at: datetime
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    model_version: str | None = None
    data_cutoff_at: datetime | None = None


class SensorResponse(BaseModel):
    location_id: int
    sensor_name: str
    sensor_description: str
    status: str
    latitude: float
    longitude: float


class SensorListMeta(BaseModel):
    count: int
    active_only: bool


class SensorListResponse(BaseModel):
    data: list[SensorResponse]
    meta: SensorListMeta


class HistoryPoint(BaseModel):
    observed_at: datetime
    pedestrian_count: int


class HistoryResponse(BaseModel):
    location_id: int
    start: datetime
    end: datetime
    data: list[HistoryPoint]


class ForecastPredictionResponse(BaseModel):
    forecast_horizon: int
    target_at: datetime
    predicted_count: float


class ForecastResponse(BaseModel):
    location_id: int
    model_name: str
    model_version: str | None = None
    generated_at: datetime
    forecast_origin_at: datetime
    data_cutoff_at: datetime
    horizon_hours: int
    predictions: list[ForecastPredictionResponse]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[object] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
