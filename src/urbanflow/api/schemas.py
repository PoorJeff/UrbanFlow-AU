from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded", "unavailable"]
ComponentStatus = Literal["available", "unconfigured", "unavailable"]


class ComponentHealth(BaseModel):
    status: ComponentStatus


class HealthComponents(BaseModel):
    api_process: ComponentHealth
    model_provider: ComponentHealth
    data_store: ComponentHealth
    data_freshness: ComponentHealth


class HealthResult(BaseModel):
    status: HealthStatus
    service: str
    version: str
    generated_at: datetime
    components: HealthComponents
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


class FinalTestWindowResponse(BaseModel):
    name: str
    start: datetime
    end: datetime


class ModelMetricValues(BaseModel):
    mae: float
    rmse: float
    wape: float
    seasonal_naive_wape: float
    relative_wape_improvement: float


class ModelMetricsResponse(BaseModel):
    model_name: str
    model_version: str | None = None
    evaluation_source: str
    final_test_window: FinalTestWindowResponse
    metrics: ModelMetricValues
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None
    report_path: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[object] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
