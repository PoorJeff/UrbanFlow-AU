from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from urbanflow import __version__
from urbanflow.api.errors import (
    UrbanFlowApiError,
    data_store_unavailable_error,
    forecast_unavailable_error,
)
from urbanflow.api.schemas import (
    ComponentHealth,
    FinalTestWindowResponse,
    HealthComponents,
    HealthResult,
    ModelMetricsResponse,
    ModelMetricValues,
)

API_SERVICE_NAME = "urbanflow-au-api"
MAX_HISTORY_RANGE = timedelta(days=31)
SUPPORTED_MODEL_COMPARISON_KEYS = {
    "ridge_wape": "ridge",
    "lightgbm_wape": "lightgbm",
}


class HealthService(Protocol):
    def __call__(self) -> HealthResult: ...


@dataclass(frozen=True, slots=True)
class SensorRecord:
    location_id: int
    sensor_name: str
    sensor_description: str
    status: str
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    observed_at: datetime
    pedestrian_count: int


class SensorRepository(Protocol):
    def list_sensors(self, active_only: bool) -> list[SensorRecord]: ...

    def get_sensor(self, location_id: int) -> SensorRecord | None: ...


class HistoryRepository(Protocol):
    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]: ...


class RecentHistoryRepository(Protocol):
    def get_recent_history(
        self,
        location_id: int,
        *,
        limit: int,
    ) -> list[HistoryRecord]: ...


@dataclass(frozen=True, slots=True)
class ForecastPrediction:
    forecast_horizon: int
    target_at: datetime
    predicted_count: float


@dataclass(frozen=True, slots=True)
class ForecastBatch:
    model_name: str
    model_version: str | None
    generated_at: datetime
    forecast_origin_at: datetime
    data_cutoff_at: datetime
    predictions: tuple[ForecastPrediction, ...]


class ForecastModelProvider(Protocol):
    def predict(self, location_id: int, horizon: int) -> ForecastBatch: ...


class ModelMetadataProvider(Protocol):
    def get_metrics(self) -> ModelMetricsResponse: ...


class DataStoreUnavailableError(RuntimeError):
    """Raised by a configured API repository when its backing store cannot be read."""


class ForecastInputUnavailableError(RuntimeError):
    """Raised when serving inputs cannot satisfy the forecast contract."""


class MetricsUnavailableError(RuntimeError):
    """Raised when configured model evaluation metadata cannot be read or validated."""


@dataclass(frozen=True, slots=True)
class EvaluationSummaryMetadataProvider:
    path: Path | None

    @classmethod
    def from_environment(cls) -> EvaluationSummaryMetadataProvider:
        configured_path = os.getenv("URBANFLOW_API_METRICS_PATH")
        return cls(path=Path(configured_path) if configured_path else None)

    def get_metrics(self) -> ModelMetricsResponse:
        if self.path is None:
            raise MetricsUnavailableError("no evaluation summary path is configured")
        try:
            summary = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetricsUnavailableError("could not read evaluation summary") from exc
        try:
            return _model_metrics_from_summary(summary)
        except ValidationError as exc:
            raise MetricsUnavailableError(
                "evaluation summary has an invalid response shape"
            ) from exc


class EmptySensorRepository:
    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        return []

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        return None


class EmptyHistoryRepository:
    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]:
        return []


@dataclass(frozen=True, slots=True)
class HistoryService:
    sensor_repository: SensorRepository
    history_repository: HistoryRepository

    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]:
        _validate_history_range(start=start, end=end)
        _ensure_sensor_exists(self.sensor_repository, location_id)
        try:
            records = self.history_repository.get_history(location_id, start, end)
        except DataStoreUnavailableError as exc:
            raise data_store_unavailable_error() from exc
        return sorted(
            (record for record in records if start <= record.observed_at < end),
            key=lambda record: record.observed_at,
        )


@dataclass(frozen=True, slots=True)
class ForecastService:
    sensor_repository: SensorRepository
    model_provider: ForecastModelProvider | None

    def forecast(self, location_id: int, horizon: int) -> ForecastBatch:
        if self.model_provider is None:
            raise UrbanFlowApiError(
                status_code=503,
                code="model_unavailable",
                message="No forecast model is configured for serving.",
            )
        _ensure_sensor_exists(self.sensor_repository, location_id)
        try:
            batch = self.model_provider.predict(location_id, horizon)
        except DataStoreUnavailableError as exc:
            raise data_store_unavailable_error() from exc
        except ForecastInputUnavailableError as exc:
            raise forecast_unavailable_error() from exc
        _validate_forecast_horizons(batch=batch, horizon=horizon)
        return ForecastBatch(
            model_name=batch.model_name,
            model_version=batch.model_version,
            generated_at=batch.generated_at,
            forecast_origin_at=batch.forecast_origin_at,
            data_cutoff_at=batch.data_cutoff_at,
            predictions=tuple(
                ForecastPrediction(
                    forecast_horizon=prediction.forecast_horizon,
                    target_at=prediction.target_at,
                    predicted_count=max(float(prediction.predicted_count), 0.0),
                )
                for prediction in sorted(
                    batch.predictions,
                    key=lambda prediction: prediction.forecast_horizon,
                )
            ),
        )


def default_health() -> HealthResult:
    return HealthResult(
        status="degraded",
        service=API_SERVICE_NAME,
        version=__version__,
        generated_at=datetime.now(UTC),
        components=HealthComponents(
            api_process=ComponentHealth(status="available"),
            model_provider=ComponentHealth(status="unconfigured"),
            data_store=ComponentHealth(status="unconfigured"),
            data_freshness=ComponentHealth(status="unconfigured"),
        ),
    )


@dataclass(frozen=True, slots=True)
class ApiServices:
    health: HealthService = default_health
    sensor_repository: SensorRepository = field(default_factory=EmptySensorRepository)
    history_repository: HistoryRepository = field(default_factory=EmptyHistoryRepository)
    model_provider: ForecastModelProvider | None = None
    model_metadata_provider: ModelMetadataProvider = field(
        default_factory=EvaluationSummaryMetadataProvider.from_environment
    )

    @property
    def history_service(self) -> HistoryService:
        return HistoryService(
            sensor_repository=self.sensor_repository,
            history_repository=self.history_repository,
        )

    @property
    def forecast_service(self) -> ForecastService:
        return ForecastService(
            sensor_repository=self.sensor_repository,
            model_provider=self.model_provider,
        )


def _validate_history_range(*, start: datetime, end: datetime) -> None:
    if not _is_timezone_aware(start) or not _is_timezone_aware(end):
        raise UrbanFlowApiError(
            status_code=422,
            code="history_range_invalid",
            message="History start and end timestamps must include timezones.",
        )
    if start >= end:
        raise UrbanFlowApiError(
            status_code=422,
            code="history_range_invalid",
            message="History start must be earlier than end.",
        )
    if end - start > MAX_HISTORY_RANGE:
        raise UrbanFlowApiError(
            status_code=422,
            code="history_range_invalid",
            message="History range must not exceed 31 days.",
        )


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _ensure_sensor_exists(sensor_repository: SensorRepository, location_id: int) -> None:
    try:
        sensor = sensor_repository.get_sensor(location_id)
    except DataStoreUnavailableError as exc:
        raise data_store_unavailable_error() from exc
    if sensor is None:
        raise UrbanFlowApiError(
            status_code=404,
            code="sensor_not_found",
            message=f"Sensor {location_id} was not found.",
        )


def _validate_forecast_horizons(*, batch: ForecastBatch, horizon: int) -> None:
    expected_horizons = list(range(1, horizon + 1))
    actual_horizons = sorted(prediction.forecast_horizon for prediction in batch.predictions)
    if actual_horizons != expected_horizons:
        raise UrbanFlowApiError(
            status_code=503,
            code="model_unavailable",
            message="Forecast provider returned an incomplete horizon batch.",
        )
    if not all(
        _is_finite_forecast_value(prediction.predicted_count) for prediction in batch.predictions
    ):
        raise UrbanFlowApiError(
            status_code=503,
            code="model_unavailable",
            message="Forecast provider returned a non-finite predicted count.",
        )


def _is_finite_forecast_value(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _model_metrics_from_summary(summary: object) -> ModelMetricsResponse:
    root = _as_mapping(summary)
    final_test = _required_mapping(root, "final_test")
    overall = _required_mapping(final_test, "overall")
    seasonal_naive_overall = _required_mapping(final_test, "seasonal_naive_overall")
    comparison = _required_mapping(final_test, "model_comparison")
    model_name = _infer_model_name(comparison)
    _required_number(comparison, f"{model_name}_wape")
    return ModelMetricsResponse(
        model_name=model_name,
        model_version=_optional_text(root, "model_version"),
        evaluation_source="evaluation_summary",
        final_test_window=FinalTestWindowResponse(
            name=_required_text(final_test, "name"),
            start=_required_text(final_test, "start"),
            end=_required_text(final_test, "end"),
        ),
        metrics=ModelMetricValues(
            mae=_required_number(overall, "mae"),
            rmse=_required_number(overall, "rmse"),
            wape=_required_number(overall, "wape"),
            seasonal_naive_wape=_required_number(seasonal_naive_overall, "wape"),
            relative_wape_improvement=_required_number(
                comparison,
                "relative_wape_improvement",
            ),
        ),
        mlflow_run_id=_optional_text(root, "mlflow_run_id"),
        mlflow_tracking_uri=_optional_text(root, "mlflow_tracking_uri"),
        report_path=_optional_text(root, "report_path"),
    )


def _as_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricsUnavailableError("evaluation summary must be a JSON object")
    return value


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise MetricsUnavailableError(f"missing or invalid evaluation summary field: {key}")
    return value


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise MetricsUnavailableError(f"missing or invalid evaluation summary field: {key}")
    return value


def _optional_text(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MetricsUnavailableError(f"invalid optional evaluation summary field: {key}")
    return value


def _infer_model_name(comparison: Mapping[str, Any]) -> str:
    model_names = [
        model_name
        for comparison_key, model_name in SUPPORTED_MODEL_COMPARISON_KEYS.items()
        if comparison_key in comparison
    ]
    if len(model_names) != 1:
        raise MetricsUnavailableError(
            "evaluation summary must identify exactly one supported model"
        )
    return model_names[0]


def _required_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise MetricsUnavailableError(f"missing or invalid evaluation summary metric: {key}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetricsUnavailableError(
            f"missing or invalid evaluation summary metric: {key}"
        ) from exc
    if not math.isfinite(number):
        raise MetricsUnavailableError(f"missing or invalid evaluation summary metric: {key}")
    return number
