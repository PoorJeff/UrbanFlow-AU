from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TypeVar

import httpx
from pydantic import BaseModel

from urbanflow.api.schemas import (
    ErrorResponse,
    ForecastResponse,
    HealthResult,
    HistoryResponse,
    ModelMetricsResponse,
    SensorListResponse,
)
from urbanflow.dashboard.errors import DashboardApiError

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
SemanticValidator = Callable[[ResponseModel], None]


class DashboardApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client if http_client is not None else httpx.Client()
        self._owns_http_client = http_client is None
        self._timeout_seconds = timeout_seconds

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def get_health(self) -> HealthResult:
        return self._get(
            "/health",
            response_model=HealthResult,
            accepted_error_status=503,
            semantic_validator=_validate_health,
        )

    def list_sensors(self, *, active_only: bool = True) -> SensorListResponse:
        return self._get(
            "/api/v1/sensors",
            response_model=SensorListResponse,
            params={"active_only": active_only},
        )

    def get_history(
        self,
        location_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> HistoryResponse:
        if not _is_offset_aware(start) or not _is_offset_aware(end):
            raise ValueError("History timestamps must be offset-aware.")
        if start >= end:
            raise ValueError("History start must be earlier than end.")

        return self._get(
            f"/api/v1/sensors/{location_id}/history",
            response_model=HistoryResponse,
            params={"start": start.isoformat(), "end": end.isoformat()},
            semantic_validator=lambda response: _validate_history(
                response,
                requested_location_id=location_id,
            ),
        )

    def get_forecast(self, location_id: int, *, horizon: int) -> ForecastResponse:
        if not 1 <= horizon <= 24:
            raise ValueError("Forecast horizon must be between 1 and 24 hours.")

        return self._get(
            f"/api/v1/sensors/{location_id}/forecast",
            response_model=ForecastResponse,
            params={"horizon": horizon},
            semantic_validator=lambda response: _validate_forecast(
                response,
                requested_location_id=location_id,
                requested_horizon=horizon,
            ),
        )

    def get_model_metrics(self) -> ModelMetricsResponse:
        return self._get(
            "/api/v1/model/metrics",
            response_model=ModelMetricsResponse,
            semantic_validator=_validate_model_metrics,
        )

    def _get(
        self,
        path: str,
        *,
        response_model: type[ResponseModel],
        params: Mapping[str, object] | None = None,
        accepted_error_status: int | None = None,
        semantic_validator: SemanticValidator[ResponseModel] | None = None,
    ) -> ResponseModel:
        try:
            response = self._http_client.get(
                f"{self._base_url}{path}",
                params=params,
                timeout=self._timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise DashboardApiError(
                status_code=None,
                code="api_unreachable",
                message="The dashboard API is unreachable.",
                details=(),
            ) from exc

        accepted_non_success = (
            not response.is_success and response.status_code == accepted_error_status
        )
        if not response.is_success and not accepted_non_success:
            self._raise_request_error(response)

        try:
            parsed_response = response_model.model_validate(response.json())
        except (TypeError, ValueError) as exc:
            if accepted_non_success:
                self._raise_request_error(response)
            raise DashboardApiError(
                status_code=response.status_code,
                code="invalid_api_response",
                message="The dashboard API returned an invalid response.",
                details=(),
            ) from exc

        try:
            if semantic_validator is not None:
                semantic_validator(parsed_response)
        except (TypeError, ValueError) as exc:
            raise DashboardApiError(
                status_code=response.status_code,
                code="invalid_api_response",
                message="The dashboard API returned an invalid response.",
                details=(),
            ) from exc

        return parsed_response

    @staticmethod
    def _raise_request_error(response: httpx.Response) -> None:
        try:
            error_response = ErrorResponse.model_validate(response.json())
        except (TypeError, ValueError):
            raise DashboardApiError(
                status_code=response.status_code,
                code="api_request_failed",
                message="The dashboard API request failed.",
                details=(),
            ) from None

        raise DashboardApiError(
            status_code=response.status_code,
            code=error_response.error.code,
            message=error_response.error.message,
            details=tuple(error_response.error.details),
        )


def _is_offset_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _require_offset_aware(*values: datetime | None) -> None:
    if any(value is not None and not _is_offset_aware(value) for value in values):
        raise ValueError("API response timestamps must be offset-aware.")


def _validate_health(response: HealthResult) -> None:
    _require_offset_aware(response.generated_at, response.data_cutoff_at)


def _validate_history(
    response: HistoryResponse,
    *,
    requested_location_id: int,
) -> None:
    if response.location_id != requested_location_id:
        raise ValueError("History response location does not match the request.")
    _require_offset_aware(
        response.start,
        response.end,
        *(point.observed_at for point in response.data),
    )


def _validate_forecast(
    response: ForecastResponse,
    *,
    requested_location_id: int,
    requested_horizon: int,
) -> None:
    if response.location_id != requested_location_id:
        raise ValueError("Forecast response location does not match the request.")
    if response.horizon_hours != requested_horizon:
        raise ValueError("Forecast response horizon does not match the request.")
    if len(response.predictions) != requested_horizon:
        raise ValueError("Forecast response prediction count does not match the request.")

    actual_horizons = [prediction.forecast_horizon for prediction in response.predictions]
    if actual_horizons != list(range(1, requested_horizon + 1)):
        raise ValueError("Forecast response horizons are incomplete or out of order.")
    if any(
        not math.isfinite(prediction.predicted_count) or prediction.predicted_count < 0
        for prediction in response.predictions
    ):
        raise ValueError("Forecast response contains an invalid prediction.")

    _require_offset_aware(
        response.generated_at,
        response.forecast_origin_at,
        response.data_cutoff_at,
        *(prediction.target_at for prediction in response.predictions),
    )


def _validate_model_metrics(response: ModelMetricsResponse) -> None:
    final_test_window = response.final_test_window
    _require_offset_aware(final_test_window.start, final_test_window.end)
    if final_test_window.start >= final_test_window.end:
        raise ValueError("Model metrics final-test window is invalid.")

    metrics = response.metrics
    if not all(
        math.isfinite(value)
        for value in (
            metrics.mae,
            metrics.rmse,
            metrics.wape,
            metrics.seasonal_naive_wape,
            metrics.relative_wape_improvement,
        )
    ):
        raise ValueError("Model metrics contain a non-finite value.")
