from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from urbanflow.api.schemas import ForecastResponse, HistoryResponse
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.errors import DashboardApiError
from urbanflow.dashboard.time_utils import MELBOURNE_TIME_ZONE, melbourne_now

HISTORY_DURATION = timedelta(hours=24)
FALLBACK_FORECAST_ERROR_CODES = frozenset({"model_unavailable", "forecast_unavailable"})


@dataclass(frozen=True, slots=True)
class TodaySnapshot:
    forecast: ForecastResponse | None
    forecast_error: DashboardApiError | None
    history: HistoryResponse | None
    history_error: DashboardApiError | None


def load_today_snapshot(
    client: DashboardApiClient,
    *,
    location_id: int,
    now: Callable[[], datetime] = melbourne_now,
) -> TodaySnapshot:
    try:
        forecast = client.get_forecast(location_id, horizon=24)
    except DashboardApiError as forecast_error:
        if forecast_error.code not in FALLBACK_FORECAST_ERROR_CODES:
            return TodaySnapshot(None, forecast_error, None, None)

        fallback_end = _as_melbourne(now())
        history, history_error = _load_history(
            client,
            location_id=location_id,
            end=fallback_end,
        )
        return TodaySnapshot(None, forecast_error, history, history_error)

    history, history_error = _load_history(
        client,
        location_id=location_id,
        end=_history_end_from_cutoff(forecast.data_cutoff_at),
    )
    return TodaySnapshot(forecast, None, history, history_error)


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    forecast: ForecastResponse | None
    forecast_error: DashboardApiError | None
    history: HistoryResponse | None
    history_error: DashboardApiError | None


def load_forecast_snapshot(
    client: DashboardApiClient,
    *,
    location_id: int,
    horizon: int,
) -> ForecastSnapshot:
    try:
        forecast = client.get_forecast(location_id, horizon=horizon)
    except DashboardApiError as forecast_error:
        return ForecastSnapshot(None, forecast_error, None, None)

    history, history_error = _load_history(
        client,
        location_id=location_id,
        end=_history_end_from_cutoff(forecast.data_cutoff_at),
    )
    return ForecastSnapshot(forecast, None, history, history_error)


def _load_history(
    client: DashboardApiClient,
    *,
    location_id: int,
    end: datetime,
) -> tuple[HistoryResponse | None, DashboardApiError | None]:
    start = (end.astimezone(UTC) - HISTORY_DURATION).astimezone(end.tzinfo)
    try:
        return client.get_history(
            location_id,
            start=start,
            end=end,
        ), None
    except DashboardApiError as history_error:
        return None, history_error


def _history_end_from_cutoff(data_cutoff_at: datetime) -> datetime:
    if data_cutoff_at.tzinfo is None or data_cutoff_at.utcoffset() is None:
        raise ValueError("Forecast data cutoff must be offset-aware.")
    return data_cutoff_at + timedelta(microseconds=1)


def _as_melbourne(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Current time must be offset-aware.")
    return value.astimezone(MELBOURNE_TIME_ZONE)
