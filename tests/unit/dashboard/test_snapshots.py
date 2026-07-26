from datetime import UTC, datetime, timedelta

import pytest

from urbanflow.api.schemas import (
    ForecastPredictionResponse,
    ForecastResponse,
    HistoryResponse,
)
from urbanflow.dashboard.errors import DashboardApiError
from urbanflow.dashboard.snapshots import (
    load_forecast_snapshot,
    load_today_snapshot,
)
from urbanflow.dashboard.time_utils import MELBOURNE_TIME_ZONE

FORECAST_ERROR_CODES_WITH_HISTORY_FALLBACK = (
    "model_unavailable",
    "forecast_unavailable",
)


def _api_error(code: str) -> DashboardApiError:
    return DashboardApiError(
        status_code=503,
        code=code,
        message=f"{code} test error",
        details=(),
    )


def _forecast(*, horizon: int, cutoff: datetime) -> ForecastResponse:
    return ForecastResponse(
        location_id=101,
        model_name="lightgbm",
        model_version="test-v1",
        generated_at=cutoff + timedelta(minutes=1),
        forecast_origin_at=cutoff,
        data_cutoff_at=cutoff,
        horizon_hours=horizon,
        predictions=[
            ForecastPredictionResponse(
                forecast_horizon=index,
                target_at=cutoff + timedelta(hours=index),
                predicted_count=index + 0.25,
            )
            for index in range(1, horizon + 1)
        ],
    )


def _history(*, start: datetime, end: datetime) -> HistoryResponse:
    return HistoryResponse(location_id=101, start=start, end=end, data=[])


class RecordingClient:
    def __init__(
        self,
        *,
        forecast_result: ForecastResponse | DashboardApiError,
        history_result: HistoryResponse | DashboardApiError | None,
    ) -> None:
        self.forecast_result = forecast_result
        self.history_result = history_result
        self.calls: list[tuple[object, ...]] = []

    def get_forecast(self, location_id: int, *, horizon: int) -> ForecastResponse:
        self.calls.append(("forecast", location_id, horizon))
        if isinstance(self.forecast_result, DashboardApiError):
            raise self.forecast_result
        return self.forecast_result

    def get_history(
        self,
        location_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> HistoryResponse:
        self.calls.append(("history", location_id, start, end))
        if isinstance(self.history_result, DashboardApiError):
            raise self.history_result
        assert self.history_result is not None
        return self.history_result


def test_today_loads_forecast_before_cutoff_aligned_history() -> None:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    end = cutoff + timedelta(microseconds=1)
    history = _history(start=end - timedelta(hours=24), end=end)
    forecast = _forecast(horizon=24, cutoff=cutoff)
    client = RecordingClient(forecast_result=forecast, history_result=history)

    snapshot = load_today_snapshot(client, location_id=101)

    assert client.calls == [
        ("forecast", 101, 24),
        ("history", 101, end - timedelta(hours=24), end),
    ]
    assert snapshot.forecast is forecast
    assert snapshot.forecast_error is None
    assert snapshot.history is history
    assert snapshot.history_error is None


def test_today_rejects_naive_forecast_cutoff_before_requesting_history() -> None:
    forecast = _forecast(horizon=24, cutoff=datetime(2026, 7, 12, 10))
    client = RecordingClient(forecast_result=forecast, history_result=None)

    with pytest.raises(ValueError, match="Forecast data cutoff must be offset-aware"):
        load_today_snapshot(client, location_id=101)

    assert client.calls == [("forecast", 101, 24)]


@pytest.mark.parametrize("error_code", FORECAST_ERROR_CODES_WITH_HISTORY_FALLBACK)
def test_today_uses_melbourne_now_history_only_for_availability_errors(
    error_code: str,
) -> None:
    current = datetime(2026, 7, 12, 20, tzinfo=MELBOURNE_TIME_ZONE)
    history = _history(start=current - timedelta(hours=24), end=current)
    error = _api_error(error_code)
    client = RecordingClient(forecast_result=error, history_result=history)

    def now() -> datetime:
        return current

    snapshot = load_today_snapshot(client, location_id=101, now=now)

    assert client.calls == [
        ("forecast", 101, 24),
        ("history", 101, current - timedelta(hours=24), current),
    ]
    assert snapshot.forecast is None
    assert snapshot.forecast_error is error
    assert snapshot.history is history
    assert snapshot.history_error is None


def test_today_fallback_history_covers_24_elapsed_hours_across_daylight_saving() -> None:
    current = datetime(2026, 10, 4, 12, tzinfo=MELBOURNE_TIME_ZONE)
    expected_start = (current.astimezone(UTC) - timedelta(hours=24)).astimezone(MELBOURNE_TIME_ZONE)
    history = _history(start=expected_start, end=current)
    client = RecordingClient(
        forecast_result=_api_error("model_unavailable"),
        history_result=history,
    )

    load_today_snapshot(client, location_id=101, now=lambda: current)

    assert current.utcoffset() != expected_start.utcoffset()
    assert client.calls == [
        ("forecast", 101, 24),
        ("history", 101, expected_start, current),
    ]


def test_today_stops_after_any_other_forecast_error() -> None:
    error = _api_error("api_unreachable")
    client = RecordingClient(forecast_result=error, history_result=None)

    snapshot = load_today_snapshot(client, location_id=101)

    assert client.calls == [("forecast", 101, 24)]
    assert snapshot.forecast is None
    assert snapshot.forecast_error is error
    assert snapshot.history is None
    assert snapshot.history_error is None


def test_today_keeps_forecast_when_empty_history_succeeds() -> None:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    end = cutoff + timedelta(microseconds=1)
    forecast = _forecast(horizon=24, cutoff=cutoff)
    empty_history = _history(start=end - timedelta(hours=24), end=end)
    client = RecordingClient(
        forecast_result=forecast,
        history_result=empty_history,
    )

    snapshot = load_today_snapshot(client, location_id=101)

    assert snapshot.forecast is forecast
    assert snapshot.history is empty_history
    assert snapshot.history.data == []
    assert snapshot.history_error is None


def test_today_records_auxiliary_history_error_separately() -> None:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    forecast = _forecast(horizon=24, cutoff=cutoff)
    history_error = _api_error("api_request_failed")
    client = RecordingClient(
        forecast_result=forecast,
        history_result=history_error,
    )

    snapshot = load_today_snapshot(client, location_id=101)

    assert snapshot.forecast is forecast
    assert snapshot.forecast_error is None
    assert snapshot.history is None
    assert snapshot.history_error is history_error


def test_forecast_loads_cutoff_aligned_history_only_after_success() -> None:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    end = cutoff + timedelta(microseconds=1)
    forecast = _forecast(horizon=2, cutoff=cutoff)
    history = _history(start=end - timedelta(hours=24), end=end)
    client = RecordingClient(forecast_result=forecast, history_result=history)

    snapshot = load_forecast_snapshot(client, location_id=101, horizon=2)

    assert client.calls == [
        ("forecast", 101, 2),
        ("history", 101, end - timedelta(hours=24), end),
    ]
    assert snapshot.forecast is forecast
    assert snapshot.forecast_error is None
    assert snapshot.history is history
    assert snapshot.history_error is None


def test_forecast_rejects_naive_cutoff_before_requesting_history() -> None:
    forecast = _forecast(horizon=2, cutoff=datetime(2026, 7, 12, 10))
    client = RecordingClient(forecast_result=forecast, history_result=None)

    with pytest.raises(ValueError, match="Forecast data cutoff must be offset-aware"):
        load_forecast_snapshot(client, location_id=101, horizon=2)

    assert client.calls == [("forecast", 101, 2)]


def test_forecast_keeps_forecast_when_auxiliary_history_fails() -> None:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    forecast = _forecast(horizon=2, cutoff=cutoff)
    history_error = _api_error("api_unreachable")
    client = RecordingClient(
        forecast_result=forecast,
        history_result=history_error,
    )

    snapshot = load_forecast_snapshot(client, location_id=101, horizon=2)

    assert snapshot.forecast is forecast
    assert snapshot.forecast_error is None
    assert snapshot.history is None
    assert snapshot.history_error is history_error


def test_forecast_stops_before_history_when_forecast_fails() -> None:
    error = _api_error("forecast_unavailable")
    client = RecordingClient(forecast_result=error, history_result=None)

    snapshot = load_forecast_snapshot(client, location_id=101, horizon=2)

    assert client.calls == [("forecast", 101, 2)]
    assert snapshot.forecast is None
    assert snapshot.forecast_error is error
    assert snapshot.history is None
    assert snapshot.history_error is None
