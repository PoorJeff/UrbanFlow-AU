from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from urbanflow.api.schemas import (
    ForecastResponse,
    HealthResult,
    HistoryResponse,
    ModelMetricsResponse,
    SensorListResponse,
)
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.errors import DashboardApiError

BASE_URL = "https://dashboard-api.example.test"
Payload = dict[str, Any]


def make_api_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[DashboardApiClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return DashboardApiClient(BASE_URL, http_client=http_client), http_client


def json_handler(
    payload: object,
    *,
    status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    return handler


def assert_invalid_response(
    payload: Payload,
    invoke: Callable[[DashboardApiClient], object],
) -> None:
    api_client, http_client = make_api_client(json_handler(payload))
    try:
        with pytest.raises(DashboardApiError) as exc_info:
            invoke(api_client)
    finally:
        http_client.close()

    assert exc_info.value.status_code == 200
    assert exc_info.value.code == "invalid_api_response"


def test_get_health_requests_health_and_parses_source_model(
    health_payload: Payload,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        assert request.url.query == b""
        return httpx.Response(200, json=health_payload)

    api_client, http_client = make_api_client(handler)
    try:
        result = api_client.get_health()
    finally:
        http_client.close()

    assert isinstance(result, HealthResult)
    assert result.status == "ok"


def test_list_sensors_requests_active_catalog_and_parses_source_model(
    sensor_list_payload: Payload,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/sensors"
        assert dict(request.url.params) == {"active_only": "true"}
        return httpx.Response(200, json=sensor_list_payload)

    api_client, http_client = make_api_client(handler)
    try:
        result = api_client.list_sensors()
    finally:
        http_client.close()

    assert isinstance(result, SensorListResponse)
    assert result.data[0].location_id == 101


def test_get_history_requests_offset_aware_interval_and_parses_source_model(
    history_payload: Payload,
    history_start: datetime,
    history_end: datetime,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/sensors/101/history"
        assert dict(request.url.params) == {
            "start": "2026-07-12T08:00:00+10:00",
            "end": "2026-07-12T10:00:00+10:00",
        }
        return httpx.Response(200, json=history_payload)

    api_client, http_client = make_api_client(handler)
    try:
        result = api_client.get_history(101, start=history_start, end=history_end)
    finally:
        http_client.close()

    assert isinstance(result, HistoryResponse)
    assert result.location_id == 101


def test_get_forecast_requests_horizon_and_parses_source_model(
    forecast_payload: Payload,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/sensors/101/forecast"
        assert dict(request.url.params) == {"horizon": "2"}
        return httpx.Response(200, json=forecast_payload)

    api_client, http_client = make_api_client(handler)
    try:
        result = api_client.get_forecast(101, horizon=2)
    finally:
        http_client.close()

    assert isinstance(result, ForecastResponse)
    assert [prediction.forecast_horizon for prediction in result.predictions] == [1, 2]


def test_get_model_metrics_requests_metrics_and_parses_source_model(
    model_metrics_payload: Payload,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/model/metrics"
        assert request.url.query == b""
        return httpx.Response(200, json=model_metrics_payload)

    api_client, http_client = make_api_client(handler)
    try:
        result = api_client.get_model_metrics()
    finally:
        http_client.close()

    assert isinstance(result, ModelMetricsResponse)
    assert result.model_name == "lightgbm"


def test_health_parses_valid_unavailable_503_response(health_payload: Payload) -> None:
    payload = deepcopy(health_payload)
    payload["status"] = "unavailable"
    api_client, http_client = make_api_client(json_handler(payload, status_code=503))
    try:
        result = api_client.get_health()
    finally:
        http_client.close()

    assert isinstance(result, HealthResult)
    assert result.status == "unavailable"


def test_health_503_error_envelope_becomes_typed_dashboard_error() -> None:
    payload = {
        "error": {
            "code": "health_check_failed",
            "message": "Health status could not be determined.",
            "details": [],
        }
    }
    api_client, http_client = make_api_client(json_handler(payload, status_code=503))
    try:
        with pytest.raises(DashboardApiError) as exc_info:
            api_client.get_health()
    finally:
        http_client.close()

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "health_check_failed"


def test_error_envelope_becomes_typed_dashboard_error() -> None:
    payload = {
        "error": {
            "code": "sensor_not_found",
            "message": "Sensor 999 was not found.",
            "details": [{"location_id": 999}],
        }
    }
    api_client, http_client = make_api_client(json_handler(payload, status_code=404))
    try:
        with pytest.raises(DashboardApiError) as exc_info:
            api_client.get_forecast(999, horizon=1)
    finally:
        http_client.close()

    error = exc_info.value
    assert error.status_code == 404
    assert error.code == "sensor_not_found"
    assert error.message == "Sensor 999 was not found."
    assert error.details == ({"location_id": 999},)


@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda request: httpx.ReadTimeout("request timed out", request=request),
        lambda request: httpx.ConnectError("connection refused", request=request),
    ],
    ids=["timeout", "connection-failure"],
)
def test_transport_failure_becomes_api_unreachable_without_retry(
    exception_factory: Callable[[httpx.Request], httpx.RequestError],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise exception_factory(request)

    api_client, http_client = make_api_client(handler)
    try:
        with pytest.raises(DashboardApiError) as exc_info:
            api_client.list_sensors()
    finally:
        http_client.close()

    assert len(requests) == 1
    assert exc_info.value.status_code is None
    assert exc_info.value.code == "api_unreachable"
    assert exc_info.value.details == ()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json", headers={"content-type": "text/plain"}),
        httpx.Response(200, json={"data": []}),
    ],
    ids=["non-json", "pydantic-invalid"],
)
def test_malformed_success_becomes_invalid_api_response(response: httpx.Response) -> None:
    api_client, http_client = make_api_client(lambda request: response)
    try:
        with pytest.raises(DashboardApiError) as exc_info:
            api_client.list_sensors()
    finally:
        http_client.close()

    assert exc_info.value.status_code == 200
    assert exc_info.value.code == "invalid_api_response"


def test_unrecognised_error_response_becomes_api_request_failed() -> None:
    api_client, http_client = make_api_client(
        json_handler({"detail": "gateway unavailable"}, status_code=502)
    )
    try:
        with pytest.raises(DashboardApiError) as exc_info:
            api_client.get_model_metrics()
    finally:
        http_client.close()

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == "api_request_failed"
    assert exc_info.value.details == ()


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (datetime(2026, 7, 12, 8), datetime(2026, 7, 12, 10, tzinfo=UTC)),
        (datetime(2026, 7, 12, 8, tzinfo=UTC), datetime(2026, 7, 12, 10)),
        (
            datetime(2026, 7, 12, 10, tzinfo=UTC),
            datetime(2026, 7, 12, 10, tzinfo=UTC),
        ),
        (
            datetime(2026, 7, 12, 11, tzinfo=UTC),
            datetime(2026, 7, 12, 10, tzinfo=UTC),
        ),
    ],
    ids=["naive-start", "naive-end", "equal", "reversed"],
)
def test_get_history_rejects_invalid_interval_before_http(
    start: datetime,
    end: datetime,
) -> None:
    requests: list[httpx.Request] = []
    api_client, http_client = make_api_client(
        lambda request: requests.append(request) or httpx.Response(500)
    )
    try:
        with pytest.raises(ValueError):
            api_client.get_history(101, start=start, end=end)
    finally:
        http_client.close()

    assert requests == []


@pytest.mark.parametrize("horizon", [0, 25])
def test_get_forecast_rejects_invalid_horizon_before_http(horizon: int) -> None:
    requests: list[httpx.Request] = []
    api_client, http_client = make_api_client(
        lambda request: requests.append(request) or httpx.Response(500)
    )
    try:
        with pytest.raises(ValueError):
            api_client.get_forecast(101, horizon=horizon)
    finally:
        http_client.close()

    assert requests == []


def test_construction_has_no_io_default_timeout_is_five_seconds_and_close_preserves_injected_client(
    sensor_list_payload: Payload,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.extensions["timeout"] == {
            "connect": 5.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }
        return httpx.Response(200, json=sensor_list_payload)

    api_client, http_client = make_api_client(handler)

    assert requests == []
    api_client.list_sensors()
    assert len(requests) == 1

    api_client.close()
    assert not http_client.is_closed
    http_client.close()


@pytest.mark.parametrize(
    "field",
    ["generated_at", "data_cutoff_at"],
)
def test_health_rejects_naive_page_timestamp(
    health_payload: Payload,
    field: str,
) -> None:
    payload = deepcopy(health_payload)
    payload[field] = "2026-07-12T10:30:00"

    assert_invalid_response(payload, lambda client: client.get_health())


@pytest.mark.parametrize(
    "timestamp_path",
    [
        ("start",),
        ("end",),
        ("data", 0, "observed_at"),
    ],
    ids=["start", "end", "point"],
)
def test_history_rejects_naive_page_timestamp(
    history_payload: Payload,
    history_start: datetime,
    history_end: datetime,
    timestamp_path: tuple[str | int, ...],
) -> None:
    payload = deepcopy(history_payload)
    target: Any = payload
    for part in timestamp_path[:-1]:
        target = target[part]
    target[timestamp_path[-1]] = "2026-07-12T08:00:00"

    assert_invalid_response(
        payload,
        lambda client: client.get_history(101, start=history_start, end=history_end),
    )


def test_history_rejects_location_mismatch(
    history_payload: Payload,
    history_start: datetime,
    history_end: datetime,
) -> None:
    payload = deepcopy(history_payload)
    payload["location_id"] = 999

    assert_invalid_response(
        payload,
        lambda client: client.get_history(101, start=history_start, end=history_end),
    )


@pytest.mark.parametrize(
    "timestamp_path",
    [
        ("generated_at",),
        ("forecast_origin_at",),
        ("data_cutoff_at",),
        ("predictions", 0, "target_at"),
    ],
    ids=["generated", "origin", "cutoff", "prediction"],
)
def test_forecast_rejects_naive_page_timestamp(
    forecast_payload: Payload,
    timestamp_path: tuple[str | int, ...],
) -> None:
    payload = deepcopy(forecast_payload)
    target: Any = payload
    for part in timestamp_path[:-1]:
        target = target[part]
    target[timestamp_path[-1]] = "2026-07-12T10:30:00"

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


def test_forecast_rejects_location_mismatch(forecast_payload: Payload) -> None:
    payload = deepcopy(forecast_payload)
    payload["location_id"] = 999

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


def test_forecast_rejects_horizon_hours_mismatch(forecast_payload: Payload) -> None:
    payload = deepcopy(forecast_payload)
    payload["horizon_hours"] = 1

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


def test_forecast_rejects_out_of_order_predictions_without_sorting(
    forecast_payload: Payload,
) -> None:
    payload = deepcopy(forecast_payload)
    payload["predictions"] = list(reversed(payload["predictions"]))

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


@pytest.mark.parametrize("second_horizon", [1, 3], ids=["duplicate", "unexpected"])
def test_forecast_requires_horizons_exactly_once(
    forecast_payload: Payload,
    second_horizon: int,
) -> None:
    payload = deepcopy(forecast_payload)
    payload["predictions"][1]["forecast_horizon"] = second_horizon

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


@pytest.mark.parametrize("prediction_count", [1, 3], ids=["missing", "extra"])
def test_forecast_requires_exact_prediction_count_without_clipping(
    forecast_payload: Payload,
    prediction_count: int,
) -> None:
    payload = deepcopy(forecast_payload)
    if prediction_count == 1:
        payload["predictions"] = payload["predictions"][:1]
    else:
        payload["predictions"].append(
            {
                "forecast_horizon": 3,
                "target_at": "2026-07-12T13:00:00+10:00",
                "predicted_count": 17.0,
            }
        )

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


@pytest.mark.parametrize(
    "predicted_count",
    [float("nan"), float("inf"), float("-inf"), -0.1],
    ids=["nan", "positive-infinity", "negative-infinity", "negative"],
)
def test_forecast_rejects_non_finite_or_negative_prediction_without_clipping(
    forecast_payload: Payload,
    predicted_count: float,
) -> None:
    payload = deepcopy(forecast_payload)
    payload["predictions"][0]["predicted_count"] = predicted_count

    assert_invalid_response(payload, lambda client: client.get_forecast(101, horizon=2))


@pytest.mark.parametrize("field", ["start", "end"])
def test_model_metrics_rejects_naive_window_timestamp(
    model_metrics_payload: Payload,
    field: str,
) -> None:
    payload = deepcopy(model_metrics_payload)
    payload["final_test_window"][field] = "2025-02-01T00:00:00"

    assert_invalid_response(payload, lambda client: client.get_model_metrics())


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2025-03-01T00:00:00+11:00", "2025-03-01T00:00:00+11:00"),
        ("2025-03-02T00:00:00+11:00", "2025-03-01T00:00:00+11:00"),
    ],
    ids=["equal", "reversed"],
)
def test_model_metrics_requires_ordered_final_test_window(
    model_metrics_payload: Payload,
    start: str,
    end: str,
) -> None:
    payload = deepcopy(model_metrics_payload)
    payload["final_test_window"]["start"] = start
    payload["final_test_window"]["end"] = end

    assert_invalid_response(payload, lambda client: client.get_model_metrics())


@pytest.mark.parametrize(
    "field",
    [
        "mae",
        "rmse",
        "wape",
        "seasonal_naive_wape",
        "relative_wape_improvement",
    ],
)
def test_model_metrics_rejects_non_finite_displayed_metric(
    model_metrics_payload: Payload,
    field: str,
) -> None:
    payload = deepcopy(model_metrics_payload)
    payload["metrics"][field] = float("nan")

    assert_invalid_response(payload, lambda client: client.get_model_metrics())


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_model_metrics_rejects_infinite_displayed_metric(
    model_metrics_payload: Payload,
    value: float,
) -> None:
    payload = deepcopy(model_metrics_payload)
    payload["metrics"]["mae"] = value

    assert_invalid_response(payload, lambda client: client.get_model_metrics())
