from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from urbanflow.api.schemas import (
    ComponentHealth,
    FinalTestWindowResponse,
    ForecastPredictionResponse,
    ForecastResponse,
    HealthComponents,
    HealthResult,
    HistoryPoint,
    HistoryResponse,
    ModelMetricsResponse,
    ModelMetricValues,
    SensorListMeta,
    SensorListResponse,
    SensorResponse,
)
from urbanflow.dashboard.errors import DashboardApiError


def _error(code: str, message: str | None = None) -> DashboardApiError:
    return DashboardApiError(
        status_code=503,
        code=code,
        message=message or f"{code} test message",
        details=(),
    )


def _health(status: str = "ok") -> HealthResult:
    component_status = "available" if status == "ok" else "unconfigured"
    return HealthResult(
        status=status,
        service="urbanflow-au-api",
        version="0.1.0",
        generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=UTC),
        components=HealthComponents(
            api_process=ComponentHealth(status="available"),
            model_provider=ComponentHealth(status=component_status),
            data_store=ComponentHealth(status=component_status),
            data_freshness=ComponentHealth(status=component_status),
        ),
        model_version="model-v1" if status == "ok" else None,
        data_cutoff_at=datetime(2026, 7, 12, 10, tzinfo=UTC) if status == "ok" else None,
    )


def _sensors(*location_ids: int) -> SensorListResponse:
    return SensorListResponse(
        data=[
            SensorResponse(
                location_id=location_id,
                sensor_name=f"Sensor {location_id}",
                sensor_description=f"Description {location_id}",
                status="Active",
                latitude=-37.81,
                longitude=144.96,
            )
            for location_id in location_ids
        ],
        meta=SensorListMeta(count=len(location_ids), active_only=True),
    )


def _forecast(location_id: int = 101) -> ForecastResponse:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    return ForecastResponse(
        location_id=location_id,
        model_name="lightgbm",
        model_version="model-v1",
        generated_at=cutoff + timedelta(minutes=5),
        forecast_origin_at=cutoff,
        data_cutoff_at=cutoff,
        horizon_hours=2,
        predictions=[
            ForecastPredictionResponse(
                forecast_horizon=1,
                target_at=cutoff + timedelta(hours=1),
                predicted_count=12.5,
            ),
            ForecastPredictionResponse(
                forecast_horizon=2,
                target_at=cutoff + timedelta(hours=2),
                predicted_count=18.0,
            ),
        ],
    )


def _history(location_id: int = 101, *, empty: bool = False) -> HistoryResponse:
    end = datetime(2026, 7, 12, 10, tzinfo=UTC) + timedelta(microseconds=1)
    return HistoryResponse(
        location_id=location_id,
        start=end - timedelta(hours=24),
        end=end,
        data=[]
        if empty
        else [
            HistoryPoint(
                observed_at=datetime(2026, 7, 12, 8, tzinfo=UTC),
                pedestrian_count=24,
            ),
            HistoryPoint(
                observed_at=datetime(2026, 7, 12, 9, tzinfo=UTC),
                pedestrian_count=31,
            ),
        ],
    )


def _metrics() -> ModelMetricsResponse:
    return ModelMetricsResponse(
        model_name="lightgbm",
        model_version="model-v1",
        evaluation_source="evaluation_summary",
        final_test_window=FinalTestWindowResponse(
            name="final_test_2025-02",
            start=datetime(2025, 2, 1, tzinfo=UTC),
            end=datetime(2025, 3, 1, tzinfo=UTC),
        ),
        metrics=ModelMetricValues(
            mae=1.2,
            rmse=1.7,
            wape=0.07,
            seasonal_naive_wape=0.095,
            relative_wape_improvement=0.263,
        ),
    )


class RecordingClient:
    def __init__(
        self,
        *,
        health: HealthResult | DashboardApiError | None = None,
        sensors: SensorListResponse | DashboardApiError | None = None,
        forecast: ForecastResponse | DashboardApiError | None = None,
        history: HistoryResponse | DashboardApiError | None = None,
        metrics: ModelMetricsResponse | DashboardApiError | None = None,
    ) -> None:
        self.health = health if health is not None else _health()
        self.sensors = sensors if sensors is not None else _sensors(101, 202)
        self.forecast = forecast if forecast is not None else _forecast()
        self.history = history if history is not None else _history()
        self.metrics = metrics if metrics is not None else _metrics()
        self.calls: list[tuple[Any, ...]] = []

    @staticmethod
    def _return_or_raise(result: Any) -> Any:
        if isinstance(result, DashboardApiError):
            raise result
        return result

    def get_health(self) -> HealthResult:
        self.calls.append(("health",))
        return self._return_or_raise(self.health)

    def list_sensors(self, *, active_only: bool = True) -> SensorListResponse:
        self.calls.append(("sensors", active_only))
        return self._return_or_raise(self.sensors)

    def get_forecast(self, location_id: int, *, horizon: int) -> ForecastResponse:
        self.calls.append(("forecast", location_id, horizon))
        return self._return_or_raise(self.forecast)

    def get_history(
        self,
        location_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> HistoryResponse:
        self.calls.append(("history", location_id, start, end))
        return self._return_or_raise(self.history)

    def get_model_metrics(self) -> ModelMetricsResponse:
        self.calls.append(("metrics",))
        return self._return_or_raise(self.metrics)


def _today_harness(client):
    from urbanflow.dashboard.pages.today import render_today

    render_today(client)


def _run(client: RecordingClient) -> AppTest:
    return AppTest.from_function(_today_harness, args=(client,)).run()


def _visible_text(at: AppTest) -> str:
    values: list[str] = []
    for element_type in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "text",
        "info",
        "warning",
        "error",
        "success",
        "metric",
        "button",
        "selectbox",
    ):
        for element in at.get(element_type):
            for attribute in ("value", "label"):
                value = getattr(element, attribute, None)
                if value is not None:
                    values.append(str(value))
    return "\n".join(values)


def _click(at: AppTest, key: str) -> AppTest:
    return at.button(key=key).click().run()


def _plotly_spec(at: AppTest) -> dict[str, Any]:
    charts = at.get("plotly_chart")
    assert len(charts) == 1
    return json.loads(charts[0].proto.spec)


def test_first_visit_is_guided_and_requests_only_context() -> None:
    client = RecordingClient()

    at = _run(client)

    assert not at.exception
    assert client.calls == [("health",), ("sensors", True)]
    text = _visible_text(at)
    assert "Today" in text
    assert "Choose an active sensor" in text
    assert at.selectbox(key="today_location_selector").label == "Active sensor"
    assert at.button(key="load_today_location").label == "Load this location"
    assert not at.metric
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_degraded_health_is_an_availability_signal_and_still_lists_sensors() -> None:
    client = RecordingClient(health=_health("degraded"))

    at = _run(client)

    assert client.calls == [("health",), ("sensors", True)]
    text = _visible_text(at)
    assert "degraded" in text.lower()
    assert "configuration and availability" in text.lower()
    assert "does not indicate that data or a model is ready" in text.lower()


@pytest.mark.parametrize(
    ("health", "expected_text"),
    [
        (_health("unavailable"), "Service unavailable"),
        (
            _error("api_unreachable", "The API could not be reached."),
            "The API could not be reached.",
        ),
    ],
)
def test_unavailable_or_failed_health_stops_before_catalog(
    health: HealthResult | DashboardApiError,
    expected_text: str,
) -> None:
    client = RecordingClient(health=health)

    at = _run(client)

    assert client.calls == [("health",)]
    assert expected_text in _visible_text(at)
    assert not at.selectbox
    assert not at.get("plotly_chart")


def test_empty_active_catalog_has_a_visible_state_without_details() -> None:
    client = RecordingClient(sensors=_sensors())

    at = _run(client)

    assert client.calls == [("health",), ("sensors", True)]
    assert "No active sensors were returned." in _visible_text(at)
    assert not at.selectbox
    assert not at.get("plotly_chart")


def test_selection_alone_never_loads_details_and_hides_an_old_result() -> None:
    client = RecordingClient(forecast=_forecast(202), history=_history(202))
    at = _run(client)

    at.selectbox(key="today_location_selector").select(202)
    at = at.run()
    assert client.calls == [
        ("health",),
        ("sensors", True),
        ("health",),
        ("sensors", True),
    ]

    at = _click(at, "load_today_location")
    assert client.calls[-4] == ("health",)
    assert client.calls[-3] == ("sensors", True)
    assert client.calls[-2] == ("forecast", 202, 24)
    assert client.calls[-1][:2] == ("history", 202)
    assert at.session_state.filtered_state["selected_location_id"] == 202
    assert "Sensor 202" in _visible_text(at)

    at.selectbox(key="today_location_selector").select(101)
    at = at.run()
    assert [call[0] for call in client.calls[-2:]] == ["health", "sensors"]
    assert "Latest returned observation" not in _visible_text(at)
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_submit_renders_full_current_response_and_no_response_session_state() -> None:
    client = RecordingClient()
    at = _run(client)

    at = _click(at, "load_today_location")

    assert [call[0] for call in client.calls] == [
        "health",
        "sensors",
        "health",
        "sensors",
        "forecast",
        "history",
    ]
    text = _visible_text(at)
    assert "Sensor 101" in text
    assert "Description 101" in text
    assert "Latest returned observation" in text
    assert "31" in text
    assert "Largest returned prediction" in text
    assert "18" in text
    assert "lightgbm" in text
    assert "model-v1" in text
    assert "Generated at" in text
    assert "Data cutoff" in text
    assert len(at.dataframe) == 1
    assert list(at.dataframe[0].value.columns) == ["Observed at", "Pedestrian count"]

    spec = _plotly_spec(at)
    assert [trace["name"] for trace in spec["data"]] == ["Observed", "Forecast"]
    assert spec["data"][0]["line"]["dash"] == "solid"
    assert spec["data"][1]["line"]["dash"] == "dash"

    state = at.session_state.filtered_state
    assert state["selected_location_id"] == 101
    assert set(state) == {
        "selected_location_id",
        "load_today_location",
        "today_location_selector",
        "view_historical_model_evaluation",
    }


def test_forecast_with_empty_history_is_explicitly_forecast_only() -> None:
    client = RecordingClient(history=_history(empty=True))

    at = _click(_run(client), "load_today_location")

    text = _visible_text(at)
    assert "No observations were returned for the matching interval." in text
    assert not at.dataframe
    spec = _plotly_spec(at)
    assert [trace["name"] for trace in spec["data"]] == ["Forecast"]
    assert "Largest returned prediction" in text


@pytest.mark.parametrize("code", ["model_unavailable", "forecast_unavailable"])
def test_forecast_availability_error_keeps_independent_history_only(code: str) -> None:
    client = RecordingClient(forecast=_error(code))

    at = _click(_run(client), "load_today_location")

    text = _visible_text(at)
    assert "Forecast unavailable" in text
    assert "Largest returned prediction" not in text
    assert len(at.dataframe) == 1
    spec = _plotly_spec(at)
    assert [trace["name"] for trace in spec["data"]] == ["Observed"]


def test_history_failure_is_visible_and_never_removes_current_forecast() -> None:
    client = RecordingClient(history=_error("api_request_failed", "Observations request failed."))

    at = _click(_run(client), "load_today_location")

    text = _visible_text(at)
    assert "Observations unavailable" in text
    assert "Observations request failed." in text
    assert not at.dataframe
    spec = _plotly_spec(at)
    assert [trace["name"] for trace in spec["data"]] == ["Forecast"]
    assert "Largest returned prediction" in text


def test_other_forecast_error_is_visible_without_history_or_fallback() -> None:
    client = RecordingClient(
        forecast=_error("invalid_api_response", "Forecast response was invalid.")
    )

    at = _click(_run(client), "load_today_location")

    assert [call[0] for call in client.calls].count("history") == 0
    text = _visible_text(at)
    assert "Forecast response was invalid." in text
    assert "Largest returned prediction" not in text
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_metrics_are_requested_only_by_explicit_historical_evaluation_action() -> None:
    client = RecordingClient()
    at = _run(client)

    assert [call[0] for call in client.calls].count("metrics") == 0
    assert not at.metric

    at = _click(at, "view_historical_model_evaluation")

    assert [call[0] for call in client.calls].count("metrics") == 1
    text = _visible_text(at)
    assert "Historical evaluation context" in text
    assert [metric.label for metric in at.metric] == [
        "Historical MAE",
        "Historical RMSE",
        "Historical WAPE",
    ]
    assert "current accuracy" not in text.lower()


def test_metrics_unavailable_is_optional_visible_state() -> None:
    client = RecordingClient(
        metrics=_error("metrics_unavailable", "Evaluation metrics were not returned.")
    )

    at = _click(_run(client), "view_historical_model_evaluation")

    assert "Historical evaluation unavailable" in _visible_text(at)
    assert "Evaluation metrics were not returned." in _visible_text(at)
    assert not at.metric


def test_catalog_refresh_clears_focus_that_is_no_longer_active() -> None:
    client = RecordingClient()
    at = _run(client)
    at.session_state["selected_location_id"] = 999

    at = at.run()

    assert "selected_location_id" not in at.session_state.filtered_state
