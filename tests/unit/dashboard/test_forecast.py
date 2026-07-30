from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from urbanflow.api.schemas import (
    ComponentHealth,
    ForecastPredictionResponse,
    ForecastResponse,
    HealthComponents,
    HealthResult,
    HistoryPoint,
    HistoryResponse,
    SensorListMeta,
    SensorListResponse,
    SensorResponse,
)
from urbanflow.dashboard.errors import DashboardApiError


def _error(code: str, message: str | None = None) -> DashboardApiError:
    return DashboardApiError(
        status_code=None if code == "api_unreachable" else 503,
        code=code,
        message=message or f"{code} test message",
        details=(),
    )


def _health(status: str = "ok") -> HealthResult:
    component_status = "available" if status == "ok" else "unavailable"
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


def _forecast(
    location_id: int = 101,
    *,
    horizon: int = 24,
    model_version: str | None = "model-v1",
    predicted_counts: list[float] | None = None,
    target_hour_offsets: list[int] | None = None,
) -> ForecastResponse:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    counts = predicted_counts or [float(index) for index in range(1, horizon + 1)]
    offsets = target_hour_offsets or list(range(1, horizon + 1))
    return ForecastResponse(
        location_id=location_id,
        model_name="lightgbm",
        model_version=model_version,
        generated_at=cutoff + timedelta(minutes=5),
        forecast_origin_at=cutoff + timedelta(minutes=30),
        data_cutoff_at=cutoff,
        horizon_hours=horizon,
        predictions=[
            ForecastPredictionResponse(
                forecast_horizon=index,
                target_at=cutoff + timedelta(hours=offset),
                predicted_count=count,
            )
            for index, (offset, count) in enumerate(zip(offsets, counts, strict=True), start=1)
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


class RecordingClient:
    def __init__(
        self,
        *,
        health: HealthResult | DashboardApiError | None = None,
        sensors: SensorListResponse | DashboardApiError | None = None,
        forecast: ForecastResponse | DashboardApiError | None = None,
        history: HistoryResponse | DashboardApiError | None = None,
    ) -> None:
        self.health = health if health is not None else _health()
        self.sensors = sensors if sensors is not None else _sensors(101, 202)
        self.forecast = forecast
        self.history = history if history is not None else _history()
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
        if self.forecast is None:
            return _forecast(location_id, horizon=horizon)
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


def _forecast_harness(client):
    from urbanflow.dashboard.pages.forecast import render_forecast

    render_forecast(client)


def _dashboard_harness(client):
    from urbanflow.dashboard.application import render_dashboard

    render_dashboard(client)


def _run(client: RecordingClient) -> AppTest:
    return AppTest.from_function(_forecast_harness, args=(client,)).run()


def _submit(at: AppTest) -> AppTest:
    return at.button(key="load_forecast").click().run()


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
        "button",
        "selectbox",
        "number_input",
        "radio",
    ):
        for element in at.get(element_type):
            for attribute in ("value", "label"):
                value = getattr(element, attribute, None)
                if value is not None:
                    values.append(str(value))
    return "\n".join(values)


def _forecast_calls(client: RecordingClient) -> list[tuple[Any, ...]]:
    return [call for call in client.calls if call[0] == "forecast"]


def _history_calls(client: RecordingClient) -> list[tuple[Any, ...]]:
    return [call for call in client.calls if call[0] == "history"]


def _button_with_label(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def _dataframe_with_columns(at: AppTest, columns: list[str]):
    return next(frame for frame in at.dataframe if list(frame.value.columns) == columns)


def test_first_visit_loads_fresh_context_and_labels_bounded_controls() -> None:
    client = RecordingClient()

    at = _run(client)

    assert not at.exception
    assert client.calls == [("health",), ("sensors", True)]
    selector = at.selectbox(key="forecast_sensor_selector")
    horizon = at.number_input(key="forecast_horizon")
    assert selector.label == "Active sensor"
    assert selector.proto.form_id == ""
    assert horizon.label == "Forecast horizon (hours)"
    assert horizon.value == 24
    assert horizon.min == 1
    assert horizon.max == 24
    assert horizon.step == 1
    assert horizon.proto.form_id == "forecast_request_form"
    assert at.button(key="load_forecast").label == "Load returned forecast"
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_focus_prefills_only_from_the_current_active_catalog() -> None:
    active_client = RecordingClient()
    active_at = AppTest.from_function(_forecast_harness, args=(active_client,))
    active_at.session_state["selected_location_id"] = 202
    active_at = active_at.run()

    assert active_at.selectbox(key="forecast_sensor_selector").value == 202

    stale_client = RecordingClient()
    stale_at = AppTest.from_function(_forecast_harness, args=(stale_client,))
    stale_at.session_state["selected_location_id"] = 999
    stale_at = stale_at.run()

    assert stale_at.selectbox(key="forecast_sensor_selector").value == 101
    assert "selected_location_id" not in stale_at.session_state.filtered_state


def test_catalog_refresh_clears_removed_initialized_focus_without_defaulting() -> None:
    client = RecordingClient()
    at = AppTest.from_function(_forecast_harness, args=(client,))
    at.session_state["selected_location_id"] = 202
    at = at.run()
    assert at.selectbox(key="forecast_sensor_selector").value == 202

    client.sensors = _sensors(101)
    at = at.run()

    assert at.selectbox(key="forecast_sensor_selector").value == 101
    assert "selected_location_id" not in at.session_state.filtered_state
    assert _forecast_calls(client) == []
    assert _history_calls(client) == []

    at = at.run()

    assert at.selectbox(key="forecast_sensor_selector").value == 101
    assert "selected_location_id" not in at.session_state.filtered_state
    assert _forecast_calls(client) == []
    assert _history_calls(client) == []


def test_catalog_refresh_clears_removed_focus_despite_different_stale_widget() -> None:
    client = RecordingClient(sensors=_sensors(101))
    at = AppTest.from_function(_forecast_harness, args=(client,))
    at.session_state["selected_location_id"] = 202
    at.session_state["forecast_sensor_selector"] = 101

    at = at.run()

    assert at.selectbox(key="forecast_sensor_selector").value == 101
    assert "selected_location_id" not in at.session_state.filtered_state
    assert _forecast_calls(client) == []
    assert _history_calls(client) == []


def test_valid_business_focus_overrides_different_stale_widget() -> None:
    client = RecordingClient()
    at = AppTest.from_function(_forecast_harness, args=(client,))
    at.session_state["selected_location_id"] = 202
    at.session_state["forecast_sensor_selector"] = 101

    at = at.run()

    assert at.selectbox(key="forecast_sensor_selector").value == 202
    assert at.session_state.filtered_state["selected_location_id"] == 202
    assert _forecast_calls(client) == []
    assert _history_calls(client) == []


def test_sensor_and_horizon_changes_never_request_forecast_without_submit() -> None:
    client = RecordingClient()
    at = _run(client)

    at = at.selectbox(key="forecast_sensor_selector").select(202).run()
    at.number_input(key="forecast_horizon").set_value(1)
    at = at.run()

    assert at.session_state.filtered_state["selected_location_id"] == 202
    assert _forecast_calls(client) == []
    assert _history_calls(client) == []
    assert [call[0] for call in client.calls] == [
        "health",
        "sensors",
        "health",
        "sensors",
        "health",
        "sensors",
    ]


@pytest.mark.parametrize("horizon", [1, 24])
def test_submit_accepts_inclusive_horizon_boundaries(horizon: int) -> None:
    client = RecordingClient()
    at = _run(client)
    at.number_input(key="forecast_horizon").set_value(horizon)

    _submit(at)

    assert _forecast_calls(client) == [("forecast", 101, horizon)]


@pytest.mark.parametrize("invalid_horizon", [0, 25])
def test_number_input_rejects_out_of_bounds_before_any_request(
    invalid_horizon: int,
) -> None:
    client = RecordingClient()
    at = _run(client)

    at.number_input(key="forecast_horizon").set_value(invalid_horizon)
    at = at.run()

    assert at.number_input(key="forecast_horizon").value == 24
    assert _forecast_calls(client) == []
    assert _history_calls(client) == []


def test_submit_loads_forecast_once_then_exact_cutoff_aligned_history() -> None:
    client = RecordingClient()

    _submit(_run(client))

    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    end = cutoff + timedelta(microseconds=1)
    assert client.calls[-2:] == [
        ("forecast", 101, 24),
        ("history", 101, end - timedelta(hours=24), end),
    ]
    assert len(_forecast_calls(client)) == 1


def test_full_success_renders_factual_metadata_ordered_tables_and_labelled_chart() -> None:
    client = RecordingClient(
        forecast=_forecast(
            horizon=3,
            predicted_counts=[12.5, 27.0, 18.25],
            target_hour_offsets=[3, 2, 1],
        )
    )
    at = _run(client)
    at.number_input(key="forecast_horizon").set_value(3)

    at = _submit(at)

    text = _visible_text(at)
    assert "Sensor 101" in text
    assert "Model: lightgbm" in text
    assert "Model version: model-v1" in text
    assert "Generated at:" in text
    assert "Forecast origin:" in text
    assert "Data cutoff:" in text
    assert "Largest returned prediction: 27.0 pedestrians at 12 Jul 2026, 22:00 AEST." in text
    assert "Observed values use a solid line. Forecast values use a dashed line." in text

    history_table = _dataframe_with_columns(
        at,
        ["Observed at", "Pedestrian count"],
    )
    assert history_table.value["Pedestrian count"].tolist() == [24, 31]
    prediction_table = _dataframe_with_columns(
        at,
        ["Forecast horizon", "Target at", "Predicted count"],
    )
    assert prediction_table.value["Forecast horizon"].tolist() == [1, 2, 3]
    assert prediction_table.value["Predicted count"].tolist() == [12.5, 27.0, 18.25]
    assert prediction_table.value["Target at"].tolist() == [
        "12 Jul 2026, 23:00 AEST",
        "12 Jul 2026, 22:00 AEST",
        "12 Jul 2026, 21:00 AEST",
    ]

    chart = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert [trace["name"] for trace in chart["data"]] == ["Observed", "Forecast"]
    assert chart["data"][0]["line"]["dash"] == "solid"
    assert chart["data"][1]["line"]["dash"] == "dash"
    assert chart["data"][1]["x"] == [
        "2026-07-12T23:00:00+10:00",
        "2026-07-12T22:00:00+10:00",
        "2026-07-12T21:00:00+10:00",
    ]
    assert chart["data"][1]["y"] == [12.5, 27.0, 18.25]


def test_largest_prediction_preserves_high_precision_api_value() -> None:
    client = RecordingClient(
        forecast=_forecast(
            horizon=1,
            predicted_counts=[12.3456789],
        )
    )
    at = _run(client)
    at.number_input(key="forecast_horizon").set_value(1)

    at = _submit(at)

    assert (
        "Largest returned prediction: 12.3456789 pedestrians at 12 Jul 2026, 21:00 AEST."
    ) in _visible_text(at)


def test_nullable_model_version_is_reported_without_invention() -> None:
    client = RecordingClient(forecast=_forecast(model_version=None))

    at = _submit(_run(client))

    assert "Model version: Not returned" in _visible_text(at)


def test_auxiliary_history_failure_keeps_only_truthful_forecast_outputs() -> None:
    history_error = _error("api_unreachable", "Matching history could not be reached.")
    client = RecordingClient(history=history_error)

    at = _submit(_run(client))

    text = _visible_text(at)
    assert "Returned history unavailable" in text
    assert history_error.message in text
    assert "forecast values only" in text
    assert len(at.dataframe) == 1
    assert list(at.dataframe[0].value.columns) == [
        "Forecast horizon",
        "Target at",
        "Predicted count",
    ]
    chart = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert [trace["name"] for trace in chart["data"]] == ["Forecast"]
    assert "Largest returned prediction" in text


@pytest.mark.parametrize(
    ("error", "heading"),
    [
        (_error("model_unavailable"), "Forecast unavailable"),
        (_error("forecast_unavailable"), "Forecast unavailable"),
        (_error("data_store_unavailable"), "Forecast data unavailable"),
        (_error("sensor_not_found"), "Selected sensor unavailable"),
        (_error("invalid_api_response"), "Forecast response invalid"),
        (_error("api_unreachable"), "Forecast request failed"),
    ],
)
def test_forecast_generation_errors_have_no_output_or_fallback(
    error: DashboardApiError,
    heading: str,
) -> None:
    client = RecordingClient(forecast=error)

    at = _submit(_run(client))

    text = _visible_text(at)
    assert heading in text
    assert error.message in text
    assert _history_calls(client) == []
    assert not at.dataframe
    assert not at.get("plotly_chart")
    assert "Largest returned prediction" not in text


def test_forecast_failure_clears_an_earlier_success_without_stale_output() -> None:
    client = RecordingClient()
    at = _submit(_run(client))
    assert at.dataframe
    assert at.get("plotly_chart")

    client.forecast = _error("forecast_unavailable")
    at = _submit(at)

    assert _forecast_calls(client) == [
        ("forecast", 101, 24),
        ("forecast", 101, 24),
    ]
    assert len(_history_calls(client)) == 1
    assert not at.dataframe
    assert not at.get("plotly_chart")
    assert "Largest returned prediction" not in _visible_text(at)


def test_changing_sensor_records_focus_and_clears_previous_forecast_output() -> None:
    client = RecordingClient()
    at = _submit(_run(client))
    assert at.dataframe
    assert at.get("plotly_chart")

    at = at.selectbox(key="forecast_sensor_selector").select(202).run()

    assert at.session_state.filtered_state["selected_location_id"] == 202
    assert len(_forecast_calls(client)) == 1
    assert len(_history_calls(client)) == 1
    assert not at.dataframe
    assert not at.get("plotly_chart")
    assert "Largest returned prediction" not in _visible_text(at)


def test_success_never_caches_responses_figures_tables_or_metrics_in_state() -> None:
    client = RecordingClient()

    at = _submit(_run(client))

    state = at.session_state.filtered_state
    assert state == {
        "forecast_horizon": 24,
        "forecast_sensor_selector": 101,
        "load_forecast": False,
        "selected_location_id": 101,
    }
    assert all(
        not isinstance(
            value,
            (DashboardApiError, ForecastResponse, HistoryResponse, dict, list),
        )
        for value in state.values()
    )


def test_today_question_navigates_to_forecast_with_only_selected_id() -> None:
    client = RecordingClient()
    at = AppTest.from_function(_dashboard_harness, args=(client,))
    at = at.run()

    assert at.radio(key="dashboard_page").options == ["Today", "Explore", "Forecast"]
    at = at.button(key="load_today_location").click().run()
    assert at.session_state.filtered_state["selected_location_id"] == 101
    assert _forecast_calls(client) == [("forecast", 101, 24)]
    assert len(_history_calls(client)) == 1
    assert at.dataframe
    assert at.get("plotly_chart")

    action = "What returned forecast is available next?"
    assert action in _visible_text(at)

    at = _button_with_label(at, action).click().run()

    assert at.title[0].value == "Forecast"
    assert at.radio(key="dashboard_page").value == "Forecast"
    assert at.selectbox(key="forecast_sensor_selector").value == 101
    assert client.calls[-2:] == [("health",), ("sensors", True)]
    assert _forecast_calls(client) == [("forecast", 101, 24)]
    assert len(_history_calls(client)) == 1
    assert "Latest returned observation" not in _visible_text(at)
    assert "Returned forecast details" not in _visible_text(at)
    assert not at.dataframe
    assert not at.get("plotly_chart")
    state = at.session_state.filtered_state
    assert state["selected_location_id"] == 101
    assert all(
        not isinstance(value, (DashboardApiError, ForecastResponse, HistoryResponse))
        for value in state.values()
    )
