from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from urbanflow.api.schemas import (
    ComponentHealth,
    HealthComponents,
    HealthResult,
    HistoryPoint,
    HistoryResponse,
    SensorListMeta,
    SensorListResponse,
    SensorResponse,
)
from urbanflow.dashboard.errors import DashboardApiError
from urbanflow.dashboard.time_utils import MELBOURNE_TIME_ZONE


def _health() -> HealthResult:
    return HealthResult(
        status="ok",
        service="urbanflow-au-api",
        version="0.1.0",
        generated_at=datetime(2026, 7, 26, 2, tzinfo=UTC),
        components=HealthComponents(
            api_process=ComponentHealth(status="available"),
            model_provider=ComponentHealth(status="available"),
            data_store=ComponentHealth(status="available"),
            data_freshness=ComponentHealth(status="available"),
        ),
        model_version="model-v1",
        data_cutoff_at=datetime(2026, 7, 26, 1, tzinfo=UTC),
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


class RecordingClient:
    def __init__(
        self,
        *,
        sensors: SensorListResponse | DashboardApiError | None = None,
        history: HistoryResponse | DashboardApiError | None = None,
    ) -> None:
        self.sensors = sensors if sensors is not None else _sensors(101, 202)
        self.history = history if history is not None else _history()
        self.calls: list[tuple[Any, ...]] = []

    @staticmethod
    def _return_or_raise(result: Any) -> Any:
        if isinstance(result, DashboardApiError):
            raise result
        return result

    def get_health(self) -> HealthResult:
        self.calls.append(("health",))
        return _health()

    def list_sensors(self, *, active_only: bool = True) -> SensorListResponse:
        self.calls.append(("sensors", active_only))
        return self._return_or_raise(self.sensors)

    def get_history(
        self,
        location_id: int,
        *,
        start: datetime,
        end: datetime,
    ) -> HistoryResponse:
        self.calls.append(("history", location_id, start, end))
        return self._return_or_raise(self.history)


def _history(
    location_id: int = 101,
    *,
    empty: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
) -> HistoryResponse:
    returned_start = start or datetime(2026, 4, 1, tzinfo=MELBOURNE_TIME_ZONE)
    returned_end = end or datetime(2026, 4, 8, tzinfo=MELBOURNE_TIME_ZONE)
    return HistoryResponse(
        location_id=location_id,
        start=returned_start,
        end=returned_end,
        data=[]
        if empty
        else [
            HistoryPoint(
                observed_at=datetime(2026, 4, 2, 9, tzinfo=MELBOURNE_TIME_ZONE),
                pedestrian_count=24,
            ),
            HistoryPoint(
                observed_at=datetime(2026, 4, 3, 10, tzinfo=MELBOURNE_TIME_ZONE),
                pedestrian_count=31,
            ),
        ],
    )


def _error(code: str, message: str) -> DashboardApiError:
    return DashboardApiError(
        status_code=None if code == "api_unreachable" else 503,
        code=code,
        message=message,
        details=(),
    )


def _explore_harness(client):
    from urbanflow.dashboard.pages.explore import render_explore

    render_explore(client)


def _run(client: RecordingClient) -> AppTest:
    return AppTest.from_function(_explore_harness, args=(client,)).run()


def _dashboard_harness(client):
    from urbanflow.dashboard.application import render_dashboard

    render_dashboard(client)


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
        "button",
        "selectbox",
        "date_input",
        "radio",
    ):
        for element in at.get(element_type):
            for attribute in ("value", "label"):
                value = getattr(element, attribute, None)
                if value is not None:
                    values.append(str(value))
    return "\n".join(values)


def _submit(at: AppTest) -> AppTest:
    return at.button(key="load_explore_history").click().run()


def _history_calls(client: RecordingClient) -> list[tuple[Any, ...]]:
    return [call for call in client.calls if call[0] == "history"]


def _button_with_label(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def test_first_visit_requests_only_current_context_and_labels_the_form() -> None:
    client = RecordingClient()

    at = _run(client)

    assert not at.exception
    assert client.calls == [("health",), ("sensors", True)]
    assert at.selectbox(key="explore_sensor_selector").label == "Active sensor"
    assert at.date_input(key="explore_start_date").label == "Start date"
    assert at.date_input(key="explore_end_date").label == "End date"
    assert at.button(key="load_explore_history").label == "Load history"
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_default_dates_are_the_most_recent_seven_complete_melbourne_days(
    monkeypatch,
) -> None:
    from urbanflow.dashboard.pages import explore

    monkeypatch.setattr(
        explore,
        "melbourne_now",
        lambda: datetime(2026, 4, 8, 15, tzinfo=MELBOURNE_TIME_ZONE),
    )

    at = _run(RecordingClient())

    assert at.date_input(key="explore_start_date").value == date(2026, 4, 1)
    assert at.date_input(key="explore_end_date").value == date(2026, 4, 8)


def test_focus_prefills_only_when_it_is_in_the_current_active_catalog() -> None:
    active_client = RecordingClient()
    active_at = AppTest.from_function(
        _explore_harness,
        args=(active_client,),
    )
    active_at.session_state["selected_location_id"] = 202
    active_at = active_at.run()

    assert active_at.selectbox(key="explore_sensor_selector").value == 202

    stale_client = RecordingClient()
    stale_at = AppTest.from_function(
        _explore_harness,
        args=(stale_client,),
    )
    stale_at.session_state["selected_location_id"] = 999
    stale_at = stale_at.run()

    assert stale_at.selectbox(key="explore_sensor_selector").value == 101
    assert "selected_location_id" not in stale_at.session_state.filtered_state


def test_changing_sensor_or_dates_updates_focus_without_requesting_history() -> None:
    client = RecordingClient()
    at = _run(client)

    selector = at.selectbox(key="explore_sensor_selector")
    assert selector.proto.form_id == ""
    at = selector.select(202).run()
    at.date_input(key="explore_start_date").set_value(date(2026, 4, 2))

    assert _history_calls(client) == []
    assert at.session_state.filtered_state["selected_location_id"] == 202
    assert [call[0] for call in client.calls] == [
        "health",
        "sensors",
        "health",
        "sensors",
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2026, 4, 8), date(2026, 4, 8)),
        (date(2026, 4, 9), date(2026, 4, 8)),
        (date(2026, 3, 6), date(2026, 4, 6)),
    ],
    ids=["equal", "reverse", "31-calendar-days-crossing-dst"],
)
def test_invalid_intervals_are_visible_and_never_request_history(
    start: date,
    end: date,
) -> None:
    client = RecordingClient()
    at = _run(client)
    at.date_input(key="explore_start_date").set_value(start)
    at.date_input(key="explore_end_date").set_value(end)

    at = _submit(at)

    assert "Invalid history interval" in _visible_text(at)
    assert _history_calls(client) == []
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_exact_31_elapsed_days_are_accepted() -> None:
    client = RecordingClient(history=_history(empty=True))
    at = _run(client)
    at.date_input(key="explore_start_date").set_value(date(2026, 1, 1))
    at.date_input(key="explore_end_date").set_value(date(2026, 2, 1))

    at = _submit(at)

    assert not at.exception
    assert len(_history_calls(client)) == 1


def test_submit_preserves_exact_offset_aware_local_midnights() -> None:
    client = RecordingClient(history=_history(empty=True))
    at = _run(client)
    at = at.selectbox(key="explore_sensor_selector").select(202).run()
    at.date_input(key="explore_start_date").set_value(date(2026, 4, 1))
    at.date_input(key="explore_end_date").set_value(date(2026, 4, 8))

    _submit(at)

    assert _history_calls(client) == [
        (
            "history",
            202,
            datetime(2026, 4, 1, tzinfo=MELBOURNE_TIME_ZONE),
            datetime(2026, 4, 8, tzinfo=MELBOURNE_TIME_ZONE),
        )
    ]
    _, _, start, end = _history_calls(client)[0]
    assert start.utcoffset().total_seconds() == 11 * 60 * 60
    assert end.utcoffset().total_seconds() == 10 * 60 * 60


def test_non_empty_success_renders_identity_interval_observed_chart_and_table() -> None:
    client = RecordingClient()

    at = _submit(_run(client))

    text = _visible_text(at)
    assert "Sensor 101" in text
    assert "location 101" in text
    assert "Returned interval" in text
    assert "01 Apr 2026, 00:00 AEDT" in text
    assert "08 Apr 2026, 00:00 AEST" in text
    assert list(at.dataframe[0].value.columns) == ["Observed at", "Pedestrian count"]
    assert at.dataframe[0].value["Pedestrian count"].tolist() == [24, 31]
    chart = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert [trace["name"] for trace in chart["data"]] == ["Observed"]
    assert chart["data"][0]["line"]["dash"] == "solid"

    prohibited = (
        "live",
        "fresh",
        "accurate",
        "busy",
        "rising",
        "healthy",
        "city-wide",
        "caused",
    )
    assert not any(claim in text.lower() for claim in prohibited)
    state = at.session_state.filtered_state
    assert state["selected_location_id"] == 101
    assert set(state) == {
        "selected_location_id",
        "explore_sensor_selector",
        "explore_start_date",
        "explore_end_date",
        "load_explore_history",
    }
    assert all(
        not isinstance(value, (DashboardApiError, HistoryResponse)) for value in state.values()
    )


def test_empty_success_only_reports_no_observations_for_the_returned_interval() -> None:
    client = RecordingClient(history=_history(empty=True))

    at = _submit(_run(client))

    assert "No observations were returned in the selected interval." in _visible_text(at)
    assert "Returned interval" not in _visible_text(at)
    assert not at.dataframe
    assert not at.get("plotly_chart")


@pytest.mark.parametrize(
    ("error", "expected_heading"),
    [
        (
            _error("sensor_not_found", "The selected sensor was not found."),
            "Selected sensor unavailable",
        ),
        (
            _error("data_store_unavailable", "The data store is unavailable."),
            "History data unavailable",
        ),
        (
            _error("invalid_api_response", "The history response was invalid."),
            "History response invalid",
        ),
        (
            _error("api_unreachable", "The dashboard API is unreachable."),
            "History request failed",
        ),
    ],
)
def test_known_history_errors_are_distinct_and_never_render_output(
    error: DashboardApiError,
    expected_heading: str,
) -> None:
    client = RecordingClient(history=error)

    at = _submit(_run(client))

    assert expected_heading in _visible_text(at)
    assert error.message in _visible_text(at)
    assert len(_history_calls(client)) == 1
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_changing_sensor_clears_the_previous_history_output() -> None:
    client = RecordingClient()
    at = _submit(_run(client))
    assert at.dataframe
    assert at.get("plotly_chart")

    selector = at.selectbox(key="explore_sensor_selector")
    assert selector.proto.form_id == ""
    at = selector.select(202).run()

    assert len(_history_calls(client)) == 1
    assert at.session_state.filtered_state["selected_location_id"] == 202
    assert "Returned interval" not in _visible_text(at)
    assert not at.dataframe
    assert not at.get("plotly_chart")


def test_today_action_navigates_with_selected_id_and_no_response_state() -> None:
    client = RecordingClient()
    at = AppTest.from_function(_dashboard_harness, args=(client,))
    at = at.run()

    assert at.radio(key="dashboard_page").options == ["Today", "Explore"]
    assert "How has this location changed?" not in _visible_text(at)
    at.session_state["selected_location_id"] = 101
    at = at.run()

    assert "How has this location changed?" in _visible_text(at)
    at = _button_with_label(at, "How has this location changed?").click().run()

    assert at.title[0].value == "Explore"
    assert at.selectbox(key="explore_sensor_selector").value == 101
    assert client.calls[-2:] == [("health",), ("sensors", True)]
    assert _history_calls(client) == []
    state = at.session_state.filtered_state
    assert state["selected_location_id"] == 101
    assert all(
        not isinstance(value, (DashboardApiError, HistoryResponse)) for value in state.values()
    )
