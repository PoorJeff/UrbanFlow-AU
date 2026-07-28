from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from urbanflow.api.schemas import ForecastResponse, HistoryResponse, SensorResponse
from urbanflow.dashboard.charts import build_forecast_figure
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.context import (
    clear_selected_location_if_missing,
    get_selected_location_id,
    set_selected_location_id,
)
from urbanflow.dashboard.pages.shared import load_page_context, render_api_error
from urbanflow.dashboard.snapshots import ForecastSnapshot, load_forecast_snapshot
from urbanflow.dashboard.time_utils import format_melbourne_timestamp

FORECAST_ERROR_HEADINGS = {
    "model_unavailable": "Forecast unavailable",
    "forecast_unavailable": "Forecast unavailable",
    "data_store_unavailable": "Forecast data unavailable",
    "sensor_not_found": "Selected sensor unavailable",
    "invalid_api_response": "Forecast response invalid",
}
FORECAST_SENSOR_SELECTOR_KEY = "forecast_sensor_selector"


def _record_forecast_sensor_focus() -> None:
    set_selected_location_id(
        st.session_state,
        st.session_state[FORECAST_SENSOR_SELECTOR_KEY],
    )


def render_forecast(client: DashboardApiClient) -> None:
    st.title("Forecast")
    st.write("Choose an active sensor and horizon, then request the forecast returned by the API.")

    context = load_page_context(client)
    if context.health is None:
        assert context.error is not None
        render_api_error(context.error, heading="Service check unavailable")
        return
    if context.health.status == "unavailable":
        st.error("Service unavailable: the API reported that service is unavailable.")
        return
    if context.error is not None:
        render_api_error(context.error, heading="Sensor catalog unavailable")
        return

    assert context.sensors is not None
    previous_selected_location_id = get_selected_location_id(st.session_state)
    selected_location_id = clear_selected_location_if_missing(
        st.session_state,
        context.sensors.data,
    )
    if previous_selected_location_id is not None and (
        selected_location_id is None
        or st.session_state.get(FORECAST_SENSOR_SELECTOR_KEY) != selected_location_id
    ):
        st.session_state.pop(FORECAST_SENSOR_SELECTOR_KEY, None)
    if not context.sensors.data:
        st.info("No active sensors were returned.")
        return

    _render_forecast_form(
        client,
        context.sensors.data,
        selected_location_id=selected_location_id,
    )


def _render_forecast_form(
    client: DashboardApiClient,
    sensors: Sequence[SensorResponse],
    *,
    selected_location_id: int | None,
) -> None:
    location_ids = [sensor.location_id for sensor in sensors]
    initial_index = (
        location_ids.index(selected_location_id) if selected_location_id in location_ids else 0
    )
    sensors_by_id = {sensor.location_id: sensor for sensor in sensors}
    location_id = st.selectbox(
        "Active sensor",
        options=location_ids,
        index=initial_index,
        format_func=lambda value: f"{sensors_by_id[value].sensor_name} (location {value})",
        key=FORECAST_SENSOR_SELECTOR_KEY,
        on_change=_record_forecast_sensor_focus,
    )

    with st.form("forecast_request_form"):
        horizon = st.number_input(
            "Forecast horizon (hours)",
            min_value=1,
            max_value=24,
            value=24,
            step=1,
            key="forecast_horizon",
        )
        submitted = st.form_submit_button(
            "Load returned forecast",
            key="load_forecast",
        )

    if not submitted:
        return
    if type(horizon) is not int or not 1 <= horizon <= 24:
        st.error("Forecast horizon must be a whole number between 1 and 24 hours.")
        return

    set_selected_location_id(st.session_state, location_id)
    snapshot = load_forecast_snapshot(
        client,
        location_id=location_id,
        horizon=horizon,
    )
    _render_snapshot(sensors_by_id[location_id], snapshot)


def _render_snapshot(sensor: SensorResponse, snapshot: ForecastSnapshot) -> None:
    if snapshot.forecast_error is not None:
        render_api_error(
            snapshot.forecast_error,
            heading=FORECAST_ERROR_HEADINGS.get(
                snapshot.forecast_error.code,
                "Forecast request failed",
            ),
        )
        return

    assert snapshot.forecast is not None
    st.header(f"{sensor.sensor_name} (location {sensor.location_id})")
    st.write(sensor.sensor_description)
    _render_forecast_metadata(snapshot.forecast)

    chart_history = (
        snapshot.history if snapshot.history is not None and snapshot.history.data else None
    )
    if snapshot.history_error is not None:
        render_api_error(
            snapshot.history_error,
            heading="Returned history unavailable",
        )
        st.info(
            "Matching returned history is unavailable; "
            "the chart and table show forecast values only."
        )
    elif snapshot.history is not None:
        if not snapshot.history.data:
            st.info("No observations were returned for the matching interval.")
        _render_history_table(snapshot.history)

    st.subheader("Returned observations and forecast")
    st.caption("Observed values use a solid line. Forecast values use a dashed line.")
    st.plotly_chart(
        build_forecast_figure(
            history=chart_history,
            forecast=snapshot.forecast,
        ),
        width="stretch",
    )
    _render_prediction_table(snapshot.forecast)
    _render_largest_prediction(snapshot.forecast)


def _render_forecast_metadata(forecast: ForecastResponse) -> None:
    st.subheader("Returned forecast context")
    st.write(f"Model: {forecast.model_name}")
    st.write(f"Model version: {forecast.model_version or 'Not returned'}")
    st.write(f"Generated at: {format_melbourne_timestamp(forecast.generated_at)}")
    st.write(f"Forecast origin: {format_melbourne_timestamp(forecast.forecast_origin_at)}")
    st.write(f"Data cutoff: {format_melbourne_timestamp(forecast.data_cutoff_at)}")


def _render_history_table(history: HistoryResponse) -> None:
    st.subheader("Returned observation table")
    rows: list[dict[str, object]] | dict[str, list[object]]
    if history.data:
        rows = [
            {
                "Observed at": format_melbourne_timestamp(point.observed_at),
                "Pedestrian count": point.pedestrian_count,
            }
            for point in history.data
        ]
    else:
        rows = {"Observed at": [], "Pedestrian count": []}
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_prediction_table(forecast: ForecastResponse) -> None:
    st.subheader("Returned prediction table")
    st.dataframe(
        [
            {
                "Forecast horizon": prediction.forecast_horizon,
                "Target at": format_melbourne_timestamp(prediction.target_at),
                "Predicted count": prediction.predicted_count,
            }
            for prediction in forecast.predictions
        ],
        width="stretch",
        hide_index=True,
    )


def _render_largest_prediction(forecast: ForecastResponse) -> None:
    returned_nonnegative = [
        prediction for prediction in forecast.predictions if prediction.predicted_count >= 0
    ]
    if not returned_nonnegative:
        return

    largest = max(
        returned_nonnegative,
        key=lambda prediction: prediction.predicted_count,
    )
    st.write(
        "Largest returned prediction: "
        f"{largest.predicted_count} pedestrians at "
        f"{format_melbourne_timestamp(largest.target_at)}."
    )
