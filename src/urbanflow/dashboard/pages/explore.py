from __future__ import annotations

from datetime import timedelta

import streamlit as st

from urbanflow.dashboard.charts import build_history_figure
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.context import (
    clear_selected_location_if_missing,
    set_selected_location_id,
)
from urbanflow.dashboard.errors import DashboardApiError
from urbanflow.dashboard.pages.shared import load_page_context, render_api_error
from urbanflow.dashboard.time_utils import (
    format_melbourne_timestamp,
    local_midnight,
    melbourne_now,
    validate_history_interval,
)

HISTORY_ERROR_HEADINGS = {
    "sensor_not_found": "Selected sensor unavailable",
    "data_store_unavailable": "History data unavailable",
    "invalid_api_response": "History response invalid",
}


def render_explore(client: DashboardApiClient) -> None:
    st.title("Explore")

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
    selected_location_id = clear_selected_location_if_missing(
        st.session_state,
        context.sensors.data,
    )
    if not context.sensors.data:
        st.info("No active sensors were returned.")
        return

    location_ids = [sensor.location_id for sensor in context.sensors.data]
    initial_index = (
        location_ids.index(selected_location_id) if selected_location_id in location_ids else 0
    )
    sensors_by_id = {sensor.location_id: sensor for sensor in context.sensors.data}
    end_date = melbourne_now().date()
    selector_was_initialized = "explore_sensor_selector" in st.session_state
    location_id = st.selectbox(
        "Active sensor",
        options=location_ids,
        index=initial_index,
        format_func=lambda value: f"{sensors_by_id[value].sensor_name} (location {value})",
        key="explore_sensor_selector",
    )

    if selector_was_initialized and location_id != selected_location_id:
        set_selected_location_id(st.session_state, location_id)

    with st.form("explore_history_form"):
        start_date = st.date_input(
            "Start date",
            value=end_date - timedelta(days=7),
            key="explore_start_date",
        )
        selected_end_date = st.date_input(
            "End date",
            value=end_date,
            key="explore_end_date",
        )
        submitted = st.form_submit_button(
            "Load history",
            key="load_explore_history",
        )

    if not submitted:
        return

    set_selected_location_id(st.session_state, location_id)
    start = local_midnight(start_date)
    end = local_midnight(selected_end_date)
    validation_error = validate_history_interval(start, end)
    if validation_error is not None:
        st.error(f"Invalid history interval: {validation_error}")
        return

    try:
        history = client.get_history(location_id, start=start, end=end)
    except DashboardApiError as error:
        render_api_error(
            error,
            heading=HISTORY_ERROR_HEADINGS.get(error.code, "History request failed"),
        )
        return

    if not history.data:
        st.info("No observations were returned in the selected interval.")
        return

    sensor = sensors_by_id[location_id]
    st.header(f"{sensor.sensor_name} (location {sensor.location_id})")
    st.write(
        "Returned interval: "
        f"{format_melbourne_timestamp(history.start)} to "
        f"{format_melbourne_timestamp(history.end)}"
    )
    st.subheader("Observed history")
    st.caption("Observed values use a solid line.")
    st.plotly_chart(build_history_figure(history), width="stretch")
    st.subheader("Returned observation table")
    st.dataframe(
        [
            {
                "Observed at": format_melbourne_timestamp(point.observed_at),
                "Pedestrian count": point.pedestrian_count,
            }
            for point in history.data
        ],
        width="stretch",
        hide_index=True,
    )
