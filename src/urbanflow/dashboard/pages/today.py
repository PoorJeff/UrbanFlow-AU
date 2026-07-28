from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from urbanflow.api.schemas import (
    ForecastResponse,
    HealthResult,
    HistoryResponse,
    ModelMetricsResponse,
    SensorListResponse,
    SensorResponse,
)
from urbanflow.dashboard.charts import build_forecast_figure, build_history_figure
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.context import (
    clear_selected_location_if_missing,
    get_selected_location_id,
    set_selected_location_id,
)
from urbanflow.dashboard.errors import DashboardApiError
from urbanflow.dashboard.pages.shared import load_page_context, render_api_error
from urbanflow.dashboard.snapshots import TodaySnapshot, load_today_snapshot
from urbanflow.dashboard.time_utils import format_melbourne_timestamp

FORECAST_AVAILABILITY_ERRORS = frozenset({"model_unavailable", "forecast_unavailable"})
DASHBOARD_PAGE_KEY = "dashboard_page"


def render_today(client: DashboardApiClient) -> None:
    st.title("Today")
    st.write(
        "Choose an active sensor, then load its returned observations and forecast for review."
    )

    context = load_page_context(client)
    if context.health is None:
        assert context.error is not None
        render_api_error(context.error, heading="Service check unavailable")
        return

    if context.health.status == "unavailable":
        st.error("Service unavailable: the API reported that service is unavailable.")
        _render_data_and_model_context(context.health, None, client, allow_metrics=False)
        return

    if context.error is not None:
        render_api_error(context.error, heading="Sensor catalog unavailable")
        _render_data_and_model_context(context.health, None, client, allow_metrics=False)
        return

    assert context.sensors is not None
    clear_selected_location_if_missing(st.session_state, context.sensors.data)
    _render_data_and_model_context(context.health, context.sensors, client, allow_metrics=True)

    if not context.sensors.data:
        st.info("No active sensors were returned.")
        return

    _render_location_form(client, context.sensors.data)
    if get_selected_location_id(st.session_state) is not None:
        st.button(
            "How has this location changed?",
            on_click=_navigate_to_explore,
        )
        st.button(
            "What returned forecast is available next?",
            on_click=_navigate_to_forecast,
        )


def _navigate_to_explore() -> None:
    st.session_state[DASHBOARD_PAGE_KEY] = "Explore"


def _navigate_to_forecast() -> None:
    st.session_state[DASHBOARD_PAGE_KEY] = "Forecast"


def _render_location_form(
    client: DashboardApiClient,
    sensors: Sequence[SensorResponse],
) -> None:
    selected_location_id = get_selected_location_id(st.session_state)
    location_ids = [sensor.location_id for sensor in sensors]
    initial_index = (
        location_ids.index(selected_location_id) if selected_location_id in location_ids else 0
    )
    sensors_by_id = {sensor.location_id: sensor for sensor in sensors}

    with st.form("today_location_form"):
        location_id = st.selectbox(
            "Active sensor",
            options=location_ids,
            index=initial_index,
            format_func=lambda value: f"{sensors_by_id[value].sensor_name} (location {value})",
            key="today_location_selector",
        )
        submitted = st.form_submit_button(
            "Load this location",
            key="load_today_location",
        )

    if not submitted:
        st.info("Choose an active sensor and select Load this location to request details.")
        return

    set_selected_location_id(st.session_state, location_id)
    sensor = sensors_by_id[location_id]
    snapshot = load_today_snapshot(client, location_id=location_id)
    _render_snapshot(sensor, snapshot)


def _render_snapshot(sensor: SensorResponse, snapshot: TodaySnapshot) -> None:
    st.header(sensor.sensor_name)
    st.write(sensor.sensor_description)

    if snapshot.forecast_error is not None:
        if snapshot.forecast_error.code in FORECAST_AVAILABILITY_ERRORS:
            render_api_error(snapshot.forecast_error, heading="Forecast unavailable")
            _render_history(snapshot.history, snapshot.history_error)
        else:
            render_api_error(snapshot.forecast_error, heading="Location data unavailable")
        return

    assert snapshot.forecast is not None
    _render_forecast_with_history(
        forecast=snapshot.forecast,
        history=snapshot.history,
        history_error=snapshot.history_error,
    )


def _render_forecast_with_history(
    *,
    forecast: ForecastResponse,
    history: HistoryResponse | None,
    history_error: DashboardApiError | None,
) -> None:
    chart_history = history if history is not None and history.data else None
    if history_error is not None:
        render_api_error(history_error, heading="Observations unavailable")
    elif history is not None and not history.data:
        st.info("No observations were returned for the matching interval.")
    elif history is not None:
        latest = max(history.data, key=lambda point: point.observed_at)
        st.write(
            "Latest returned observation: "
            f"{latest.pedestrian_count} pedestrians at "
            f"{format_melbourne_timestamp(latest.observed_at)}."
        )

    st.subheader("Returned observations and forecast")
    st.caption("Observed values use a solid line. Forecast values use a dashed line.")
    st.plotly_chart(
        build_forecast_figure(history=chart_history, forecast=forecast),
        width="stretch",
    )

    if history is not None and history.data:
        _render_history_table(history)

    _render_forecast_details(forecast)


def _render_history(
    history: HistoryResponse | None,
    history_error: DashboardApiError | None,
) -> None:
    if history_error is not None:
        render_api_error(history_error, heading="Observations unavailable")
        return
    if history is None:
        return
    if not history.data:
        st.info("No observations were returned for the matching interval.")
        return

    latest = max(history.data, key=lambda point: point.observed_at)
    st.write(
        "Latest returned observation: "
        f"{latest.pedestrian_count} pedestrians at "
        f"{format_melbourne_timestamp(latest.observed_at)}."
    )
    st.subheader("Returned observations")
    st.caption("Observed values use a solid line.")
    st.plotly_chart(build_history_figure(history), width="stretch")
    _render_history_table(history)


def _render_history_table(history: HistoryResponse) -> None:
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


def _render_forecast_details(forecast: ForecastResponse) -> None:
    st.subheader("Returned forecast details")
    st.write(f"Model: {forecast.model_name}")
    st.write(f"Model version: {forecast.model_version or 'Not returned'}")
    st.write(f"Generated at: {format_melbourne_timestamp(forecast.generated_at)}")
    st.write(f"Data cutoff: {format_melbourne_timestamp(forecast.data_cutoff_at)}")
    if not forecast.predictions:
        return

    largest = max(forecast.predictions, key=lambda prediction: prediction.predicted_count)
    st.write(
        "Largest returned prediction: "
        f"{largest.predicted_count:g} pedestrians at "
        f"{format_melbourne_timestamp(largest.target_at)}."
    )


def _render_data_and_model_context(
    health: HealthResult,
    sensors: SensorListResponse | None,
    client: DashboardApiClient,
    *,
    allow_metrics: bool,
) -> None:
    with st.expander("Data and model context"):
        if health.status == "degraded":
            st.warning(
                "The API reported degraded configuration and availability. "
                "This status does not indicate that data or a model is ready."
            )
        st.write(f"API status: {health.status}")
        st.write(f"Health generated at: {format_melbourne_timestamp(health.generated_at)}")
        st.write(f"Model version: {health.model_version or 'Not returned'}")
        st.write(
            "Data cutoff: "
            + (
                format_melbourne_timestamp(health.data_cutoff_at)
                if health.data_cutoff_at is not None
                else "Not returned"
            )
        )
        st.write(f"API process: {health.components.api_process.status}")
        st.write(f"Model provider: {health.components.model_provider.status}")
        st.write(f"Data store: {health.components.data_store.status}")
        st.write(f"Data freshness component: {health.components.data_freshness.status}")
        if sensors is not None:
            st.write(f"Active sensor count: {sensors.meta.count}")

        if allow_metrics and st.button(
            "View historical model evaluation",
            key="view_historical_model_evaluation",
        ):
            _render_historical_metrics(client)


def _render_historical_metrics(client: DashboardApiClient) -> None:
    try:
        metrics = client.get_model_metrics()
    except DashboardApiError as error:
        render_api_error(error, heading="Historical evaluation unavailable")
        return

    _render_metrics(metrics)


def _render_metrics(response: ModelMetricsResponse) -> None:
    st.subheader("Historical evaluation context")
    st.caption(
        "These metrics describe the returned historical final-test evaluation, "
        "not current sensor or serving accuracy."
    )
    st.write(f"Evaluation model: {response.model_name}")
    st.write(f"Model version: {response.model_version or 'Not returned'}")
    st.write(
        "Final-test window: "
        f"{format_melbourne_timestamp(response.final_test_window.start)} to "
        f"{format_melbourne_timestamp(response.final_test_window.end)}"
    )
    first, second, third = st.columns(3)
    first.metric("Historical MAE", f"{response.metrics.mae:g}")
    second.metric("Historical RMSE", f"{response.metrics.rmse:g}")
    third.metric("Historical WAPE", f"{response.metrics.wape:.1%}")
