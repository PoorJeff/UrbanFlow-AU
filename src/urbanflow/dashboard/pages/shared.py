from dataclasses import dataclass

import streamlit as st

from urbanflow.api.schemas import HealthResult, SensorListResponse
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.errors import DashboardApiError


@dataclass(frozen=True, slots=True)
class PageContext:
    health: HealthResult | None
    sensors: SensorListResponse | None
    error: DashboardApiError | None


def load_page_context(client: DashboardApiClient) -> PageContext:
    try:
        health = client.get_health()
    except DashboardApiError as error:
        return PageContext(health=None, sensors=None, error=error)

    if health.status == "unavailable":
        return PageContext(health=health, sensors=None, error=None)

    try:
        sensors = client.list_sensors(active_only=True)
    except DashboardApiError as error:
        return PageContext(health=health, sensors=None, error=error)

    return PageContext(health=health, sensors=sensors, error=None)


def render_api_error(error: DashboardApiError, *, heading: str) -> None:
    st.error(f"{heading}: {error.message} (error code: {error.code})")
