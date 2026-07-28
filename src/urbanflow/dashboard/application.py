import streamlit as st

from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.config import (
    DashboardConfig,
    DashboardConfigError,
    load_dashboard_config,
)
from urbanflow.dashboard.pages.explore import render_explore
from urbanflow.dashboard.pages.forecast import render_forecast
from urbanflow.dashboard.pages.today import DASHBOARD_PAGE_KEY, render_today

DASHBOARD_PAGES = ("Today", "Explore", "Forecast")


def create_dashboard_client(config: DashboardConfig) -> DashboardApiClient:
    return DashboardApiClient(config.api_base_url)


def render_dashboard(client: DashboardApiClient) -> None:
    page = st.radio(
        "Page",
        DASHBOARD_PAGES,
        horizontal=True,
        key=DASHBOARD_PAGE_KEY,
    )
    if page == "Explore":
        render_explore(client)
    elif page == "Forecast":
        render_forecast(client)
    else:
        render_today(client)


def main() -> None:
    st.set_page_config(page_title="UrbanFlow AU", page_icon="🚶", layout="wide")
    try:
        config = load_dashboard_config()
    except DashboardConfigError as error:
        st.title("UrbanFlow AU")
        st.error(f"Dashboard configuration error: {error}")
        return

    client = create_dashboard_client(config)
    try:
        render_dashboard(client)
    finally:
        client.close()
