import streamlit as st

from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.config import (
    DashboardConfig,
    DashboardConfigError,
    load_dashboard_config,
)
from urbanflow.dashboard.pages.today import render_today


def create_dashboard_client(config: DashboardConfig) -> DashboardApiClient:
    return DashboardApiClient(config.api_base_url)


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
        render_today(client)
    finally:
        client.close()
