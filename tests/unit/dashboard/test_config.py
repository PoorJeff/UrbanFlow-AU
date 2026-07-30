import pytest

from urbanflow.dashboard.config import (
    DEFAULT_API_BASE_URL,
    DashboardConfigError,
    load_dashboard_config,
)


def test_absent_api_base_url_uses_default() -> None:
    config = load_dashboard_config({})

    assert config.api_base_url == DEFAULT_API_BASE_URL


def test_explicit_api_base_url_removes_trailing_slashes() -> None:
    config = load_dashboard_config(
        {"URBANFLOW_DASHBOARD_API_BASE_URL": "https://dashboard-api.example.test///"}
    )

    assert config.api_base_url == "https://dashboard-api.example.test"


@pytest.mark.parametrize(
    "api_base_url",
    [
        "",
        "   ",
        "ftp://dashboard-api.example.test",
        "http:///path",
        "http://",
    ],
    ids=["empty", "whitespace", "ftp", "missing-host-with-path", "missing-host"],
)
def test_explicit_unusable_api_base_url_is_rejected(api_base_url: str) -> None:
    with pytest.raises(DashboardConfigError):
        load_dashboard_config({"URBANFLOW_DASHBOARD_API_BASE_URL": api_base_url})


@pytest.mark.parametrize(
    "api_base_url",
    [
        "https://dashboard-api.example.test/prefix",
        "https://dashboard-api.example.test?environment=test",
        "https://dashboard-api.example.test#health",
        "https://operator:secret@dashboard-api.example.test",
        "https://dashboard-api.example.test:",
        "https://dashboard-api.example.test:not-a-port",
        "https://dashboard-api.example.test:65536",
    ],
    ids=[
        "path",
        "query",
        "fragment",
        "userinfo",
        "empty-port",
        "non-numeric-port",
        "out-of-range-port",
    ],
)
def test_explicit_api_base_url_rejects_non_origin_components(api_base_url: str) -> None:
    with pytest.raises(DashboardConfigError):
        load_dashboard_config({"URBANFLOW_DASHBOARD_API_BASE_URL": api_base_url})
