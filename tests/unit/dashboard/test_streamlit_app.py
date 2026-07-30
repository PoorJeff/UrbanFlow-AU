from __future__ import annotations

import runpy
from pathlib import Path

from streamlit.testing.v1 import AppTest

from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.dashboard.config import DashboardConfig

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "app" / "streamlit_app.py"


def _visible_text(at: AppTest) -> str:
    values: list[str] = []
    for element_type in ("title", "header", "markdown", "caption", "info", "warning", "error"):
        for element in at.get(element_type):
            value = getattr(element, "value", None)
            if value is not None:
                values.append(str(value))
    return "\n".join(values)


def test_importing_entrypoint_does_not_create_an_http_client(monkeypatch) -> None:
    def fail_if_created(*args: object, **kwargs: object) -> None:
        raise AssertionError("Import created an HTTP client.")

    monkeypatch.setattr(DashboardApiClient, "__init__", fail_if_created)

    runpy.run_path(str(ENTRYPOINT), run_name="urbanflow_test_import")


def test_real_entrypoint_renders_invalid_configuration_without_http(
    monkeypatch,
) -> None:
    monkeypatch.setenv("URBANFLOW_DASHBOARD_API_BASE_URL", "not-an-http-origin")

    at = AppTest.from_file(ENTRYPOINT).run()

    assert not at.exception
    text = _visible_text(at)
    assert "Dashboard configuration error" in text
    assert "URBANFLOW_DASHBOARD_API_BASE_URL must be a valid HTTP(S) origin." in text


def test_main_creates_one_client_and_closes_it_after_render(monkeypatch) -> None:
    from urbanflow.dashboard import application

    calls: list[object] = []

    class Client:
        def close(self) -> None:
            calls.append("close")

    client = Client()
    config = DashboardConfig(api_base_url="https://dashboard.example")

    monkeypatch.setattr(application.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(application, "load_dashboard_config", lambda: config)

    def create(received: DashboardConfig) -> Client:
        calls.append(("create", received))
        return client

    monkeypatch.setattr(application, "create_dashboard_client", create)
    monkeypatch.setattr(
        application,
        "render_dashboard",
        lambda received, *, api_origin: calls.append(("render", received, api_origin)),
    )

    application.main()

    assert calls == [
        ("create", config),
        ("render", client, "https://dashboard.example"),
        "close",
    ]


def test_main_closes_client_when_render_raises(monkeypatch) -> None:
    from urbanflow.dashboard import application

    closed: list[bool] = []

    class Client:
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(application.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(
        application,
        "load_dashboard_config",
        lambda: DashboardConfig(api_base_url="https://dashboard.example"),
    )
    monkeypatch.setattr(application, "create_dashboard_client", lambda config: Client())

    def fail_render(client: Client, *, api_origin: str) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr(application, "render_dashboard", fail_render)

    try:
        application.main()
    except RuntimeError as exc:
        assert str(exc) == "render failed"
    else:
        raise AssertionError("Expected render failure.")

    assert closed == [True]
