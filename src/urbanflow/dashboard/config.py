from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

DASHBOARD_API_BASE_URL_ENV_VAR = "URBANFLOW_DASHBOARD_API_BASE_URL"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class DashboardConfigError(ValueError):
    """Raised when the explicitly supplied dashboard origin is unusable."""


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    api_base_url: str


def load_dashboard_config(
    environ: Mapping[str, str] | None = None,
) -> DashboardConfig:
    values = os.environ if environ is None else environ
    if DASHBOARD_API_BASE_URL_ENV_VAR not in values:
        return DashboardConfig(api_base_url=DEFAULT_API_BASE_URL)

    api_base_url = values[DASHBOARD_API_BASE_URL_ENV_VAR].strip()
    try:
        parsed_url = urlsplit(api_base_url)
        _ = parsed_url.port
    except ValueError as exc:
        raise DashboardConfigError(
            f"{DASHBOARD_API_BASE_URL_ENV_VAR} must be a valid HTTP(S) origin."
        ) from exc

    if (
        not api_base_url
        or parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.netloc.endswith(":")
        or parsed_url.path.strip("/")
        or parsed_url.query
        or "?" in api_base_url
        or parsed_url.fragment
        or "#" in api_base_url
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise DashboardConfigError(
            f"{DASHBOARD_API_BASE_URL_ENV_VAR} must be a valid HTTP(S) origin."
        )

    return DashboardConfig(api_base_url=api_base_url.rstrip("/"))
