from __future__ import annotations

import os
from collections.abc import Mapping

DATABASE_URL_ENV_VAR = "URBANFLOW_DATABASE_URL"


class DatabaseConfigError(Exception):
    """Raised when database configuration is missing or unusable."""


def get_database_url(
    database_url: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environ is None else environ
    resolved = database_url if database_url is not None else values.get(DATABASE_URL_ENV_VAR)
    if resolved is None or not resolved.strip():
        raise DatabaseConfigError(
            f"Database URL is required. Pass --database-url or set {DATABASE_URL_ENV_VAR}."
        )
    return resolved.strip()
