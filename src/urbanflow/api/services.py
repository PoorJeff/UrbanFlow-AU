from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from urbanflow import __version__
from urbanflow.api.schemas import ComponentHealth, HealthResult

API_SERVICE_NAME = "urbanflow-au-api"


class HealthService(Protocol):
    def __call__(self) -> HealthResult: ...


def default_health() -> HealthResult:
    return HealthResult(
        status="degraded",
        service=API_SERVICE_NAME,
        version=__version__,
        generated_at=datetime.now(UTC),
        components={
            "api_process": ComponentHealth(status="available"),
            "model_provider": ComponentHealth(status="unconfigured"),
            "data_store": ComponentHealth(status="unconfigured"),
            "data_freshness": ComponentHealth(status="unconfigured"),
        },
    )


@dataclass(frozen=True, slots=True)
class ApiServices:
    health: HealthService = default_health
