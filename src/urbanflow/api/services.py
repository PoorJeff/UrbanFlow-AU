from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from urbanflow.api.schemas import ComponentHealth, HealthResult


class HealthService(Protocol):
    def __call__(self) -> HealthResult: ...


def default_health() -> HealthResult:
    return HealthResult(
        status="degraded",
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
