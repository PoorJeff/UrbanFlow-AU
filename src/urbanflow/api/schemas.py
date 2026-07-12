from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["healthy", "degraded", "unavailable"]
ComponentStatus = Literal["available", "unconfigured", "unavailable"]


class ComponentHealth(BaseModel):
    status: ComponentStatus


class HealthResult(BaseModel):
    status: HealthStatus
    components: dict[str, ComponentHealth] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[object] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
