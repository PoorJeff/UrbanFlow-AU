from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded", "unavailable"]
ComponentStatus = Literal["available", "unconfigured", "unavailable"]


class ComponentHealth(BaseModel):
    status: ComponentStatus


class HealthResult(BaseModel):
    status: HealthStatus
    service: str
    version: str
    generated_at: datetime
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    model_version: str | None = None
    data_cutoff_at: datetime | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[object] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
