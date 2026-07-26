from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DashboardApiError(RuntimeError):
    status_code: int | None
    code: str
    message: str
    details: tuple[object, ...]
