from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from urbanflow.api.schemas import ErrorBody, ErrorResponse


class UrbanFlowApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: list[object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = [] if details is None else list(details)


def data_store_unavailable_error() -> UrbanFlowApiError:
    return UrbanFlowApiError(
        status_code=503,
        code="data_store_unavailable",
        message="Sensor data is currently unavailable.",
    )


async def urbanflow_api_error_handler(
    _request: Request,
    error: UrbanFlowApiError,
) -> JSONResponse:
    response = ErrorResponse(
        error=ErrorBody(
            code=error.code,
            message=error.message,
            details=error.details,
        )
    )
    return JSONResponse(status_code=error.status_code, content=response.model_dump(mode="json"))
