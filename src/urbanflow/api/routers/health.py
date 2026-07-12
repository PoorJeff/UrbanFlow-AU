from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from urbanflow.api.dependencies import get_services
from urbanflow.api.schemas import HealthResult
from urbanflow.api.services import ApiServices

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResult,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResult}},
)
def get_health(
    response: Response,
    services: Annotated[ApiServices, Depends(get_services)],
) -> HealthResult:
    result = services.health()
    if result.status == "unavailable":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
