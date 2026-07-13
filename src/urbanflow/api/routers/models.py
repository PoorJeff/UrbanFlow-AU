from typing import Annotated

from fastapi import APIRouter, Depends, status

from urbanflow.api.dependencies import get_services
from urbanflow.api.errors import metrics_unavailable_error
from urbanflow.api.schemas import ModelMetricsResponse
from urbanflow.api.services import ApiServices, MetricsUnavailableError

router = APIRouter(prefix="/api/v1/model", tags=["model"])


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Metrics unavailable"}},
)
def get_model_metrics(
    services: Annotated[ApiServices, Depends(get_services)],
) -> ModelMetricsResponse:
    try:
        return services.model_metadata_provider.get_metrics()
    except MetricsUnavailableError as exc:
        raise metrics_unavailable_error() from exc
