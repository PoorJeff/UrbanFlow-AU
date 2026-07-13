from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from urbanflow.api.dependencies import get_services
from urbanflow.api.schemas import ForecastPredictionResponse, ForecastResponse
from urbanflow.api.services import ApiServices

router = APIRouter(prefix="/api/v1/sensors", tags=["forecasts"])


@router.get("/{location_id}/forecast", response_model=ForecastResponse)
def get_sensor_forecast(
    location_id: Annotated[int, Path(ge=1)],
    services: Annotated[ApiServices, Depends(get_services)],
    horizon: Annotated[int, Query(ge=1, le=24)] = 24,
) -> ForecastResponse:
    batch = services.forecast_service.forecast(location_id=location_id, horizon=horizon)
    return ForecastResponse(
        location_id=location_id,
        model_name=batch.model_name,
        model_version=batch.model_version,
        generated_at=batch.generated_at,
        forecast_origin_at=batch.forecast_origin_at,
        data_cutoff_at=batch.data_cutoff_at,
        horizon_hours=horizon,
        predictions=[
            ForecastPredictionResponse(
                forecast_horizon=prediction.forecast_horizon,
                target_at=prediction.target_at,
                predicted_count=prediction.predicted_count,
            )
            for prediction in batch.predictions
        ],
    )
