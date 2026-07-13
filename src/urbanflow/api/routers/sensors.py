from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from urbanflow.api.dependencies import get_services
from urbanflow.api.errors import data_store_unavailable_error
from urbanflow.api.schemas import (
    HistoryPoint,
    HistoryResponse,
    SensorListMeta,
    SensorListResponse,
    SensorResponse,
)
from urbanflow.api.services import ApiServices, DataStoreUnavailableError

router = APIRouter(prefix="/api/v1/sensors", tags=["sensors"])


@router.get("", response_model=SensorListResponse)
def list_sensors(
    services: Annotated[ApiServices, Depends(get_services)],
    active_only: Annotated[bool, Query()] = True,
) -> SensorListResponse:
    try:
        records = services.sensor_repository.list_sensors(active_only)
    except DataStoreUnavailableError as exc:
        raise data_store_unavailable_error() from exc
    data = [
        SensorResponse(
            location_id=record.location_id,
            sensor_name=record.sensor_name,
            sensor_description=record.sensor_description,
            status=record.status,
            latitude=record.latitude,
            longitude=record.longitude,
        )
        for record in records
    ]
    return SensorListResponse(
        data=data,
        meta=SensorListMeta(count=len(data), active_only=active_only),
    )


@router.get("/{location_id}/history", response_model=HistoryResponse)
def get_sensor_history(
    location_id: Annotated[int, Path(ge=1)],
    start: datetime,
    end: datetime,
    services: Annotated[ApiServices, Depends(get_services)],
) -> HistoryResponse:
    records = services.history_service.get_history(
        location_id=location_id,
        start=start,
        end=end,
    )
    return HistoryResponse(
        location_id=location_id,
        start=start,
        end=end,
        data=[
            HistoryPoint(
                observed_at=record.observed_at,
                pedestrian_count=record.pedestrian_count,
            )
            for record in records
        ],
    )
