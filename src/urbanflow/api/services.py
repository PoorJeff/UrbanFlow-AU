from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from urbanflow import __version__
from urbanflow.api.errors import UrbanFlowApiError, data_store_unavailable_error
from urbanflow.api.schemas import ComponentHealth, HealthResult

API_SERVICE_NAME = "urbanflow-au-api"
MAX_HISTORY_RANGE = timedelta(days=31)


class HealthService(Protocol):
    def __call__(self) -> HealthResult: ...


@dataclass(frozen=True, slots=True)
class SensorRecord:
    location_id: int
    sensor_name: str
    sensor_description: str
    status: str
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    observed_at: datetime
    pedestrian_count: int


class SensorRepository(Protocol):
    def list_sensors(self, active_only: bool) -> list[SensorRecord]: ...

    def get_sensor(self, location_id: int) -> SensorRecord | None: ...


class HistoryRepository(Protocol):
    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]: ...


class DataStoreUnavailableError(RuntimeError):
    """Raised by a configured API repository when its backing store cannot be read."""


class EmptySensorRepository:
    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        return []

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        return None


class EmptyHistoryRepository:
    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]:
        return []


@dataclass(frozen=True, slots=True)
class HistoryService:
    sensor_repository: SensorRepository
    history_repository: HistoryRepository

    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]:
        _validate_history_range(start=start, end=end)
        try:
            sensor = self.sensor_repository.get_sensor(location_id)
            if sensor is None:
                raise UrbanFlowApiError(
                    status_code=404,
                    code="sensor_not_found",
                    message=f"Sensor {location_id} was not found.",
                )
            records = self.history_repository.get_history(location_id, start, end)
        except DataStoreUnavailableError as exc:
            raise data_store_unavailable_error() from exc
        return sorted(
            (record for record in records if start <= record.observed_at < end),
            key=lambda record: record.observed_at,
        )


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
    sensor_repository: SensorRepository = field(default_factory=EmptySensorRepository)
    history_repository: HistoryRepository = field(default_factory=EmptyHistoryRepository)

    @property
    def history_service(self) -> HistoryService:
        return HistoryService(
            sensor_repository=self.sensor_repository,
            history_repository=self.history_repository,
        )


def _validate_history_range(*, start: datetime, end: datetime) -> None:
    if not _is_timezone_aware(start) or not _is_timezone_aware(end):
        raise UrbanFlowApiError(
            status_code=422,
            code="history_range_invalid",
            message="History start and end timestamps must include timezones.",
        )
    if start >= end:
        raise UrbanFlowApiError(
            status_code=422,
            code="history_range_invalid",
            message="History start must be earlier than end.",
        )
    if end - start > MAX_HISTORY_RANGE:
        raise UrbanFlowApiError(
            status_code=422,
            code="history_range_invalid",
            message="History range must not exceed 31 days.",
        )


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None
