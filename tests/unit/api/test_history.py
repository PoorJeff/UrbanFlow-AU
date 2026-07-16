from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from tests.unit.api.helpers import InMemorySensorRepository, api_get, make_sensor
from urbanflow.api.app import create_app
from urbanflow.api.services import (
    ApiServices,
    DataStoreUnavailableError,
    HistoryRecord,
    SensorRecord,
)


@dataclass
class InMemoryHistoryRepository:
    records: list[HistoryRecord]
    fail_on_read: bool = False

    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]:
        if self.fail_on_read:
            raise DataStoreUnavailableError("history is unavailable")
        return list(self.records)


class FailingLookupSensorRepository:
    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        return []

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        raise DataStoreUnavailableError("sensor lookup is unavailable")


def test_history_returns_rows_in_observed_time_order() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        history_repository=InMemoryHistoryRepository(
            records=[
                HistoryRecord(
                    observed_at=datetime(2026, 7, 3, 0, tzinfo=UTC),
                    pedestrian_count=99,
                ),
                HistoryRecord(
                    observed_at=datetime(2026, 7, 2, 1, tzinfo=UTC),
                    pedestrian_count=31,
                ),
                HistoryRecord(
                    observed_at=datetime(2026, 7, 2, 0, tzinfo=UTC),
                    pedestrian_count=24,
                ),
            ]
        ),
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": "2026-07-02T00:00:00Z", "end": "2026-07-03T00:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "location_id": 101,
        "start": "2026-07-02T00:00:00Z",
        "end": "2026-07-03T00:00:00Z",
        "data": [
            {"observed_at": "2026-07-02T00:00:00Z", "pedestrian_count": 24},
            {"observed_at": "2026-07-02T01:00:00Z", "pedestrian_count": 31},
        ],
    }


def test_history_allows_an_exactly_31_day_range() -> None:
    services = ApiServices(sensor_repository=InMemorySensorRepository(records=[make_sensor()]))

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json()["data"] == []


def test_history_rejects_timestamps_without_timezones() -> None:
    services = ApiServices(sensor_repository=InMemorySensorRepository(records=[make_sensor()]))

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": "2026-07-02T00:00:00", "end": "2026-07-03T00:00:00Z"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "history_range_invalid"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-07-03T00:00:00Z", "2026-07-02T00:00:00Z"),
        ("2026-07-01T00:00:00Z", "2026-08-01T00:00:01Z"),
    ],
)
def test_history_rejects_reversed_or_oversized_ranges(start: str, end: str) -> None:
    services = ApiServices(sensor_repository=InMemorySensorRepository(records=[make_sensor()]))

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": start, "end": end},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "history_range_invalid"


def test_history_rejects_non_positive_location_ids() -> None:
    response = api_get(
        create_app(),
        "/api/v1/sensors/0/history",
        params={"start": "2026-07-02T00:00:00Z", "end": "2026-07-03T00:00:00Z"},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_history_returns_sensor_not_found_before_reading_history() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[]),
        history_repository=InMemoryHistoryRepository(records=[]),
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": "2026-07-02T00:00:00Z", "end": "2026-07-03T00:00:00Z"},
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "sensor_not_found",
        "message": "Sensor 101 was not found.",
        "details": [],
    }


def test_history_returns_a_project_error_for_sensor_lookup_data_store_failure() -> None:
    services = ApiServices(
        sensor_repository=FailingLookupSensorRepository(),
        history_repository=InMemoryHistoryRepository(records=[]),
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": "2026-07-02T00:00:00Z", "end": "2026-07-03T00:00:00Z"},
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "data_store_unavailable",
        "message": "Sensor data is currently unavailable.",
        "details": [],
    }


def test_history_returns_a_project_error_for_data_store_failure() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        history_repository=InMemoryHistoryRepository(records=[], fail_on_read=True),
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/history",
        params={"start": "2026-07-02T00:00:00Z", "end": "2026-07-03T00:00:00Z"},
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "data_store_unavailable",
        "message": "Sensor data is currently unavailable.",
        "details": [],
    }
