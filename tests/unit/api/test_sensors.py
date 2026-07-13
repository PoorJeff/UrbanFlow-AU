import asyncio
from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI

from urbanflow.api.app import create_app
from urbanflow.api.services import (
    ApiServices,
    DataStoreUnavailableError,
    SensorRecord,
)


@dataclass
class InMemorySensorRepository:
    records: list[SensorRecord]
    fail_on_list: bool = False

    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        if self.fail_on_list:
            raise DataStoreUnavailableError("sensor catalog is unavailable")
        if active_only:
            return [record for record in self.records if record.status.casefold() == "active"]
        return list(self.records)

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        return next((record for record in self.records if record.location_id == location_id), None)


def get(application: FastAPI, path: str, **kwargs: object) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path, **kwargs)

    return asyncio.run(send_request())


@pytest.mark.parametrize("params", [{}, {"active_only": "true"}])
def test_sensor_catalog_filters_active_sensors_by_default_and_explicit_flag(
    params: dict[str, str],
) -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(
            records=[
                SensorRecord(
                    location_id=101,
                    sensor_name="Swanston Street",
                    sensor_description="Melbourne Central",
                    status="Active",
                    latitude=-37.8102,
                    longitude=144.9631,
                ),
                SensorRecord(
                    location_id=102,
                    sensor_name="Bourke Street",
                    sensor_description="Bourke Street Mall",
                    status="Inactive",
                    latitude=-37.8136,
                    longitude=144.9632,
                ),
            ]
        )
    )

    response = get(create_app(services=services), "/api/v1/sensors", params=params)

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "location_id": 101,
                "sensor_name": "Swanston Street",
                "sensor_description": "Melbourne Central",
                "status": "Active",
                "latitude": -37.8102,
                "longitude": 144.9631,
            }
        ],
        "meta": {"count": 1, "active_only": True},
    }


def test_sensor_catalog_can_include_inactive_sensors() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(
            records=[
                SensorRecord(
                    location_id=101,
                    sensor_name="Swanston Street",
                    sensor_description="Melbourne Central",
                    status="Active",
                    latitude=-37.8102,
                    longitude=144.9631,
                ),
                SensorRecord(
                    location_id=102,
                    sensor_name="Bourke Street",
                    sensor_description="Bourke Street Mall",
                    status="Inactive",
                    latitude=-37.8136,
                    longitude=144.9632,
                ),
            ]
        )
    )

    response = get(
        create_app(services=services),
        "/api/v1/sensors",
        params={"active_only": "false"},
    )

    assert response.status_code == 200
    assert response.json()["meta"] == {"count": 2, "active_only": False}
    assert [sensor["location_id"] for sensor in response.json()["data"]] == [101, 102]


def test_sensor_catalog_is_empty_without_a_configured_data_source() -> None:
    response = get(create_app(), "/api/v1/sensors")

    assert response.status_code == 200
    assert response.json() == {"data": [], "meta": {"count": 0, "active_only": True}}


def test_sensor_catalog_returns_a_project_error_for_data_store_failure() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[], fail_on_list=True)
    )

    response = get(create_app(services=services), "/api/v1/sensors")

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "data_store_unavailable",
        "message": "Sensor data is currently unavailable.",
        "details": [],
    }
