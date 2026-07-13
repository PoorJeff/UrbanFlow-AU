import pytest

from tests.unit.api.helpers import InMemorySensorRepository, api_get
from urbanflow.api.app import create_app
from urbanflow.api.services import ApiServices, SensorRecord


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

    response = api_get(create_app(services=services), "/api/v1/sensors", params=params)

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

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors",
        params={"active_only": "false"},
    )

    assert response.status_code == 200
    assert response.json()["meta"] == {"count": 2, "active_only": False}
    assert [sensor["location_id"] for sensor in response.json()["data"]] == [101, 102]


def test_sensor_catalog_is_empty_without_a_configured_data_source() -> None:
    response = api_get(create_app(), "/api/v1/sensors")

    assert response.status_code == 200
    assert response.json() == {"data": [], "meta": {"count": 0, "active_only": True}}


def test_sensor_catalog_returns_a_project_error_for_data_store_failure() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[], fail_on_list=True)
    )

    response = api_get(create_app(services=services), "/api/v1/sensors")

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "data_store_unavailable",
        "message": "Sensor data is currently unavailable.",
        "details": [],
    }
