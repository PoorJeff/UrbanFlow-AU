from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tests.unit.api.helpers import InMemorySensorRepository, api_get, make_sensor
from urbanflow.api.app import create_app
from urbanflow.api.services import (
    ApiServices,
    DataStoreUnavailableError,
    ForecastBatch,
    ForecastInputUnavailableError,
    ForecastPrediction,
    SensorRecord,
)


class FailingSensorRepository:
    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        raise AssertionError("forecast must not access sensors without a model provider")

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        raise AssertionError("forecast must not access sensors without a model provider")


@dataclass
class RecordingForecastProvider:
    batch: ForecastBatch
    calls: list[tuple[int, int]] = field(default_factory=list)

    def predict(self, location_id: int, horizon: int) -> ForecastBatch:
        self.calls.append((location_id, horizon))
        return self.batch


class UnavailableHistoryProvider:
    def predict(self, location_id: int, horizon: int) -> ForecastBatch:
        raise DataStoreUnavailableError("database unavailable")


class InvalidServingInputProvider:
    def predict(self, location_id: int, horizon: int) -> ForecastBatch:
        raise ForecastInputUnavailableError("missing contiguous history")


def forecast_batch(horizon: int) -> ForecastBatch:
    generated_at = datetime(2026, 7, 12, 10, 30, tzinfo=UTC)
    forecast_origin_at = datetime(2026, 7, 12, 10, 0, tzinfo=UTC)
    return ForecastBatch(
        model_name="lightgbm",
        model_version="lightgbm-demo-v1",
        generated_at=generated_at,
        forecast_origin_at=forecast_origin_at,
        data_cutoff_at=forecast_origin_at,
        predictions=tuple(
            ForecastPrediction(
                forecast_horizon=step,
                target_at=forecast_origin_at + timedelta(hours=step),
                predicted_count=float(step),
            )
            for step in range(1, horizon + 1)
        ),
    )


def test_forecast_defaults_to_24_direct_horizons() -> None:
    provider = RecordingForecastProvider(batch=forecast_batch(24))
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=provider,
    )

    response = api_get(create_app(services=services), "/api/v1/sensors/101/forecast")

    assert response.status_code == 200
    assert response.json()["horizon_hours"] == 24
    assert [row["forecast_horizon"] for row in response.json()["predictions"]] == list(range(1, 25))
    assert provider.calls == [(101, 24)]


@pytest.mark.parametrize("horizon", [1, 24])
def test_forecast_accepts_boundary_horizons(horizon: int) -> None:
    provider = RecordingForecastProvider(batch=forecast_batch(horizon))
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=provider,
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/forecast",
        params={"horizon": str(horizon)},
    )

    assert response.status_code == 200
    assert response.json()["horizon_hours"] == horizon
    assert provider.calls == [(101, horizon)]


@pytest.mark.parametrize("horizon", ["0", "25", "not-an-integer"])
def test_forecast_rejects_invalid_horizons(horizon: str) -> None:
    response = api_get(
        create_app(),
        "/api/v1/sensors/101/forecast",
        params={"horizon": horizon},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_forecast_requires_a_provider_before_accessing_sensor_data() -> None:
    services = ApiServices(sensor_repository=FailingSensorRepository())

    response = api_get(create_app(services=services), "/api/v1/sensors/999/forecast")

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "model_unavailable",
        "message": "No forecast model is configured for serving.",
        "details": [],
    }


def test_forecast_returns_sensor_not_found_when_a_provider_is_configured() -> None:
    provider = RecordingForecastProvider(batch=forecast_batch(1))
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[]),
        model_provider=provider,
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/999/forecast",
        params={"horizon": "1"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "sensor_not_found"
    assert provider.calls == []


def test_forecast_maps_provider_data_store_failure() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=UnavailableHistoryProvider(),
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/forecast",
        params={"horizon": "1"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "data_store_unavailable",
            "message": "Sensor data is currently unavailable.",
            "details": [],
        }
    }


def test_forecast_maps_invalid_serving_inputs() -> None:
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=InvalidServingInputProvider(),
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/forecast",
        params={"horizon": "1"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "forecast_unavailable",
            "message": "Forecast cannot be generated from the available serving inputs.",
            "details": [],
        }
    }


def test_forecast_orders_rows_clips_counts_and_preserves_provider_metadata() -> None:
    melbourne_offset = timezone(timedelta(hours=10))
    forecast_origin_at = datetime(2026, 7, 12, 10, 0, tzinfo=melbourne_offset)
    provider = RecordingForecastProvider(
        batch=ForecastBatch(
            model_name="ridge",
            model_version="ridge-demo-v1",
            generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=melbourne_offset),
            forecast_origin_at=forecast_origin_at,
            data_cutoff_at=datetime(2026, 7, 12, 9, 0, tzinfo=melbourne_offset),
            predictions=(
                ForecastPrediction(
                    forecast_horizon=2,
                    target_at=forecast_origin_at + timedelta(hours=2),
                    predicted_count=-3.5,
                ),
                ForecastPrediction(
                    forecast_horizon=1,
                    target_at=forecast_origin_at + timedelta(hours=1),
                    predicted_count=12.5,
                ),
            ),
        )
    )
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=provider,
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/forecast",
        params={"horizon": "2"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "location_id": 101,
        "model_name": "ridge",
        "model_version": "ridge-demo-v1",
        "generated_at": "2026-07-12T10:30:00+10:00",
        "forecast_origin_at": "2026-07-12T10:00:00+10:00",
        "data_cutoff_at": "2026-07-12T09:00:00+10:00",
        "horizon_hours": 2,
        "predictions": [
            {
                "forecast_horizon": 1,
                "target_at": "2026-07-12T11:00:00+10:00",
                "predicted_count": 12.5,
            },
            {
                "forecast_horizon": 2,
                "target_at": "2026-07-12T12:00:00+10:00",
                "predicted_count": 0.0,
            },
        ],
    }
    assert provider.calls == [(101, 2)]


@pytest.mark.parametrize(
    "predicted_count",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_forecast_rejects_non_finite_provider_predictions(predicted_count: float) -> None:
    batch = forecast_batch(1)
    provider = RecordingForecastProvider(
        batch=replace(
            batch,
            predictions=(replace(batch.predictions[0], predicted_count=predicted_count),),
        )
    )
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=provider,
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/forecast",
        params={"horizon": "1"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


@pytest.mark.parametrize("returned_horizons", [(1,), (1, 1)])
def test_forecast_rejects_incomplete_or_duplicate_provider_horizons(
    returned_horizons: tuple[int, ...],
) -> None:
    forecast_origin_at = datetime(2026, 7, 12, 10, 0, tzinfo=UTC)
    provider = RecordingForecastProvider(
        batch=ForecastBatch(
            model_name="ridge",
            model_version=None,
            generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=UTC),
            forecast_origin_at=forecast_origin_at,
            data_cutoff_at=forecast_origin_at,
            predictions=tuple(
                ForecastPrediction(
                    forecast_horizon=step,
                    target_at=forecast_origin_at + timedelta(hours=step),
                    predicted_count=10.0,
                )
                for step in returned_horizons
            ),
        )
    )
    services = ApiServices(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=provider,
    )

    response = api_get(
        create_app(services=services),
        "/api/v1/sensors/101/forecast",
        params={"horizon": "2"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
