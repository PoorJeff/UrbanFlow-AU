from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def melbourne_timezone() -> timezone:
    return timezone(timedelta(hours=10))


@pytest.fixture
def history_start(melbourne_timezone: timezone) -> datetime:
    return datetime(2026, 7, 12, 8, 0, tzinfo=melbourne_timezone)


@pytest.fixture
def history_end(melbourne_timezone: timezone) -> datetime:
    return datetime(2026, 7, 12, 10, 0, tzinfo=melbourne_timezone)


@pytest.fixture
def health_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "urbanflow-au-api",
        "version": "0.1.0",
        "generated_at": "2026-07-12T10:30:00+10:00",
        "components": {
            "api_process": {"status": "available"},
            "model_provider": {"status": "available"},
            "data_store": {"status": "available"},
            "data_freshness": {"status": "available"},
        },
        "model_version": "lightgbm-demo-v1",
        "data_cutoff_at": "2026-07-12T10:00:00+10:00",
    }


@pytest.fixture
def sensor_list_payload() -> dict[str, object]:
    return {
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


@pytest.fixture
def history_payload() -> dict[str, object]:
    return {
        "location_id": 101,
        "start": "2026-07-12T08:00:00+10:00",
        "end": "2026-07-12T10:00:00+10:00",
        "data": [
            {
                "observed_at": "2026-07-12T08:00:00+10:00",
                "pedestrian_count": 24,
            },
            {
                "observed_at": "2026-07-12T09:00:00+10:00",
                "pedestrian_count": 31,
            },
        ],
    }


@pytest.fixture
def forecast_payload() -> dict[str, object]:
    return {
        "location_id": 101,
        "model_name": "lightgbm",
        "model_version": "lightgbm-demo-v1",
        "generated_at": "2026-07-12T10:30:00+10:00",
        "forecast_origin_at": "2026-07-12T10:00:00+10:00",
        "data_cutoff_at": "2026-07-12T10:00:00+10:00",
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
                "predicted_count": 15.0,
            },
        ],
    }


@pytest.fixture
def model_metrics_payload() -> dict[str, object]:
    return {
        "model_name": "lightgbm",
        "model_version": "lightgbm-demo-v1",
        "evaluation_source": "evaluation_summary",
        "final_test_window": {
            "name": "final_test_2025-02",
            "start": "2025-02-01T00:00:00+11:00",
            "end": "2025-03-01T00:00:00+11:00",
        },
        "metrics": {
            "mae": 1.2,
            "rmse": 1.7,
            "wape": 0.07,
            "seasonal_naive_wape": 0.095,
            "relative_wape_improvement": 0.2631578947368421,
        },
        "mlflow_run_id": "run-123",
        "mlflow_tracking_uri": "file:///tmp/mlruns",
        "report_path": "reports/lightgbm.md",
    }
