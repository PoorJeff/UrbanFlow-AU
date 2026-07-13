from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import urbanflow
from tests.unit.api.helpers import api_get
from urbanflow.api.app import create_app
from urbanflow.api.schemas import ComponentHealth, HealthComponents, HealthResult
from urbanflow.api.services import ApiServices


def components(
    *,
    api_process: str = "available",
    model_provider: str = "unconfigured",
    data_store: str = "unconfigured",
    data_freshness: str = "unconfigured",
) -> HealthComponents:
    return HealthComponents(
        api_process=ComponentHealth(status=api_process),
        model_provider=ComponentHealth(status=model_provider),
        data_store=ComponentHealth(status=data_store),
        data_freshness=ComponentHealth(status=data_freshness),
    )


def test_default_health_is_degraded_when_optional_components_are_unconfigured() -> None:
    response = api_get(create_app(), "/health")

    assert response.status_code == 200
    payload = response.json()
    generated_at = datetime.fromisoformat(payload.pop("generated_at"))
    assert generated_at.tzinfo is not None
    assert generated_at.utcoffset() == UTC.utcoffset(generated_at)
    assert payload == {
        "status": "degraded",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "components": {
            "api_process": {"status": "available"},
            "model_provider": {"status": "unconfigured"},
            "data_store": {"status": "unconfigured"},
            "data_freshness": {"status": "unconfigured"},
        },
        "model_version": None,
        "data_cutoff_at": None,
    }


def test_health_returns_200_when_injected_result_is_ok() -> None:
    generated_at = datetime(2026, 7, 12, 10, 30, tzinfo=UTC)
    data_cutoff_at = datetime(2026, 7, 12, 9, 0, tzinfo=UTC)
    services = ApiServices(
        health=lambda: HealthResult(
            status="ok",
            service="urbanflow-au-api",
            version=urbanflow.__version__,
            generated_at=generated_at,
            components=components(
                api_process="available",
                model_provider="available",
                data_store="available",
                data_freshness="available",
            ),
            model_version="lightgbm-demo-v1",
            data_cutoff_at=data_cutoff_at,
        )
    )

    response = api_get(create_app(services=services), "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "generated_at": "2026-07-12T10:30:00Z",
        "components": {
            "api_process": {"status": "available"},
            "model_provider": {"status": "available"},
            "data_store": {"status": "available"},
            "data_freshness": {"status": "available"},
        },
        "model_version": "lightgbm-demo-v1",
        "data_cutoff_at": "2026-07-12T09:00:00Z",
    }


def test_health_returns_503_when_injected_result_is_unavailable() -> None:
    services = ApiServices(
        health=lambda: HealthResult(
            status="unavailable",
            service="urbanflow-au-api",
            version=urbanflow.__version__,
            generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=UTC),
            components=components(api_process="unavailable"),
        )
    )

    response = api_get(create_app(services=services), "/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "generated_at": "2026-07-12T10:30:00Z",
        "components": {
            "api_process": {"status": "unavailable"},
            "model_provider": {"status": "unconfigured"},
            "data_store": {"status": "unconfigured"},
            "data_freshness": {"status": "unconfigured"},
        },
        "model_version": None,
        "data_cutoff_at": None,
    }


def test_health_result_requires_every_component_status() -> None:
    with pytest.raises(ValidationError, match="components"):
        HealthResult(
            status="ok",
            service="urbanflow-au-api",
            version=urbanflow.__version__,
            generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=UTC),
        )
