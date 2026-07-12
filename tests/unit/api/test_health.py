import asyncio
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI

import urbanflow
from urbanflow.api.app import create_app
from urbanflow.api.schemas import ComponentHealth, HealthResult
from urbanflow.api.services import ApiServices


def get(application: FastAPI, path: str) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(send_request())


def test_default_health_is_degraded_when_optional_components_are_unconfigured() -> None:
    response = get(create_app(), "/health")

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
            components={"api_process": ComponentHealth(status="available")},
            model_version="lightgbm-demo-v1",
            data_cutoff_at=data_cutoff_at,
        )
    )

    response = get(create_app(services=services), "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "generated_at": "2026-07-12T10:30:00Z",
        "components": {"api_process": {"status": "available"}},
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
        )
    )

    response = get(create_app(services=services), "/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "generated_at": "2026-07-12T10:30:00Z",
        "components": {},
        "model_version": None,
        "data_cutoff_at": None,
    }
