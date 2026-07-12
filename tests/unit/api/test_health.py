import asyncio

import httpx
from fastapi import FastAPI

from urbanflow.api.app import create_app
from urbanflow.api.schemas import HealthResult
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
    assert response.json() == {
        "status": "degraded",
        "components": {
            "api_process": {"status": "available"},
            "model_provider": {"status": "unconfigured"},
            "data_store": {"status": "unconfigured"},
            "data_freshness": {"status": "unconfigured"},
        },
    }


def test_health_returns_503_when_injected_result_is_unavailable() -> None:
    services = ApiServices(health=lambda: HealthResult(status="unavailable"))

    response = get(create_app(services=services), "/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "components": {}}
