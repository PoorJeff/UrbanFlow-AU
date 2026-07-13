import asyncio

import httpx
from fastapi import FastAPI

from urbanflow.api.app import create_app
from urbanflow.api.errors import UrbanFlowApiError


def get(application: FastAPI, path: str, **kwargs: object) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path, **kwargs)

    return asyncio.run(send_request())


def test_create_app_exposes_the_health_sensor_and_history_routes() -> None:
    application = create_app()

    assert isinstance(application, FastAPI)
    assert set(application.openapi()["paths"]) == {
        "/health",
        "/api/v1/sensors",
        "/api/v1/sensors/{location_id}/history",
    }


def test_project_errors_use_the_standard_error_response() -> None:
    application = create_app()

    @application.get("/_test/project-error")
    def raise_project_error() -> None:
        raise UrbanFlowApiError(
            status_code=503,
            code="model_unavailable",
            message="No model provider is configured.",
        )

    response = get(application, "/_test/project-error")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "model_unavailable",
            "message": "No model provider is configured.",
            "details": [],
        }
    }


def test_request_validation_keeps_fastapi_response_shape() -> None:
    application = create_app()

    @application.get("/_test/requires-integer")
    def requires_integer(value: int) -> dict[str, int]:
        return {"value": value}

    response = get(application, "/_test/requires-integer", params={"value": "bad"})

    assert response.status_code == 422
    assert "detail" in response.json()
    assert "error" not in response.json()
