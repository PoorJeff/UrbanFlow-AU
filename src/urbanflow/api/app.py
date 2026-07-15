import os
from collections.abc import Mapping

from fastapi import FastAPI
from sqlalchemy.exc import ArgumentError

from urbanflow import __version__
from urbanflow.api.errors import UrbanFlowApiError, urbanflow_api_error_handler
from urbanflow.api.postgres import PostgresSensorHistoryRepository
from urbanflow.api.routers import router as api_router
from urbanflow.api.services import ApiServices
from urbanflow.database.config import DATABASE_URL_ENV_VAR, DatabaseConfigError
from urbanflow.database.engine import create_database_engine, create_session_factory


def create_default_services(
    *,
    environ: Mapping[str, str] | None = None,
) -> ApiServices:
    values = os.environ if environ is None else environ
    configured_url = values.get(DATABASE_URL_ENV_VAR)
    if configured_url is None or not configured_url.strip():
        return ApiServices()
    try:
        engine = create_database_engine(configured_url.strip())
    except ArgumentError as exc:
        raise DatabaseConfigError(f"Invalid {DATABASE_URL_ENV_VAR} configuration.") from exc
    session_factory = create_session_factory(engine)
    repository = PostgresSensorHistoryRepository(session_factory)
    return ApiServices(
        sensor_repository=repository,
        history_repository=repository,
    )


def create_app(*, services: ApiServices | None = None) -> FastAPI:
    application = FastAPI(title="UrbanFlow AU API", version=__version__)
    application.state.services = services if services is not None else create_default_services()
    application.add_exception_handler(UrbanFlowApiError, urbanflow_api_error_handler)
    application.include_router(api_router)
    return application


app = create_app()
