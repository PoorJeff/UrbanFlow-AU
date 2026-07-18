import os
from collections.abc import Mapping

from fastapi import FastAPI
from sqlalchemy.exc import ArgumentError

from urbanflow import __version__
from urbanflow.api.errors import UrbanFlowApiError, urbanflow_api_error_handler
from urbanflow.api.lightgbm_provider import ArtifactBackedLightGBMForecastProvider
from urbanflow.api.postgres import PostgresSensorHistoryRepository
from urbanflow.api.routers import router as api_router
from urbanflow.api.services import ApiServices, ForecastModelProvider
from urbanflow.database.config import DATABASE_URL_ENV_VAR, DatabaseConfigError
from urbanflow.database.engine import create_database_engine, create_session_factory
from urbanflow.modeling.lightgbm_artifact import LightGBMArtifactError, load_lightgbm_artifact

MODEL_ARTIFACT_PATH_ENV_VAR = "URBANFLOW_API_MODEL_ARTIFACT_PATH"


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
    configured_artifact_path = values.get(MODEL_ARTIFACT_PATH_ENV_VAR)
    model_provider: ForecastModelProvider | None = None
    if configured_artifact_path is not None and configured_artifact_path.strip():
        try:
            artifact = load_lightgbm_artifact(configured_artifact_path.strip())
        except LightGBMArtifactError:
            model_provider = None
        else:
            model_provider = ArtifactBackedLightGBMForecastProvider(
                artifact=artifact,
                history_repository=repository,
            )
    return ApiServices(
        sensor_repository=repository,
        history_repository=repository,
        model_provider=model_provider,
    )


def create_app(*, services: ApiServices | None = None) -> FastAPI:
    application = FastAPI(title="UrbanFlow AU API", version=__version__)
    application.state.services = services if services is not None else create_default_services()
    application.add_exception_handler(UrbanFlowApiError, urbanflow_api_error_handler)
    application.include_router(api_router)
    return application


app = create_app()
