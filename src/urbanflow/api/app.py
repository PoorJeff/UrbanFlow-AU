from fastapi import FastAPI

from urbanflow import __version__
from urbanflow.api.errors import UrbanFlowApiError, urbanflow_api_error_handler
from urbanflow.api.routers import router as api_router
from urbanflow.api.services import ApiServices


def create_app(*, services: ApiServices | None = None) -> FastAPI:
    application = FastAPI(title="UrbanFlow AU API", version=__version__)
    application.state.services = services if services is not None else ApiServices()
    application.add_exception_handler(UrbanFlowApiError, urbanflow_api_error_handler)
    application.include_router(api_router)
    return application


app = create_app()
