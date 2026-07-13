from fastapi import APIRouter

from urbanflow.api.routers.forecasts import router as forecasts_router
from urbanflow.api.routers.health import router as health_router
from urbanflow.api.routers.sensors import router as sensors_router

router = APIRouter()
router.include_router(health_router)
router.include_router(sensors_router)
router.include_router(forecasts_router)

__all__ = ["router"]
