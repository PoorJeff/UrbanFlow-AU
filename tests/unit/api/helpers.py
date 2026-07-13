import asyncio
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from urbanflow.api.services import DataStoreUnavailableError, SensorRecord


def api_get(application: FastAPI, path: str, **kwargs: object) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path, **kwargs)

    return asyncio.run(send_request())


@dataclass
class InMemorySensorRepository:
    records: list[SensorRecord]
    fail_on_list: bool = False

    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        if self.fail_on_list:
            raise DataStoreUnavailableError("sensor catalog is unavailable")
        if active_only:
            return [record for record in self.records if record.status.casefold() == "active"]
        return list(self.records)

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        return next((record for record in self.records if record.location_id == location_id), None)


def make_sensor() -> SensorRecord:
    return SensorRecord(
        location_id=101,
        sensor_name="Swanston Street",
        sensor_description="Melbourne Central",
        status="Active",
        latitude=-37.8102,
        longitude=144.9631,
    )
