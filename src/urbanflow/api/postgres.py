from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from urbanflow.api.services import DataStoreUnavailableError, HistoryRecord, SensorRecord
from urbanflow.database.models import PedestrianHourlyFact, SensorDim

ACTIVE_SENSOR_STATUS = "A"


class PostgresSensorHistoryRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list_sensors(self, active_only: bool) -> list[SensorRecord]:
        statement = select(SensorDim).order_by(SensorDim.location_id)
        if active_only:
            statement = statement.where(SensorDim.status == ACTIVE_SENSOR_STATUS)
        try:
            with self._session_factory() as session:
                return [
                    SensorRecord(
                        location_id=sensor.location_id,
                        sensor_name=sensor.sensor_name,
                        sensor_description=sensor.sensor_description,
                        status=sensor.status,
                        latitude=sensor.latitude,
                        longitude=sensor.longitude,
                    )
                    for sensor in session.scalars(statement).all()
                ]
        except SQLAlchemyError as exc:
            raise DataStoreUnavailableError("sensor data is unavailable") from exc

    def get_sensor(self, location_id: int) -> SensorRecord | None:
        statement = select(SensorDim).where(SensorDim.location_id == location_id)
        try:
            with self._session_factory() as session:
                sensor = session.scalars(statement).one_or_none()
                if sensor is None:
                    return None
                return SensorRecord(
                    location_id=sensor.location_id,
                    sensor_name=sensor.sensor_name,
                    sensor_description=sensor.sensor_description,
                    status=sensor.status,
                    latitude=sensor.latitude,
                    longitude=sensor.longitude,
                )
        except SQLAlchemyError as exc:
            raise DataStoreUnavailableError("sensor data is unavailable") from exc

    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]:
        statement = (
            select(PedestrianHourlyFact)
            .where(
                PedestrianHourlyFact.location_id == location_id,
                PedestrianHourlyFact.observed_at >= start,
                PedestrianHourlyFact.observed_at < end,
            )
            .order_by(PedestrianHourlyFact.observed_at)
        )
        try:
            with self._session_factory() as session:
                return [
                    HistoryRecord(
                        observed_at=fact.observed_at,
                        pedestrian_count=fact.pedestrian_count,
                    )
                    for fact in sorted(
                        session.scalars(statement).all(),
                        key=lambda fact: fact.observed_at,
                    )
                ]
        except SQLAlchemyError as exc:
            raise DataStoreUnavailableError("sensor data is unavailable") from exc
