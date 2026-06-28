from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import Session

from urbanflow.database.models import PedestrianHourlyFact, SensorDim

Row = dict[str, Any]


def build_sensor_upsert_statement(rows: Sequence[Row]) -> Insert:
    statement = insert(SensorDim).values(list(rows))
    return statement.on_conflict_do_update(
        index_elements=[SensorDim.location_id],
        set_={
            "sensor_name": statement.excluded.sensor_name,
            "sensor_description": statement.excluded.sensor_description,
            "latitude": statement.excluded.latitude,
            "longitude": statement.excluded.longitude,
            "installation_date": statement.excluded.installation_date,
            "status": statement.excluded.status,
            "updated_at": func.now(),
        },
    )


def build_hourly_upsert_statement(rows: Sequence[Row]) -> Insert:
    statement = insert(PedestrianHourlyFact).values(list(rows))
    return statement.on_conflict_do_update(
        index_elements=[PedestrianHourlyFact.location_id, PedestrianHourlyFact.observed_at],
        set_={
            "source_sensing_date": statement.excluded.source_sensing_date,
            "source_hourday": statement.excluded.source_hourday,
            "pedestrian_count": statement.excluded.pedestrian_count,
            "direction_1_count": statement.excluded.direction_1_count,
            "direction_2_count": statement.excluded.direction_2_count,
            "ingested_at": func.now(),
            "source_snapshot_path": statement.excluded.source_snapshot_path,
        },
    )


def upsert_sensor_rows(session: Session, rows: Sequence[Row]) -> int:
    if not rows:
        return 0
    session.execute(build_sensor_upsert_statement(rows))
    return len(rows)


def upsert_hourly_rows(session: Session, rows: Sequence[Row]) -> int:
    if not rows:
        return 0
    session.execute(build_hourly_upsert_statement(rows))
    return len(rows)
