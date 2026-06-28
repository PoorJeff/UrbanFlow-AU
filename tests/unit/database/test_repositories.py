from datetime import UTC, date, datetime

from sqlalchemy.dialects import postgresql

from urbanflow.database.repositories import (
    build_hourly_upsert_statement,
    build_sensor_upsert_statement,
    upsert_hourly_rows,
    upsert_sensor_rows,
)


class FakeSession:
    def __init__(self) -> None:
        self.statements = []

    def execute(self, statement) -> None:
        self.statements.append(statement)


def _compile(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect())).lower()


def test_build_sensor_upsert_statement_uses_location_id_conflict_target() -> None:
    statement = build_sensor_upsert_statement(
        [
            {
                "location_id": 1,
                "sensor_name": "Sensor A",
                "sensor_description": "Bourke Street",
                "latitude": -37.81,
                "longitude": 144.96,
                "installation_date": date(2020, 1, 2),
                "status": "A",
            }
        ]
    )

    sql = _compile(statement)

    assert "insert into sensor_dim" in sql
    assert "on conflict (location_id) do update" in sql
    assert "sensor_name = excluded.sensor_name" in sql
    assert "updated_at = now()" in sql


def test_build_hourly_upsert_statement_uses_sensor_hour_conflict_target() -> None:
    statement = build_hourly_upsert_statement(
        [
            {
                "location_id": 1,
                "observed_at": datetime(2025, 1, 1, 0, tzinfo=UTC),
                "source_sensing_date": date(2025, 1, 1),
                "source_hourday": 0,
                "pedestrian_count": 5,
                "direction_1_count": 2,
                "direction_2_count": 3,
                "source_snapshot_path": "records.csv",
            }
        ]
    )

    sql = _compile(statement)

    assert "insert into pedestrian_hourly_fact" in sql
    assert "on conflict (location_id, observed_at) do update" in sql
    assert "pedestrian_count = excluded.pedestrian_count" in sql
    assert "ingested_at = now()" in sql


def test_upsert_sensor_rows_returns_zero_for_empty_rows() -> None:
    session = FakeSession()

    row_count = upsert_sensor_rows(session, [])

    assert row_count == 0
    assert session.statements == []


def test_upsert_hourly_rows_returns_zero_for_empty_rows() -> None:
    session = FakeSession()

    row_count = upsert_hourly_rows(session, [])

    assert row_count == 0
    assert session.statements == []
