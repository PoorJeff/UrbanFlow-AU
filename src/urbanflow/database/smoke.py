from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from urbanflow.database.engine import create_database_engine
from urbanflow.database.models import Base
from urbanflow.database.repositories import upsert_hourly_rows, upsert_sensor_rows

SMOKE_DATABASE_URL_ENV_VAR = "URBANFLOW_SMOKE_DATABASE_URL"

_SAFE_SCHEMA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class PostgresSmokeResult:
    schema_name: str
    sensor_row_count: int
    hourly_row_count: int


def validate_smoke_schema_name(schema_name: str) -> str:
    if not _SAFE_SCHEMA_NAME_PATTERN.fullmatch(schema_name):
        raise ValueError(
            "Smoke schema name must be a safe PostgreSQL identifier: "
            "lowercase letters, digits, and underscores only, starting with a letter."
        )
    return schema_name


def _temporary_schema_name() -> str:
    return f"urbanflow_smoke_{uuid4().hex[:12]}"


def _quote_identifier(identifier: str) -> str:
    return f'"{validate_smoke_schema_name(identifier)}"'


def _sensor_smoke_row() -> dict[str, object]:
    return {
        "location_id": 999001,
        "sensor_name": "Smoke Test Sensor",
        "sensor_description": "Synthetic PostgreSQL persistence smoke-test sensor",
        "latitude": -37.8136,
        "longitude": 144.9631,
        "installation_date": date(2025, 1, 1),
        "status": "active",
    }


def _hourly_smoke_row() -> dict[str, object]:
    observed_at = datetime(2025, 1, 1, 7, 0, tzinfo=UTC)
    return {
        "location_id": 999001,
        "observed_at": observed_at,
        "source_sensing_date": observed_at.date(),
        "source_hourday": observed_at.hour,
        "pedestrian_count": 42,
        "direction_1_count": 20,
        "direction_2_count": 22,
        "source_snapshot_path": "smoke://postgres-persistence",
    }


def run_postgres_persistence_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> PostgresSmokeResult:
    schema = validate_smoke_schema_name(schema_name or _temporary_schema_name())
    quoted_schema = _quote_identifier(schema)
    engine = create_database_engine(database_url)
    schema_created = False

    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {quoted_schema}")
            schema_created = True
            connection.exec_driver_sql(f"SET search_path TO {quoted_schema}")
            Base.metadata.create_all(connection)

            session = Session(bind=connection, autoflush=False, expire_on_commit=False)
            try:
                upsert_sensor_rows(session, [_sensor_smoke_row()])
                upsert_hourly_rows(session, [_hourly_smoke_row()])
                sensor_row_count = session.execute(
                    text("SELECT COUNT(*) FROM sensor_dim")
                ).scalar_one()
                hourly_row_count = session.execute(
                    text("SELECT COUNT(*) FROM pedestrian_hourly_fact")
                ).scalar_one()
            finally:
                session.close()

        return PostgresSmokeResult(
            schema_name=schema,
            sensor_row_count=int(sensor_row_count),
            hourly_row_count=int(hourly_row_count),
        )
    finally:
        try:
            if schema_created:
                with engine.begin() as connection:
                    connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {quoted_schema} CASCADE")
        finally:
            engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test UrbanFlow AU PostgreSQL persistence against a local database."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional temporary schema name for debugging. Defaults to a generated name.",
    )
    return parser


def _database_url(
    explicit_database_url: str | None,
    *,
    environ: Mapping[str, str] | None,
) -> str:
    values = os.environ if environ is None else environ
    database_url = (
        explicit_database_url
        if explicit_database_url is not None
        else values.get(SMOKE_DATABASE_URL_ENV_VAR)
    )
    if database_url is None or not database_url.strip():
        raise ValueError(
            "PostgreSQL smoke database URL is required. "
            f"Pass --database-url or set {SMOKE_DATABASE_URL_ENV_VAR}."
        )
    return database_url.strip()


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        database_url = _database_url(args.database_url, environ=environ)
        result = run_postgres_persistence_smoke(
            database_url,
            schema_name=args.schema_name,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SQLAlchemyError as exc:
        print(f"PostgreSQL persistence smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
