# PostgreSQL Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:subagent-driven-development` only if the user explicitly asks for subagents or parallel agent work. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLAlchemy/Alembic PostgreSQL persistence for validated `sensor_locations` and `hourly_counts` snapshots.

**Architecture:** Create a focused `urbanflow.database` package with configuration, SQLAlchemy 2.x typed models, time conversion, row transformations, PostgreSQL upsert repositories, validation-gated loaders, and a small CLI. Add Alembic configuration plus one initial migration for `sensor_dim` and `pedestrian_hourly_fact`; keep Docker, Prefect, weather, API, and modeling out of this slice.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.x, Alembic, Psycopg 3, pandas/Pandera validation layer, pytest, Ruff.

---

## File Structure

- Modify `pyproject.toml`
  - Add `SQLAlchemy`, `alembic`, and `psycopg[binary]` runtime dependencies.
- Create `src/urbanflow/database/__init__.py`
  - Export the public database API.
- Create `src/urbanflow/database/config.py`
  - Resolve database URLs from CLI input or `URBANFLOW_DATABASE_URL`.
- Create `src/urbanflow/database/engine.py`
  - Build SQLAlchemy engines and session factories.
- Create `src/urbanflow/database/models.py`
  - Define `Base`, `SensorDim`, and `PedestrianHourlyFact`.
- Create `src/urbanflow/database/time.py`
  - Convert `sensing_date` + `hourday` into Melbourne-local timezone-aware datetimes.
- Create `src/urbanflow/database/loaders.py`
  - Validate snapshots, transform rows, and call repositories.
- Create `src/urbanflow/database/repositories.py`
  - Build and execute PostgreSQL upsert statements.
- Create `src/urbanflow/database/cli.py`
  - Manual CLI for loading one validated snapshot.
- Create `scripts/load_snapshot_to_db.py`
  - Thin script wrapper.
- Create `alembic.ini`
  - Alembic configuration.
- Create `migrations/env.py`
  - Alembic environment that imports model metadata.
- Create `migrations/versions/20260628_0001_create_core_tables.py`
  - Initial migration for `sensor_dim` and `pedestrian_hourly_fact`.
- Create tests under `tests/unit/database/`
  - Cover config, model metadata, time conversion, row transforms, repository statement compilation, validation gating, and CLI behavior.
- Modify `README.md`
  - Document database URL setup, Alembic upgrade command, and snapshot load command.

## Task 1: Dependencies and Database Configuration

**Files:**
- Modify: `pyproject.toml`
- Create: `src/urbanflow/database/__init__.py`
- Create: `src/urbanflow/database/config.py`
- Create: `tests/unit/database/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/unit/database/test_config.py`:

```python
import pytest

from urbanflow.database.config import DatabaseConfigError, get_database_url


def test_get_database_url_prefers_explicit_value() -> None:
    url = get_database_url(
        database_url="postgresql+psycopg://user:pass@localhost:5432/urbanflow",
        environ={"URBANFLOW_DATABASE_URL": "postgresql+psycopg://ignored"},
    )

    assert url == "postgresql+psycopg://user:pass@localhost:5432/urbanflow"


def test_get_database_url_reads_environment_value() -> None:
    url = get_database_url(
        database_url=None,
        environ={"URBANFLOW_DATABASE_URL": "postgresql+psycopg://env"},
    )

    assert url == "postgresql+psycopg://env"


def test_get_database_url_rejects_missing_value() -> None:
    with pytest.raises(DatabaseConfigError, match="Database URL is required"):
        get_database_url(database_url=None, environ={})


def test_get_database_url_rejects_blank_value() -> None:
    with pytest.raises(DatabaseConfigError, match="Database URL is required"):
        get_database_url(database_url="  ", environ={})
```

- [ ] **Step 2: Run config tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_config.py -v
```

Expected: collection fails with `ModuleNotFoundError` for `urbanflow.database`.

- [ ] **Step 3: Add dependencies**

Modify `pyproject.toml` dependencies to include:

```toml
dependencies = [
    "alembic>=1.13,<2",
    "httpx>=0.28,<1",
    "pandas>=2.1,<4",
    "pandera[pandas]>=0.24,<1",
    "psycopg[binary]>=3.2,<4",
    "SQLAlchemy>=2.0,<3",
    "tenacity>=9,<10",
]
```

Run:

```powershell
& ..\..\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Expected: dependencies install or are already satisfied.

- [ ] **Step 4: Implement config module**

Create `src/urbanflow/database/config.py`:

```python
from __future__ import annotations

import os
from collections.abc import Mapping

DATABASE_URL_ENV_VAR = "URBANFLOW_DATABASE_URL"


class DatabaseConfigError(Exception):
    """Raised when database configuration is missing or unusable."""


def get_database_url(
    database_url: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environ is None else environ
    resolved = database_url if database_url is not None else values.get(DATABASE_URL_ENV_VAR)
    if resolved is None or not resolved.strip():
        raise DatabaseConfigError(
            f"Database URL is required. Pass --database-url or set {DATABASE_URL_ENV_VAR}."
        )
    return resolved.strip()
```

Create `src/urbanflow/database/__init__.py`:

```python
"""Database persistence utilities for UrbanFlow AU."""

from urbanflow.database.config import DATABASE_URL_ENV_VAR, DatabaseConfigError, get_database_url

__all__ = ["DATABASE_URL_ENV_VAR", "DatabaseConfigError", "get_database_url"]
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_config.py -v
```

Expected: all config tests pass.

Commit:

```powershell
git add pyproject.toml src/urbanflow/database/__init__.py src/urbanflow/database/config.py tests/unit/database/test_config.py
git commit -m "feat: add database configuration"
```

## Task 2: SQLAlchemy Models and Alembic Migration

**Files:**
- Create: `src/urbanflow/database/models.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/versions/20260628_0001_create_core_tables.py`
- Create: `tests/unit/database/test_models.py`

- [ ] **Step 1: Write failing model metadata tests**

Create `tests/unit/database/test_models.py`:

```python
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, PrimaryKeyConstraint

from urbanflow.database.models import Base, PedestrianHourlyFact, SensorDim


def test_sensor_dim_table_contract() -> None:
    table = SensorDim.__table__

    assert table.name == "sensor_dim"
    assert table.c.location_id.primary_key is True
    assert table.c.installation_date.nullable is True
    assert table.c.updated_at.nullable is False

    checks = {constraint.name for constraint in table.constraints if isinstance(constraint, CheckConstraint)}
    assert checks == {"ck_sensor_dim_latitude_range", "ck_sensor_dim_longitude_range"}


def test_pedestrian_hourly_fact_table_contract() -> None:
    table = PedestrianHourlyFact.__table__

    assert table.name == "pedestrian_hourly_fact"
    assert [column.name for column in table.primary_key.columns] == ["location_id", "observed_at"]
    assert table.c.source_snapshot_path.nullable is False

    foreign_keys = [c for c in table.constraints if isinstance(c, ForeignKeyConstraint)]
    assert len(foreign_keys) == 1
    assert list(foreign_keys[0].columns)[0].name == "location_id"

    checks = {constraint.name for constraint in table.constraints if isinstance(constraint, CheckConstraint)}
    assert checks == {
        "ck_pedestrian_hourly_fact_pedestrian_count_non_negative",
        "ck_pedestrian_hourly_fact_direction_1_count_non_negative",
        "ck_pedestrian_hourly_fact_direction_2_count_non_negative",
        "ck_pedestrian_hourly_fact_source_hourday_range",
    }


def test_metadata_contains_only_core_tables() -> None:
    assert set(Base.metadata.tables) == {"sensor_dim", "pedestrian_hourly_fact"}
```

- [ ] **Step 2: Run model tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_models.py -v
```

Expected: collection fails because `urbanflow.database.models` does not exist.

- [ ] **Step 3: Implement SQLAlchemy models**

Create `src/urbanflow/database/models.py`:

```python
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, Float, ForeignKey, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base metadata for UrbanFlow database models."""


class SensorDim(Base):
    __tablename__ = "sensor_dim"
    __table_args__ = (
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_sensor_dim_latitude_range"),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180",
            name="ck_sensor_dim_longitude_range",
        ),
    )

    location_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sensor_name: Mapped[str] = mapped_column(Text, nullable=False)
    sensor_description: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    installation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PedestrianHourlyFact(Base):
    __tablename__ = "pedestrian_hourly_fact"
    __table_args__ = (
        CheckConstraint(
            "pedestrian_count >= 0",
            name="ck_pedestrian_hourly_fact_pedestrian_count_non_negative",
        ),
        CheckConstraint(
            "direction_1_count >= 0",
            name="ck_pedestrian_hourly_fact_direction_1_count_non_negative",
        ),
        CheckConstraint(
            "direction_2_count >= 0",
            name="ck_pedestrian_hourly_fact_direction_2_count_non_negative",
        ),
        CheckConstraint(
            "source_hourday >= 0 AND source_hourday <= 23",
            name="ck_pedestrian_hourly_fact_source_hourday_range",
        ),
    )

    location_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sensor_dim.location_id"),
        primary_key=True,
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    source_sensing_date: Mapped[date] = mapped_column(Date, nullable=False)
    source_hourday: Mapped[int] = mapped_column(Integer, nullable=False)
    pedestrian_count: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_1_count: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_2_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    source_snapshot_path: Mapped[str] = mapped_column(Text, nullable=False)
```

- [ ] **Step 4: Add Alembic configuration**

Create `alembic.ini`:

```ini
[alembic]
script_location = migrations
prepend_sys_path = .
timezone = UTC

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Create `migrations/env.py`:

```python
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from urbanflow.database.config import get_database_url
from urbanflow.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    return get_database_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create `migrations/versions/20260628_0001_create_core_tables.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision = "20260628_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sensor_dim",
        sa.Column("location_id", sa.Integer(), nullable=False),
        sa.Column("sensor_name", sa.Text(), nullable=False),
        sa.Column("sensor_description", sa.Text(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("installation_date", sa.Date(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_sensor_dim_latitude_range"),
        sa.CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_sensor_dim_longitude_range"),
        sa.PrimaryKeyConstraint("location_id"),
    )
    op.create_table(
        "pedestrian_hourly_fact",
        sa.Column("location_id", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_sensing_date", sa.Date(), nullable=False),
        sa.Column("source_hourday", sa.Integer(), nullable=False),
        sa.Column("pedestrian_count", sa.Integer(), nullable=False),
        sa.Column("direction_1_count", sa.Integer(), nullable=False),
        sa.Column("direction_2_count", sa.Integer(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("source_snapshot_path", sa.Text(), nullable=False),
        sa.CheckConstraint("pedestrian_count >= 0", name="ck_pedestrian_hourly_fact_pedestrian_count_non_negative"),
        sa.CheckConstraint("direction_1_count >= 0", name="ck_pedestrian_hourly_fact_direction_1_count_non_negative"),
        sa.CheckConstraint("direction_2_count >= 0", name="ck_pedestrian_hourly_fact_direction_2_count_non_negative"),
        sa.CheckConstraint("source_hourday >= 0 AND source_hourday <= 23", name="ck_pedestrian_hourly_fact_source_hourday_range"),
        sa.ForeignKeyConstraint(["location_id"], ["sensor_dim.location_id"]),
        sa.PrimaryKeyConstraint("location_id", "observed_at"),
    )


def downgrade() -> None:
    op.drop_table("pedestrian_hourly_fact")
    op.drop_table("sensor_dim")
```

- [ ] **Step 5: Run model tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff format src/urbanflow/database/models.py migrations/env.py migrations/versions/20260628_0001_create_core_tables.py
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_models.py -v
```

Expected: model tests pass.

Commit:

```powershell
git add alembic.ini migrations src/urbanflow/database/models.py tests/unit/database/test_models.py
git commit -m "feat: add core database models"
```

## Task 3: Engine Helpers, Time Conversion, and Row Transforms

**Files:**
- Create: `src/urbanflow/database/engine.py`
- Create: `src/urbanflow/database/time.py`
- Create: `src/urbanflow/database/loaders.py`
- Modify: `src/urbanflow/database/__init__.py`
- Create: `tests/unit/database/test_engine.py`
- Create: `tests/unit/database/test_time_and_rows.py`

- [ ] **Step 1: Write failing engine tests**

Create `tests/unit/database/test_engine.py`:

```python
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from urbanflow.database.engine import create_database_engine, create_session_factory


def test_create_database_engine_returns_sqlalchemy_engine() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")

    assert isinstance(engine, Engine)


def test_create_session_factory_opens_sessions() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        assert isinstance(session, Session)
```

- [ ] **Step 2: Write failing time/row tests**

Create `tests/unit/database/test_time_and_rows.py`:

```python
import json
from datetime import date, datetime

from urbanflow.database.loaders import (
    hourly_count_rows_from_snapshot,
    sensor_rows_from_snapshot,
)
from urbanflow.database.time import melbourne_observed_at


def test_melbourne_observed_at_returns_timezone_aware_hour() -> None:
    observed_at = melbourne_observed_at("2025-01-01", "7")

    assert observed_at == datetime.fromisoformat("2025-01-01T07:00:00+11:00")
    assert observed_at.tzinfo is not None


def test_sensor_rows_from_snapshot_normalizes_database_shape(tmp_path) -> None:
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": "2020-01-02",
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                },
                {
                    "location_id": 2,
                    "sensor_description": "Null Date",
                    "sensor_name": "Sensor B",
                    "installation_date": None,
                    "status": "I",
                    "latitude": -37.82,
                    "longitude": 144.97,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = sensor_rows_from_snapshot(snapshot_path)

    assert rows == [
        {
            "location_id": 1,
            "sensor_name": "Sensor A",
            "sensor_description": "Bourke Street",
            "latitude": -37.81,
            "longitude": 144.96,
            "installation_date": date(2020, 1, 2),
            "status": "A",
        },
        {
            "location_id": 2,
            "sensor_name": "Sensor B",
            "sensor_description": "Null Date",
            "latitude": -37.82,
            "longitude": 144.97,
            "installation_date": None,
            "status": "I",
        },
    ]


def test_hourly_count_rows_from_snapshot_normalizes_database_shape(tmp_path) -> None:
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text(
        "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location\n"
        'abc,1,2025-01-01,7,2,3,5,Sensor A,"-37.81,144.96"\n',
        encoding="utf-8",
    )

    rows = hourly_count_rows_from_snapshot(snapshot_path)

    assert rows == [
        {
            "location_id": 1,
            "observed_at": datetime.fromisoformat("2025-01-01T07:00:00+11:00"),
            "source_sensing_date": date(2025, 1, 1),
            "source_hourday": 7,
            "pedestrian_count": 5,
            "direction_1_count": 2,
            "direction_2_count": 3,
            "source_snapshot_path": str(snapshot_path),
        }
    ]
```

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_engine.py tests/unit/database/test_time_and_rows.py -v
```

Expected: collection fails because engine/time/loaders functions are missing.

- [ ] **Step 4: Implement engine helpers**

Create `src/urbanflow/database/engine.py`:

```python
from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_database_engine(database_url: str, *, echo: bool = False) -> Engine:
    return create_engine(database_url, echo=echo, future=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
```

- [ ] **Step 5: Implement time conversion and row transforms**

Create `src/urbanflow/database/time.py`:

```python
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")


def parse_source_date(value: str | date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


def melbourne_observed_at(sensing_date: str | date, hourday: str | int) -> datetime:
    source_date = parse_source_date(sensing_date)
    source_hour = int(hourday)
    if source_hour < 0 or source_hour > 23:
        raise ValueError(f"hourday must be between 0 and 23: {hourday}")
    return datetime.combine(source_date, time(hour=source_hour), tzinfo=MELBOURNE_TZ)
```

Create the first part of `src/urbanflow/database/loaders.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from urbanflow.database.time import melbourne_observed_at, parse_source_date
from urbanflow.validation.pipeline import validate_snapshot
from urbanflow.validation.snapshot_readers import (
    read_hourly_counts_snapshot,
    read_sensor_locations_snapshot,
)


class DatabaseLoadError(Exception):
    """Raised when a validated snapshot cannot be loaded into the database."""


@dataclass(frozen=True)
class DatabaseLoadResult:
    dataset: str
    row_count: int
    validation_warning_count: int


def _parse_optional_date(value: Any) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    return parse_source_date(value)


def sensor_rows_from_snapshot(snapshot_path: Path) -> list[dict[str, object]]:
    frame = read_sensor_locations_snapshot(snapshot_path)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "location_id": int(record["location_id"]),
                "sensor_name": str(record["sensor_name"]),
                "sensor_description": str(record["sensor_description"]),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "installation_date": _parse_optional_date(record.get("installation_date")),
                "status": str(record["status"]),
            }
        )
    return rows


def hourly_count_rows_from_snapshot(snapshot_path: Path) -> list[dict[str, object]]:
    frame = read_hourly_counts_snapshot(snapshot_path)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        source_date = parse_source_date(record["sensing_date"])
        source_hour = int(record["hourday"])
        rows.append(
            {
                "location_id": int(record["location_id"]),
                "observed_at": melbourne_observed_at(source_date, source_hour),
                "source_sensing_date": source_date,
                "source_hourday": source_hour,
                "pedestrian_count": int(record["pedestriancount"]),
                "direction_1_count": int(record["direction_1"]),
                "direction_2_count": int(record["direction_2"]),
                "source_snapshot_path": str(snapshot_path),
            }
        )
    return rows
```

Update `src/urbanflow/database/__init__.py`:

```python
"""Database persistence utilities for UrbanFlow AU."""

from urbanflow.database.config import DATABASE_URL_ENV_VAR, DatabaseConfigError, get_database_url
from urbanflow.database.engine import create_database_engine, create_session_factory
from urbanflow.database.loaders import DatabaseLoadError, DatabaseLoadResult

__all__ = [
    "DATABASE_URL_ENV_VAR",
    "DatabaseConfigError",
    "DatabaseLoadError",
    "DatabaseLoadResult",
    "create_database_engine",
    "create_session_factory",
    "get_database_url",
]
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_engine.py tests/unit/database/test_time_and_rows.py -v
```

Expected: all focused tests pass.

Commit:

```powershell
git add src/urbanflow/database tests/unit/database/test_engine.py tests/unit/database/test_time_and_rows.py
git commit -m "feat: add database row transforms"
```

## Task 4: PostgreSQL Upsert Repositories

**Files:**
- Create: `src/urbanflow/database/repositories.py`
- Create: `tests/unit/database/test_repositories.py`

- [ ] **Step 1: Write failing repository tests**

Create `tests/unit/database/test_repositories.py`:

```python
from datetime import UTC, date, datetime

from sqlalchemy.dialects import postgresql

from urbanflow.database.repositories import (
    build_hourly_upsert_statement,
    build_sensor_upsert_statement,
)


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
```

- [ ] **Step 2: Run repository tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_repositories.py -v
```

Expected: collection fails because `urbanflow.database.repositories` does not exist.

- [ ] **Step 3: Implement repository statement builders and executors**

Create `src/urbanflow/database/repositories.py`:

```python
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
```

- [ ] **Step 4: Run repository tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_repositories.py -v
```

Expected: repository tests pass.

Commit:

```powershell
git add src/urbanflow/database/repositories.py tests/unit/database/test_repositories.py
git commit -m "feat: add database upsert repositories"
```

## Task 5: Validation-Gated Loaders

**Files:**
- Modify: `src/urbanflow/database/loaders.py`
- Create: `tests/unit/database/test_loaders.py`

- [ ] **Step 1: Write failing loader tests**

Create `tests/unit/database/test_loaders.py`:

```python
from datetime import UTC, datetime

import pytest

from urbanflow.database.loaders import (
    DatabaseLoadError,
    load_hourly_counts_snapshot,
    load_sensor_locations_snapshot,
)
from urbanflow.validation.reports import ValidationIssue, ValidationReport


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, statement) -> None:
        self.calls.append(statement)


def _passing_report(dataset: str, snapshot_path: str, warning_count: int = 0) -> ValidationReport:
    return ValidationReport(
        dataset=dataset,
        snapshot_path=snapshot_path,
        validated_at=datetime(2026, 6, 28, 12, tzinfo=UTC),
        row_count=1,
        warnings=tuple(
            ValidationIssue(code=f"WARN_{index}", message="warning") for index in range(warning_count)
        ),
    )


def _failing_report(dataset: str, snapshot_path: str) -> ValidationReport:
    return ValidationReport(
        dataset=dataset,
        snapshot_path=snapshot_path,
        validated_at=datetime(2026, 6, 28, 12, tzinfo=UTC),
        row_count=0,
        errors=(ValidationIssue(code="SCHEMA_INVALID", message="bad snapshot"),),
    )


def test_load_sensor_locations_snapshot_refuses_failed_validation(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "records.json"
    monkeypatch.setattr(
        "urbanflow.database.loaders.validate_snapshot",
        lambda dataset, path: _failing_report(dataset, str(path)),
    )

    with pytest.raises(DatabaseLoadError, match="Validation failed"):
        load_sensor_locations_snapshot(FakeSession(), snapshot_path)


def test_load_sensor_locations_snapshot_calls_repository(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "records.json"
    fake_session = FakeSession()
    captured = {}
    monkeypatch.setattr(
        "urbanflow.database.loaders.validate_snapshot",
        lambda dataset, path: _passing_report(dataset, str(path), warning_count=2),
    )
    monkeypatch.setattr(
        "urbanflow.database.loaders.sensor_rows_from_snapshot",
        lambda path: [{"location_id": 1}],
    )
    def fake_upsert_sensor_rows(session, rows):
        captured["rows"] = rows
        return 1

    monkeypatch.setattr(
        "urbanflow.database.loaders.upsert_sensor_rows",
        fake_upsert_sensor_rows,
    )

    result = load_sensor_locations_snapshot(fake_session, snapshot_path)

    assert result.dataset == "sensor_locations"
    assert result.row_count == 1
    assert result.validation_warning_count == 2
    assert captured["rows"] == [{"location_id": 1}]


def test_load_hourly_counts_snapshot_calls_repository(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "records.csv"
    fake_session = FakeSession()
    captured = {}
    monkeypatch.setattr(
        "urbanflow.database.loaders.validate_snapshot",
        lambda dataset, path: _passing_report(dataset, str(path)),
    )
    monkeypatch.setattr(
        "urbanflow.database.loaders.hourly_count_rows_from_snapshot",
        lambda path: [{"location_id": 1}],
    )
    def fake_upsert_hourly_rows(session, rows):
        captured["rows"] = rows
        return 1

    monkeypatch.setattr(
        "urbanflow.database.loaders.upsert_hourly_rows",
        fake_upsert_hourly_rows,
    )

    result = load_hourly_counts_snapshot(fake_session, snapshot_path)

    assert result.dataset == "hourly_counts"
    assert result.row_count == 1
    assert result.validation_warning_count == 0
    assert captured["rows"] == [{"location_id": 1}]
```

- [ ] **Step 2: Run loader tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_loaders.py -v
```

Expected: tests fail because `load_sensor_locations_snapshot` and `load_hourly_counts_snapshot` are missing.

- [ ] **Step 3: Implement validation-gated loader functions**

Append to `src/urbanflow/database/loaders.py`:

```python
from sqlalchemy.orm import Session

from urbanflow.database.repositories import upsert_hourly_rows, upsert_sensor_rows


def _ensure_validation_passed(dataset: str, snapshot_path: Path):
    report = validate_snapshot(dataset, snapshot_path)
    if not report.passed:
        codes = ", ".join(issue.code for issue in report.errors)
        raise DatabaseLoadError(f"Validation failed for {dataset}: {codes}")
    return report


def load_sensor_locations_snapshot(session: Session, snapshot_path: Path) -> DatabaseLoadResult:
    report = _ensure_validation_passed("sensor_locations", snapshot_path)
    rows = sensor_rows_from_snapshot(snapshot_path)
    row_count = upsert_sensor_rows(session, rows)
    return DatabaseLoadResult(
        dataset="sensor_locations",
        row_count=row_count,
        validation_warning_count=len(report.warnings),
    )


def load_hourly_counts_snapshot(session: Session, snapshot_path: Path) -> DatabaseLoadResult:
    report = _ensure_validation_passed("hourly_counts", snapshot_path)
    rows = hourly_count_rows_from_snapshot(snapshot_path)
    row_count = upsert_hourly_rows(session, rows)
    return DatabaseLoadResult(
        dataset="hourly_counts",
        row_count=row_count,
        validation_warning_count=len(report.warnings),
    )
```

- [ ] **Step 4: Run loader tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_loaders.py -v
```

Expected: loader tests pass.

Commit:

```powershell
git add src/urbanflow/database/loaders.py tests/unit/database/test_loaders.py
git commit -m "feat: gate database loads on validation"
```

## Task 6: Database Load CLI and Documentation

**Files:**
- Create: `src/urbanflow/database/cli.py`
- Create: `scripts/load_snapshot_to_db.py`
- Create: `tests/unit/database/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/database/test_cli.py`:

```python
import json
from pathlib import Path

from urbanflow.database.cli import main
from urbanflow.database.loaders import DatabaseLoadResult


def test_database_cli_returns_two_when_database_url_missing(tmp_path, capsys) -> None:
    exit_code = main(["sensor_locations", str(tmp_path / "records.json")], environ={})

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Database URL is required" in captured.err


def test_database_cli_loads_sensor_snapshot(monkeypatch, tmp_path, capsys) -> None:
    snapshot_path = tmp_path / "records.json"
    calls = {}

    class FakeSessionFactory:
        def begin(self):
            class Context:
                def __enter__(self):
                    return "session"

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Context()

    monkeypatch.setattr("urbanflow.database.cli.create_database_engine", lambda url: calls.setdefault("url", url))
    monkeypatch.setattr("urbanflow.database.cli.create_session_factory", lambda engine: FakeSessionFactory())
    def fake_load_sensor_locations_snapshot(session, path):
        calls["loaded"] = (session, path)
        return DatabaseLoadResult(
            dataset="sensor_locations",
            row_count=3,
            validation_warning_count=0,
        )

    monkeypatch.setattr(
        "urbanflow.database.cli.load_sensor_locations_snapshot",
        fake_load_sensor_locations_snapshot,
    )

    exit_code = main(
        ["sensor_locations", str(snapshot_path), "--database-url", "postgresql+psycopg://db"],
        environ={},
    )

    assert exit_code == 0
    assert calls["url"] == "postgresql+psycopg://db"
    assert calls["loaded"] == ("session", snapshot_path)
    assert json.loads(capsys.readouterr().out)["dataset"] == "sensor_locations"


def test_database_load_script_help() -> None:
    import subprocess
    import sys

    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "load_snapshot_to_db.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Load a validated UrbanFlow AU snapshot into PostgreSQL" in result.stdout
```

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_cli.py -v
```

Expected: collection fails because `urbanflow.database.cli` does not exist.

- [ ] **Step 3: Implement CLI and script**

Create `src/urbanflow/database/cli.py`:

```python
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

from urbanflow.database.config import DatabaseConfigError, get_database_url
from urbanflow.database.engine import create_database_engine, create_session_factory
from urbanflow.database.loaders import (
    DatabaseLoadError,
    DatabaseLoadResult,
    load_hourly_counts_snapshot,
    load_sensor_locations_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a validated UrbanFlow AU snapshot into PostgreSQL."
    )
    parser.add_argument("dataset", choices=("sensor_locations", "hourly_counts"))
    parser.add_argument("snapshot_path", type=Path)
    parser.add_argument("--database-url", default=None)
    return parser


def _summary(result: DatabaseLoadResult) -> dict[str, object]:
    return {
        "dataset": result.dataset,
        "row_count": result.row_count,
        "validation_warning_count": result.validation_warning_count,
    }


def _load(dataset: str, session, snapshot_path: Path) -> DatabaseLoadResult:
    if dataset == "sensor_locations":
        return load_sensor_locations_snapshot(session, snapshot_path)
    return load_hourly_counts_snapshot(session, snapshot_path)


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        database_url = get_database_url(args.database_url, environ=environ)
        engine = create_database_engine(database_url)
        session_factory = create_session_factory(engine)
        with session_factory.begin() as session:
            result = _load(args.dataset, session, args.snapshot_path)
    except DatabaseConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except DatabaseLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(_summary(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Create `scripts/load_snapshot_to_db.py`:

```python
from urbanflow.database.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update README**

Add this section after snapshot validation:

````markdown
## Load validated snapshots into PostgreSQL

Set a SQLAlchemy-compatible PostgreSQL URL, run migrations, then load validated snapshots:

```powershell
$env:URBANFLOW_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
alembic upgrade head

$sensorSnapshot = Get-ChildItem data/raw/melbourne/sensor_locations -Filter records.json -Recurse | Select-Object -First 1
python scripts/load_snapshot_to_db.py sensor_locations $sensorSnapshot.FullName

$hourlySnapshot = Get-ChildItem data/raw/melbourne/hourly_counts -Filter records.csv -Recurse | Select-Object -First 1
python scripts/load_snapshot_to_db.py hourly_counts $hourlySnapshot.FullName
```

The database loader validates each snapshot before writing. Validation hard errors stop
the load; validation warnings are reported but do not block insertion.
````

- [ ] **Step 5: Run CLI tests and full quality gate**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/database/test_cli.py -v
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff check . --no-cache
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff format --check .
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest
```

Expected: focused CLI tests pass; full quality gate passes.

Commit:

```powershell
git add README.md src/urbanflow/database/cli.py scripts/load_snapshot_to_db.py tests/unit/database/test_cli.py
git commit -m "feat: add database load CLI"
```

## Final Integration

- [ ] **Step 1: Re-run full quality gate on feature branch**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff check . --no-cache
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff format --check .
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest
```

Expected: all commands pass.

- [ ] **Step 2: Merge to `main`, verify on `main`, push only `main`**

From the main checkout:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git merge --ff-only codex/postgres-persistence
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed. No `codex/*` branch is pushed to GitHub.

- [ ] **Step 3: Clean local worktree and branch**

Verify the worktree path is under `D:\Github项目\UrbanFlow-AU\.worktrees`, then run:

```powershell
git worktree remove --force D:\Github项目\UrbanFlow-AU\.worktrees\postgres-persistence
git worktree prune
git branch -d codex/postgres-persistence
```

Expected: `git worktree list` shows only the main checkout, and `git status --short --branch`
shows `main...origin/main`.
