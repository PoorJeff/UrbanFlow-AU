from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

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
