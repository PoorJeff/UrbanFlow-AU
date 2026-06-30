from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

from urbanflow.database.config import DatabaseConfigError, get_database_url
from urbanflow.ingestion.hourly_counts import (
    HourlyCountIngestionError,
    parse_iso_date,
)
from urbanflow.ingestion.melbourne_api import MelbourneApiError
from urbanflow.ingestion.sensor_locations import SensorLocationParseError
from urbanflow.orchestration.ingestion_flow import (
    IngestionFlowConfigError,
    IngestionFlowError,
    IngestionFlowResult,
    date_range_from_options,
    run_ingestion_flow,
)
from urbanflow.validation.pipeline import ValidationPipelineError


def positive_integer(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed_value


def parse_optional_date(value: str | None) -> date | None:
    if value is None:
        return None
    return parse_iso_date(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UrbanFlow AU Prefect ingestion flow.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/data_quality"))
    parser.add_argument("--year", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--page-limit", type=positive_integer, default=100)
    parser.add_argument("--load-to-database", action="store_true")
    parser.add_argument("--database-url", default=None)
    return parser


def result_summary(result: IngestionFlowResult) -> dict[str, object]:
    return asdict(result)


def _resolve_database_url(
    *,
    load_to_database: bool,
    database_url: str | None,
    environ: Mapping[str, str] | None,
) -> str | None:
    if not load_to_database:
        return None
    return get_database_url(database_url, environ=environ)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        start_date = parse_optional_date(args.start_date)
        end_date = parse_optional_date(args.end_date)
        date_range_from_options(
            year=args.year,
            start_date=start_date,
            end_date=end_date,
        )
        resolved_database_url = _resolve_database_url(
            load_to_database=args.load_to_database,
            database_url=args.database_url,
            environ=environ,
        )
        result = run_ingestion_flow(
            raw_root_dir=args.raw_root,
            manifest_root_dir=args.manifest_root,
            report_root_dir=args.report_root,
            year=args.year,
            start_date=start_date,
            end_date=end_date,
            page_limit=args.page_limit,
            load_to_database=args.load_to_database,
            database_url=resolved_database_url,
        )
    except (
        DatabaseConfigError,
        HourlyCountIngestionError,
        IngestionFlowConfigError,
        argparse.ArgumentTypeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (
        IngestionFlowError,
        MelbourneApiError,
        OSError,
        SensorLocationParseError,
        ValidationPipelineError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result_summary(result), sort_keys=True))
    return 0
