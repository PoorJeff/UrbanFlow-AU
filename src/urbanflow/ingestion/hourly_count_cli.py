from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from urbanflow.ingestion.hourly_count_pipeline import (
    HourlyCountIngestionResult,
    SupportsHourlyCountExport,
    ingest_hourly_counts,
)
from urbanflow.ingestion.hourly_counts import (
    HourlyCountDateRange,
    HourlyCountIngestionError,
    parse_iso_date,
    validate_date_range,
    year_date_range,
)
from urbanflow.ingestion.melbourne_api import MelbourneApiClient, MelbourneApiError


def positive_year(value: str) -> int:
    try:
        year = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("year must be an integer") from exc
    if year < 1900:
        raise argparse.ArgumentTypeError("year must be 1900 or later")
    return year


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Melbourne hourly pedestrian counts.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--year", type=positive_year)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser


def date_range_from_args(args: argparse.Namespace) -> HourlyCountDateRange:
    has_year = args.year is not None
    has_start = args.start_date is not None
    has_end = args.end_date is not None
    if has_year and (has_start or has_end):
        raise argparse.ArgumentTypeError(
            "provide either --year or --start-date/--end-date, not both"
        )
    if has_year:
        return year_date_range(args.year)
    if has_start != has_end:
        raise argparse.ArgumentTypeError("provide both --start-date and --end-date")
    if not has_start:
        raise argparse.ArgumentTypeError("provide --year or both --start-date and --end-date")
    return validate_date_range(parse_iso_date(args.start_date), parse_iso_date(args.end_date))


def result_summary(result: HourlyCountIngestionResult) -> dict[str, int | str | dict[str, str]]:
    return {
        "date_range": {
            "end": result.date_range.end_date.isoformat(),
            "start": result.date_range.start_date.isoformat(),
        },
        "extracted_at": result.extracted_at.isoformat(),
        "manifest_path": result.manifest_path.as_posix(),
        "record_count": result.record_count,
        "snapshot_path": result.snapshot_path.as_posix(),
        "source_dataset": result.source_dataset,
        "source_total_count": result.source_total_count,
        "source_url": result.source_url,
    }


def _default_api_client_factory(http_client: httpx.Client) -> MelbourneApiClient:
    return MelbourneApiClient(http_client=http_client)


def main(
    argv: Sequence[str] | None = None,
    *,
    api_client_factory: Callable[
        [httpx.Client], SupportsHourlyCountExport
    ] = _default_api_client_factory,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        date_range = date_range_from_args(args)
    except (argparse.ArgumentTypeError, HourlyCountIngestionError) as exc:
        parser.error(str(exc))

    try:
        with httpx.Client(timeout=30.0) as http_client:
            result = ingest_hourly_counts(
                api_client=api_client_factory(http_client),
                raw_root_dir=args.raw_root,
                manifest_root_dir=args.manifest_root,
                date_range=date_range,
            )
    except (HourlyCountIngestionError, MelbourneApiError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result_summary(result), sort_keys=True))
    return 0
