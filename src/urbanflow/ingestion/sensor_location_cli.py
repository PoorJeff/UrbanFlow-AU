from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from urbanflow.ingestion.melbourne_api import MelbourneApiClient, MelbourneApiError
from urbanflow.ingestion.sensor_location_pipeline import (
    SensorLocationIngestionResult,
    SupportsDatasetRecords,
    ingest_sensor_locations,
)
from urbanflow.ingestion.sensor_locations import SensorLocationParseError


def positive_integer(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("page limit must be an integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("page limit must be greater than zero")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Melbourne sensor locations.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--page-limit", type=positive_integer, default=100)
    return parser


def result_summary(result: SensorLocationIngestionResult) -> dict[str, int | str]:
    return {
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
        [httpx.Client], SupportsDatasetRecords
    ] = _default_api_client_factory,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        with httpx.Client(timeout=30.0) as http_client:
            result = ingest_sensor_locations(
                api_client=api_client_factory(http_client),
                raw_root_dir=args.raw_root,
                manifest_root_dir=args.manifest_root,
                page_limit=args.page_limit,
            )
    except (MelbourneApiError, OSError, SensorLocationParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result_summary(result), sort_keys=True))
    return 0
