from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from urbanflow.modeling.lightgbm_artifact import HolidayCalendar, LightGBMArtifactError
from urbanflow.modeling.supervised_dataset import (
    SupervisedSnapshotBuildError,
    SupervisedSnapshotBuildResult,
    SupervisedSnapshotWriteError,
    build_supervised_csv_from_hourly_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local supervised CSV from a verified hourly-count snapshot."
    )
    parser.add_argument(
        "snapshot_path",
        type=Path,
        help="Local hourly-count CSV snapshot.",
    )
    parser.add_argument(
        "manifest_path",
        type=Path,
        help="Matching schema-v1 hourly-count ingestion manifest.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="New local path for the generated supervised CSV.",
    )
    parser.add_argument(
        "--holiday-calendar",
        type=Path,
        required=True,
        help="Local JSON holiday calendar with explicit date coverage.",
    )
    return parser


def _result_summary(result: SupervisedSnapshotBuildResult) -> dict[str, object]:
    return {
        "output_path": str(result.output_path),
        "snapshot_sha256": result.snapshot_sha256,
        "source_row_count": result.source_row_count,
        "supervised_row_count": result.supervised_row_count,
        "training_row_count": result.training_row_count,
        "validation_warning_count": result.validation_warning_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        holiday_calendar = HolidayCalendar.from_json_file(args.holiday_calendar)
        result = build_supervised_csv_from_hourly_snapshot(
            args.snapshot_path,
            args.manifest_path,
            args.output_csv,
            holiday_calendar=holiday_calendar,
        )
    except (LightGBMArtifactError, SupervisedSnapshotBuildError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SupervisedSnapshotWriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_result_summary(result), sort_keys=True))
    return 0
