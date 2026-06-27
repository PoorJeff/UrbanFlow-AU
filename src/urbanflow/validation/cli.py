from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from urbanflow.validation.pipeline import ValidationPipelineError, validate_snapshot
from urbanflow.validation.reports import ValidationReport

READ_ERROR_CODES = {"SNAPSHOT_READ_ERROR"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a local UrbanFlow AU raw snapshot.")
    parser.add_argument("dataset", choices=("sensor_locations", "hourly_counts"))
    parser.add_argument("snapshot_path", type=Path)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Optional root directory for full JSON validation reports.",
    )
    return parser


def _summary(report: ValidationReport) -> dict[str, object]:
    return {
        "dataset": report.dataset,
        "snapshot_path": report.snapshot_path,
        "passed": report.passed,
        "row_count": report.row_count,
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
    }


def _exit_code(report: ValidationReport) -> int:
    if any(issue.code in READ_ERROR_CODES for issue in report.errors):
        return 2
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = validate_snapshot(
            args.dataset,
            args.snapshot_path,
            report_root=args.report_root,
        )
    except ValidationPipelineError as exc:
        parser.error(str(exc))
    print(json.dumps(_summary(report), sort_keys=True))
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
