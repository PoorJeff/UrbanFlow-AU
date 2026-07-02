from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from urbanflow.modeling.reports import RidgeReportError, render_ridge_evaluation_report


class RidgeReportCliError(ValueError):
    """Raised when the Ridge report CLI receives invalid local inputs."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a Ridge evaluation Markdown report from a JSON summary."
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to a Ridge evaluation JSON summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to the input path with .md suffix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser


def _read_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RidgeReportCliError(f"summary JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RidgeReportCliError(f"could not read summary JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RidgeReportCliError(f"invalid summary JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RidgeReportCliError("summary JSON must contain an object")
    return payload


def _resolve_output_path(summary_json: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return summary_json.with_suffix(".md")


def render_report_file(
    summary_json: Path,
    *,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    destination = _resolve_output_path(summary_json, output_path)
    if destination.exists() and not force:
        raise RidgeReportCliError(f"output file already exists: {destination}")

    summary = _read_summary_json(summary_json)
    markdown = render_ridge_evaluation_report(summary)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        raise RidgeReportCliError(f"could not write report: {destination}") from exc
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = render_report_file(
            args.summary_json,
            output_path=args.output,
            force=args.force,
        )
    except (RidgeReportCliError, RidgeReportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"output_path": str(output_path)}, sort_keys=True))
    return 0
