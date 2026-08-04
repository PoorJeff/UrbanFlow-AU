from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from urbanflow.database.time import melbourne_observed_at
from urbanflow.features.supervised import build_supervised_frame
from urbanflow.modeling.lightgbm_artifact import HolidayCalendar
from urbanflow.modeling.supervised_csv import SupervisedCsvError, read_supervised_csv
from urbanflow.validation.hourly_counts import validate_hourly_counts_frame

_MANIFEST_MISMATCH_MESSAGE = "hourly-count manifest does not match snapshot"
_MANIFEST_FIELDS = {
    "schema_version",
    "dataset",
    "source_url",
    "extracted_at",
    "record_count",
    "source_total_count",
    "snapshot_path",
    "snapshot_sha256",
}
_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SupervisedSnapshotBuildError(ValueError):
    """Raised when a snapshot cannot safely become supervised rows."""


class SupervisedSnapshotWriteError(RuntimeError):
    """Raised when valid supervised rows cannot be written safely."""


@dataclass(frozen=True, slots=True)
class SupervisedSnapshotBuildResult:
    snapshot_path: Path
    manifest_path: Path
    output_path: Path
    source_row_count: int
    supervised_row_count: int
    training_row_count: int
    validation_warning_count: int
    snapshot_sha256: str


def _manifest_mismatch() -> SupervisedSnapshotBuildError:
    return SupervisedSnapshotBuildError(_MANIFEST_MISMATCH_MESSAGE)


def _load_hourly_count_manifest(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _manifest_mismatch() from exc


def _read_hourly_counts_snapshot_once(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        source_bytes = path.read_bytes()
        frame = pd.read_csv(BytesIO(source_bytes), dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SupervisedSnapshotBuildError(f"could not read hourly-count snapshot: {path}") from exc
    return frame, hashlib.sha256(source_bytes).hexdigest()


def _verify_hourly_count_manifest(
    manifest: object, *, snapshot_sha256: str, source_row_count: int
) -> None:
    if not isinstance(manifest, dict) or not _MANIFEST_FIELDS.issubset(manifest):
        raise _manifest_mismatch()
    try:
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError
        if manifest["dataset"] != "hourly_counts":
            raise ValueError
        if not isinstance(manifest["source_url"], str) or not manifest["source_url"].strip():
            raise ValueError
        if not isinstance(manifest["snapshot_path"], str) or not manifest["snapshot_path"].strip():
            raise ValueError
        extracted_at = manifest["extracted_at"]
        if not isinstance(extracted_at, str) or len(extracted_at) != 16:
            raise ValueError
        datetime.strptime(extracted_at, "%Y%m%dT%H%M%SZ")
        for field in ("record_count", "source_total_count"):
            if type(manifest[field]) is not int or manifest[field] < 0:
                raise ValueError
        stored_sha256 = manifest["snapshot_sha256"]
        if not isinstance(stored_sha256, str) or not _LOWERCASE_SHA256.fullmatch(stored_sha256):
            raise ValueError
        if manifest["record_count"] != source_row_count or stored_sha256 != snapshot_sha256:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise _manifest_mismatch() from exc


def _observations_from_hourly_count_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [int(value) for value in frame["location_id"]],
            "observed_at": [
                melbourne_observed_at(sensing_date, hourday)
                for sensing_date, hourday in zip(
                    frame["sensing_date"], frame["hourday"], strict=True
                )
            ],
            "pedestrian_count": [int(value) for value in frame["pedestriancount"]],
        }
    )


def _require_calendar_coverage(supervised: pd.DataFrame, holiday_calendar: HolidayCalendar) -> None:
    target_dates = {
        pd.Timestamp(value).tz_convert("Australia/Melbourne").date()
        for value in supervised["target_observed_at"]
    }
    if not all(holiday_calendar.contains(value) for value in target_dates):
        raise SupervisedSnapshotBuildError("holiday calendar does not cover generated target dates")


def _write_new_supervised_csv(supervised: pd.DataFrame, output_path: Path) -> None:
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{output_path.name}-", dir=output_path.parent
        )
        temporary_path = Path(raw_temporary_path)
        os.close(descriptor)
        descriptor = None
        supervised.to_csv(temporary_path, index=False)
        read_supervised_csv(temporary_path)
        os.link(temporary_path, output_path)
        temporary_path.unlink()
        temporary_path = None
    except FileExistsError as exc:
        raise SupervisedSnapshotBuildError(
            f"supervised CSV output already exists: {output_path}"
        ) from exc
    except (OSError, SupervisedCsvError) as exc:
        raise SupervisedSnapshotWriteError(
            f"could not write supervised CSV: {output_path}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None and os.path.lexists(temporary_path):
            try:
                temporary_path.unlink()
            except OSError:
                pass


def build_supervised_csv_from_hourly_snapshot(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    holiday_calendar: HolidayCalendar,
) -> SupervisedSnapshotBuildResult:
    """Build one direct-horizon supervised CSV from a verified local snapshot."""
    snapshot_path = Path(snapshot_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if os.path.lexists(output_path):
        raise SupervisedSnapshotBuildError(f"supervised CSV output already exists: {output_path}")
    manifest = _load_hourly_count_manifest(manifest_path)
    frame, snapshot_sha256 = _read_hourly_counts_snapshot_once(snapshot_path)
    _verify_hourly_count_manifest(
        manifest, snapshot_sha256=snapshot_sha256, source_row_count=len(frame)
    )
    if frame.empty:
        raise SupervisedSnapshotBuildError("hourly-count snapshot contains no rows")
    report = validate_hourly_counts_frame(frame, snapshot_path=snapshot_path)
    if report.errors:
        codes = ", ".join(issue.code for issue in report.errors)
        raise SupervisedSnapshotBuildError(f"hourly-count snapshot validation failed: {codes}")
    if any(issue.code == "DUPLICATE_SENSOR_HOUR" for issue in report.warnings):
        raise SupervisedSnapshotBuildError(
            "hourly-count snapshot contains duplicate sensor-hour rows"
        )
    observations = _observations_from_hourly_count_frame(frame)
    supervised = build_supervised_frame(
        observations, public_holidays=holiday_calendar.public_holidays
    )
    _require_calendar_coverage(supervised, holiday_calendar)
    _write_new_supervised_csv(supervised, output_path)
    return SupervisedSnapshotBuildResult(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        output_path=output_path,
        source_row_count=len(frame),
        supervised_row_count=len(supervised),
        training_row_count=int((~supervised["target_missing"]).sum()),
        validation_warning_count=len(report.warnings),
        snapshot_sha256=snapshot_sha256,
    )
