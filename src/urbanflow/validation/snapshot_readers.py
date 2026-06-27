from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class SnapshotReadError(Exception):
    """Raised when a local snapshot cannot be loaded for validation."""


def _ensure_existing_file(snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        raise SnapshotReadError(f"Snapshot file does not exist: {snapshot_path}")
    if not snapshot_path.is_file():
        raise SnapshotReadError(f"Snapshot path is not a file: {snapshot_path}")


def read_sensor_locations_snapshot(snapshot_path: Path) -> pd.DataFrame:
    _ensure_existing_file(snapshot_path)
    try:
        payload: Any = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotReadError(f"Could not parse JSON snapshot: {snapshot_path}") from exc
    if not isinstance(payload, list):
        raise SnapshotReadError("JSON snapshot must contain a list of records")
    return pd.DataFrame.from_records(payload)


def read_hourly_counts_snapshot(snapshot_path: Path) -> pd.DataFrame:
    _ensure_existing_file(snapshot_path)
    try:
        return pd.read_csv(snapshot_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise SnapshotReadError(f"CSV snapshot is empty: {snapshot_path}") from exc
    except UnicodeDecodeError as exc:
        raise SnapshotReadError(f"Could not decode CSV snapshot: {snapshot_path}") from exc
