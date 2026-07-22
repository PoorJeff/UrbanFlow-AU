from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

TIMESTAMP_COLUMNS = ("forecast_origin_at", "target_observed_at")


class SupervisedCsvError(ValueError):
    """Raised when a local supervised CSV cannot be read safely."""


def _parse_offset_aware_timestamp_column(values: pd.Series) -> pd.Series:
    for value in values:
        if pd.isna(value):
            raise ValueError("timestamp is missing")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
    return pd.to_datetime(values, format="mixed", utc=True)


def _read_supervised_csv_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise SupervisedCsvError(f"CSV file does not exist: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SupervisedCsvError(f"could not read supervised CSV: {path}") from exc


def _parse_supervised_csv(source_bytes: bytes, *, path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(source_bytes))
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SupervisedCsvError(f"could not read supervised CSV: {path}") from exc

    for column in TIMESTAMP_COLUMNS:
        if column not in frame.columns:
            continue
        try:
            frame[column] = _parse_offset_aware_timestamp_column(frame[column])
        except (TypeError, ValueError) as exc:
            raise SupervisedCsvError(f"could not parse timestamp column: {column}") from exc
    return frame


def read_supervised_csv(path: Path) -> pd.DataFrame:
    """Read a supervised CSV while preserving the instants of aware timestamps."""
    return _parse_supervised_csv(_read_supervised_csv_bytes(path), path=path)


def read_supervised_csv_snapshot(path: Path) -> tuple[pd.DataFrame, str]:
    """Parse and hash one immutable snapshot of a supervised CSV's bytes."""
    source_bytes = _read_supervised_csv_bytes(path)
    frame = _parse_supervised_csv(source_bytes, path=path)
    return frame, hashlib.sha256(source_bytes).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
