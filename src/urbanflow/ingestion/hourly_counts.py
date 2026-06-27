from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

HOURLY_COUNTS_SOURCE_DATASET = "pedestrian-counting-system-monthly-counts-per-hour"
HOURLY_COUNTS_SNAPSHOT_DATASET = "hourly_counts"
HOURLY_COUNT_COLUMNS = (
    "id",
    "location_id",
    "sensing_date",
    "hourday",
    "direction_1",
    "direction_2",
    "pedestriancount",
    "sensor_name",
    "location",
)


class HourlyCountIngestionError(ValueError):
    """Raised when hourly-count ingestion inputs or exports are unusable."""


@dataclass(frozen=True)
class HourlyCountDateRange:
    start_date: date
    end_date: date


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HourlyCountIngestionError(f"Date '{value}' must use YYYY-MM-DD format") from exc


def validate_date_range(start_date: date, end_date: date) -> HourlyCountDateRange:
    if start_date > end_date:
        raise HourlyCountIngestionError("start_date must be on or before end_date")
    return HourlyCountDateRange(start_date=start_date, end_date=end_date)


def year_date_range(year: int) -> HourlyCountDateRange:
    if year < 1900:
        raise HourlyCountIngestionError("year must be 1900 or later")
    return HourlyCountDateRange(
        start_date=date(year, 1, 1),
        end_date=date(year, 12, 31),
    )


def build_hourly_counts_where(date_range: HourlyCountDateRange) -> str:
    return (
        f"sensing_date >= date'{date_range.start_date.isoformat()}' "
        f"AND sensing_date <= date'{date_range.end_date.isoformat()}'"
    )


def count_csv_data_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if not header:
                raise HourlyCountIngestionError("CSV export is missing a header row")
            return sum(1 for _row in reader)
    except OSError as exc:
        raise HourlyCountIngestionError(f"CSV export could not be read: {exc}") from exc
