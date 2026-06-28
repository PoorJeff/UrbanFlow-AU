from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")


def parse_source_date(value: str | date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


def melbourne_observed_at(sensing_date: str | date, hourday: str | int) -> datetime:
    source_date = parse_source_date(sensing_date)
    source_hour = int(hourday)
    if source_hour < 0 or source_hour > 23:
        raise ValueError(f"hourday must be between 0 and 23: {hourday}")
    return datetime.combine(source_date, time(hour=source_hour), tzinfo=MELBOURNE_TZ)
