from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MELBOURNE_TIME_ZONE = ZoneInfo("Australia/Melbourne")
MAX_HISTORY_ELAPSED = timedelta(days=31)


def melbourne_now() -> datetime:
    return datetime.now(MELBOURNE_TIME_ZONE)


def format_melbourne_timestamp(value: datetime) -> str:
    if not _is_offset_aware(value):
        raise ValueError("Timestamp must be offset-aware.")
    return value.astimezone(MELBOURNE_TIME_ZONE).strftime("%d %b %Y, %H:%M %Z")


def local_midnight(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=MELBOURNE_TIME_ZONE)


def validate_history_interval(start: datetime, end: datetime) -> str | None:
    if not _is_offset_aware(start) or not _is_offset_aware(end):
        return "History start and end must include a time zone."

    start_in_utc = start.astimezone(UTC)
    end_in_utc = end.astimezone(UTC)
    if start_in_utc >= end_in_utc:
        return "History start must be earlier than history end."
    if end_in_utc - start_in_utc > MAX_HISTORY_ELAPSED:
        return "History interval cannot exceed 31 elapsed days."
    return None


def _is_offset_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None
