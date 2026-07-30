from datetime import UTC, date, datetime, timedelta

from urbanflow.dashboard.time_utils import (
    MELBOURNE_TIME_ZONE,
    format_melbourne_timestamp,
    local_midnight,
    melbourne_now,
    validate_history_interval,
)


def test_melbourne_now_is_offset_aware() -> None:
    current = melbourne_now()

    assert current.tzinfo is MELBOURNE_TIME_ZONE
    assert current.utcoffset() is not None


def test_format_melbourne_timestamp_converts_without_changing_the_instant() -> None:
    value = datetime(2026, 7, 12, 10, 30, tzinfo=UTC)

    assert format_melbourne_timestamp(value) == "12 Jul 2026, 20:30 AEST"


def test_local_midnight_is_offset_aware_in_melbourne() -> None:
    midnight = local_midnight(date(2026, 1, 12))

    assert midnight == datetime(2026, 1, 12, tzinfo=MELBOURNE_TIME_ZONE)
    assert midnight.utcoffset() == timedelta(hours=11)


def test_validate_history_interval_requires_aware_datetimes() -> None:
    start = datetime(2026, 7, 12, 8)
    end = datetime(2026, 7, 12, 9)

    assert validate_history_interval(start, end) == (
        "History start and end must include a time zone."
    )


def test_validate_history_interval_requires_start_before_end() -> None:
    value = datetime(2026, 7, 12, 8, tzinfo=UTC)

    assert validate_history_interval(value, value) == (
        "History start must be earlier than history end."
    )
    assert validate_history_interval(value + timedelta(hours=1), value) == (
        "History start must be earlier than history end."
    )


def test_validate_history_interval_uses_elapsed_time_across_daylight_saving() -> None:
    start = datetime(2026, 9, 3, 14, tzinfo=UTC).astimezone(MELBOURNE_TIME_ZONE)
    exact_limit = (start.astimezone(UTC) + timedelta(days=31)).astimezone(MELBOURNE_TIME_ZONE)

    assert start.utcoffset() != exact_limit.utcoffset()
    assert validate_history_interval(start, exact_limit) is None
    assert validate_history_interval(start, exact_limit + timedelta(microseconds=1)) == (
        "History interval cannot exceed 31 elapsed days."
    )
