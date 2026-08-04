from datetime import date

import pytest

from urbanflow.ingestion.hourly_counts import (
    HOURLY_COUNT_COLUMNS,
    HourlyCountDateRange,
    HourlyCountIngestionError,
    build_hourly_counts_where,
    count_csv_data_rows,
    parse_iso_date,
    validate_date_range,
    validate_location_id,
    year_date_range,
)


def test_year_date_range_expands_to_calendar_year() -> None:
    date_range = year_date_range(2025)

    assert date_range == HourlyCountDateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    )


def test_parse_iso_date_rejects_invalid_input() -> None:
    with pytest.raises(HourlyCountIngestionError, match="YYYY-MM-DD"):
        parse_iso_date("2025/01/01")


def test_validate_date_range_rejects_reversed_dates() -> None:
    with pytest.raises(HourlyCountIngestionError, match="start_date"):
        validate_date_range(date(2025, 1, 2), date(2025, 1, 1))


def test_build_hourly_counts_where_uses_inclusive_dates() -> None:
    date_range = HourlyCountDateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    assert (
        build_hourly_counts_where(date_range)
        == "sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-01-31'"
    )


def test_build_hourly_counts_where_limits_to_location() -> None:
    date_range = HourlyCountDateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 5, 31),
    )

    assert (
        build_hourly_counts_where(date_range, location_id=101)
        == "sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-05-31' "
        "AND location_id = 101"
    )


def test_validate_location_id_returns_optional_positive_integer() -> None:
    assert validate_location_id(None) is None
    assert validate_location_id(101) == 101


@pytest.mark.parametrize("location_id", [0, -1, True, False, "101", 101.0])
def test_validate_location_id_rejects_invalid_values(location_id: object) -> None:
    with pytest.raises(HourlyCountIngestionError) as exc_info:
        validate_location_id(location_id)

    assert str(exc_info.value) == "location_id must be a positive integer"


def test_count_csv_data_rows_counts_rows_after_header(tmp_path) -> None:
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "id,location_id,sensing_date\n51120250101,51,2025-01-01\n45620250101,45,2025-01-01\n",
        encoding="utf-8",
    )

    assert count_csv_data_rows(csv_path) == 2


def test_count_csv_data_rows_rejects_empty_file(tmp_path) -> None:
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(HourlyCountIngestionError, match="header"):
        count_csv_data_rows(csv_path)


def test_hourly_count_columns_preserve_source_order() -> None:
    assert HOURLY_COUNT_COLUMNS == (
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
