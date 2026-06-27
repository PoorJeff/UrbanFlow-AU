from datetime import UTC, datetime

from urbanflow.validation.hourly_counts import validate_hourly_counts_snapshot

HEADER = (
    "id,location_id,sensing_date,hourday,direction_1,direction_2,"
    "pedestriancount,sensor_name,location\n"
)


def _write_csv(tmp_path, rows):
    path = tmp_path / "records.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def test_hourly_count_snapshot_passes_and_records_metrics(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        [
            'a,1,2025-01-01,0,2,3,5,Sensor A,"-37.81,144.96"\n',
            'b,1,2025-01-01,1,1,1,2,Sensor A,"-37.81,144.96"\n',
            'c,2,2025-01-02,23,4,6,10,Sensor B,"-37.82,144.97"\n',
        ],
    )

    report = validate_hourly_counts_snapshot(
        snapshot_path,
        datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is True
    payload = report.to_dict()
    assert payload["metrics"]["row_count"] == 3
    assert payload["metrics"]["sensor_count"] == 2
    assert payload["metrics"]["date_range"] == {"start": "2025-01-01", "end": "2025-01-02"}
    assert payload["metrics"]["hour_distribution"]["0"] == 1


def test_hourly_count_snapshot_warns_for_duplicate_id(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        [
            'a,1,2025-01-01,0,2,3,5,Sensor A,"-37.81,144.96"\n',
            'a,1,2025-01-01,1,1,1,2,Sensor A,"-37.81,144.96"\n',
        ],
    )

    report = validate_hourly_counts_snapshot(snapshot_path)

    assert report.passed is True
    assert any(issue.code == "DUPLICATE_SOURCE_ID" for issue in report.warnings)


def test_hourly_count_snapshot_fails_for_hour_range_and_direction_total(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        ['a,1,2025-01-01,24,2,3,9,Sensor A,"-37.81,144.96"\n'],
    )

    report = validate_hourly_counts_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "SCHEMA_INVALID" for issue in report.errors)
    assert any(issue.code == "DIRECTION_TOTAL_MISMATCH" for issue in report.errors)


def test_hourly_count_snapshot_warns_for_duplicate_sensor_hour_and_incomplete_coverage(
    tmp_path,
):
    snapshot_path = _write_csv(
        tmp_path,
        [
            'a,1,2025-01-01,0,2,3,5,Sensor A,"-37.81,144.96"\n',
            'b,1,2025-01-01,0,1,1,2,Sensor A,"-37.81,144.96"\n',
        ],
    )

    report = validate_hourly_counts_snapshot(snapshot_path)

    assert report.passed is True
    assert any(issue.code == "DUPLICATE_SENSOR_HOUR" for issue in report.warnings)
    assert any(issue.code == "INCOMPLETE_HOUR_COVERAGE" for issue in report.warnings)
