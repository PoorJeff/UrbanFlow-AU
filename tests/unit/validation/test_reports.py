import json
from datetime import UTC, datetime

import pytest

from urbanflow.validation.reports import (
    ValidationIssue,
    ValidationMetric,
    ValidationReport,
)


def test_validation_report_serializes_stable_shape(tmp_path):
    report = ValidationReport(
        dataset="sensor_locations",
        snapshot_path="data/raw/melbourne/sensor_locations/example/records.json",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
        row_count=2,
        errors=(
            ValidationIssue(
                code="DUPLICATE_LOCATION_ID",
                message="location_id values must be unique",
                column="location_id",
                rows=(1,),
            ),
        ),
        warnings=(
            ValidationIssue(
                code="NULL_INSTALLATION_DATE",
                message="installation_date is null for 1 row",
                column="installation_date",
            ),
        ),
        metrics=(
            ValidationMetric(name="sensor_count", value=2),
            ValidationMetric(name="status_distribution", value={"A": 2}),
        ),
    )

    payload = report.to_dict()

    assert payload == {
        "schema_version": 1,
        "dataset": "sensor_locations",
        "snapshot_path": "data/raw/melbourne/sensor_locations/example/records.json",
        "validated_at": "2026-06-27T12:00:00Z",
        "passed": False,
        "row_count": 2,
        "errors": [
            {
                "code": "DUPLICATE_LOCATION_ID",
                "message": "location_id values must be unique",
                "column": "location_id",
                "rows": [1],
            }
        ],
        "warnings": [
            {
                "code": "NULL_INSTALLATION_DATE",
                "message": "installation_date is null for 1 row",
                "column": "installation_date",
                "rows": [],
            }
        ],
        "metrics": {
            "sensor_count": 2,
            "status_distribution": {"A": 2},
        },
    }

    output_path = tmp_path / "quality" / "sensor_locations" / "report.json"
    report.write_json(output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_validation_report_refuses_to_overwrite(tmp_path):
    report = ValidationReport(
        dataset="hourly_counts",
        snapshot_path="records.csv",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
        row_count=0,
    )
    output_path = tmp_path / "report.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Validation report already exists"):
        report.write_json(output_path)
