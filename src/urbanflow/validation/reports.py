from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_utc_timestamp(value: datetime) -> str:
    timestamp = value.astimezone(UTC)
    return timestamp.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "rows": list(self.rows),
        }
        return payload


@dataclass(frozen=True)
class ValidationMetric:
    name: str
    value: JsonValue


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    snapshot_path: str
    validated_at: datetime
    row_count: int
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    metrics: tuple[ValidationMetric, ...] = ()
    schema_version: int = field(default=1, init=False)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "snapshot_path": self.snapshot_path,
            "validated_at": format_utc_timestamp(self.validated_at),
            "passed": self.passed,
            "row_count": self.row_count,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "metrics": {metric.name: metric.value for metric in self.metrics},
        }

    def write_json(self, output_path: Path) -> Path:
        if output_path.exists():
            raise FileExistsError(f"Validation report already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path
