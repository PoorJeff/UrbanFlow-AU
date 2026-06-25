from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def format_extracted_at(extracted_at: datetime) -> str:
    return extracted_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_json_snapshot(
    *,
    records: list[dict[str, Any]],
    root_dir: Path,
    dataset: str,
    extracted_at: datetime,
) -> Path:
    timestamp = format_extracted_at(extracted_at)
    snapshot_path = root_dir / "melbourne" / dataset / f"extracted_at={timestamp}" / "records.json"
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_text = json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True)
    snapshot_path.write_text(f"{snapshot_text}\n", encoding="utf-8")
    return snapshot_path
