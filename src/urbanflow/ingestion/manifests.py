from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from urbanflow.ingestion.snapshots import format_extracted_at


def write_manifest(
    *,
    root_dir: Path,
    dataset: str,
    source_url: str,
    extracted_at: datetime,
    record_count: int,
    source_total_count: int,
    snapshot_path: Path,
) -> Path:
    timestamp = format_extracted_at(extracted_at)
    manifest_path = root_dir / dataset / f"{timestamp}.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "source_url": source_url,
        "extracted_at": timestamp,
        "record_count": record_count,
        "source_total_count": source_total_count,
        "snapshot_path": snapshot_path.as_posix(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_path.write_text(f"{manifest_text}\n", encoding="utf-8")
    return manifest_path
