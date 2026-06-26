# Sensor Location Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add a local command that runs the existing sensor-location ingestion pipeline and reports the generated snapshot and manifest.

**Architecture:** A testable CLI module owns argument parsing, HTTP-client lifetime, result serialization, and operational exit codes. A one-line repository script calls that module, while the existing ingestion pipeline remains the sole owner of fetch-normalize-write behavior.

**Tech Stack:** Python 3.11+, standard-library \`argparse\` and \`json\`, existing \`httpx\`, \`pytest\`, and Ruff.

---

## File Structure

- Create: \`src/urbanflow/ingestion/sensor_location_cli.py\` — parser, CLI entry point, injected client factory, JSON result rendering, and operational error handling.
- Create: \`scripts/ingest_sensor_locations.py\` — minimal executable wrapper.
- Create: \`tests/unit/ingestion/test_sensor_location_cli.py\` — network-free CLI behavior tests using a fake dataset client.
- Modify: \`.gitignore\` — ignore generated local manifests under \`data/manifests/\`.
- Modify: \`README.md\` — document the command and its default local output paths.

### Task 1: Add failing CLI behavior tests

**Files:**
- Create: \`tests/unit/ingestion/test_sensor_location_cli.py\`
- Reference: \`tests/unit/ingestion/test_sensor_location_pipeline.py\`
- Reference: \`src/urbanflow/ingestion/sensor_location_pipeline.py\`

- [ ] **Step 1: Write fake client data and success tests**

~~~python
import json
from pathlib import Path
from typing import Any

import pytest

from urbanflow.ingestion.melbourne_api import DatasetRecords
from urbanflow.ingestion.sensor_location_cli import main

SOURCE_RECORD = {
    "location_id": 3,
    "sensor_description": "Melbourne Central",
    "sensor_name": "Swa295_T",
    "installation_date": "2009-03-25",
    "status": "A",
    "latitude": -37.81101524,
    "longitude": 144.96429485,
}


class FakeApiClient:
    def __init__(self, records: list[dict[str, Any]], *, total_count: int = 136) -> None:
        self.records = records
        self.total_count = total_count
        self.calls: list[tuple[str, int]] = []

    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords:
        self.calls.append((dataset, limit))
        return DatasetRecords(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=self.total_count,
            records=self.records,
        )


def test_main_uses_default_roots_and_prints_json_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [], api_client_factory=lambda http_client: FakeApiClient([SOURCE_RECORD])
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["record_count"] == 1
    assert summary["source_total_count"] == 136
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "manifests").exists()


def test_main_accepts_output_root_and_page_limit_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root = tmp_path / "custom-raw"
    manifest_root = tmp_path / "custom-manifests"
    fake_client = FakeApiClient([SOURCE_RECORD])

    exit_code = main(
        [
            "--raw-root", str(raw_root),
            "--manifest-root", str(manifest_root),
            "--page-limit", "50",
        ],
        api_client_factory=lambda http_client: fake_client,
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert fake_client.calls == [("pedestrian-counting-system-sensor-locations", 50)]
    assert Path(summary["snapshot_path"]).is_relative_to(raw_root)
    assert Path(summary["manifest_path"]).is_relative_to(manifest_root)
~~~

- [ ] **Step 2: Add expected-operational-error and argument-validation coverage**

~~~python
def test_main_reports_expected_pipeline_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid_record = {**SOURCE_RECORD, "latitude": -120}

    exit_code = main(
        [
            "--raw-root", str(tmp_path / "raw"),
            "--manifest-root", str(tmp_path / "manifests"),
        ],
        api_client_factory=lambda http_client: FakeApiClient([invalid_record]),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: Field 'latitude' must be between -90 and 90" in captured.err
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifests").exists()


def test_main_rejects_non_positive_page_limit(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--page-limit", "0"])

    assert exc_info.value.code == 2
    assert "page limit must be greater than zero" in capsys.readouterr().err
~~~

The injected fake API client keeps the entire file network-free.

- [ ] **Step 3: Run the focused test file to confirm the expected failure**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_location_cli.py -v
~~~

Expected: collection fails with \`ModuleNotFoundError\` because \`urbanflow.ingestion.sensor_location_cli\` does not yet exist.

### Task 2: Implement the testable CLI module

**Files:**
- Create: \`src/urbanflow/ingestion/sensor_location_cli.py\`
- Test: \`tests/unit/ingestion/test_sensor_location_cli.py\`
- Reference: \`src/urbanflow/ingestion/melbourne_api.py\`

- [ ] **Step 1: Add parser and result-summary helpers**

~~~python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from urbanflow.ingestion.melbourne_api import MelbourneApiClient, MelbourneApiError
from urbanflow.ingestion.sensor_location_pipeline import (
    SensorLocationIngestionResult,
    SupportsDatasetRecords,
    ingest_sensor_locations,
)
from urbanflow.ingestion.sensor_locations import SensorLocationParseError


def positive_integer(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("page limit must be an integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("page limit must be greater than zero")
    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Melbourne sensor locations.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--page-limit", type=positive_integer, default=100)
    return parser


def result_summary(result: SensorLocationIngestionResult) -> dict[str, int | str]:
    return {
        "extracted_at": result.extracted_at.isoformat(),
        "manifest_path": result.manifest_path.as_posix(),
        "record_count": result.record_count,
        "snapshot_path": result.snapshot_path.as_posix(),
        "source_dataset": result.source_dataset,
        "source_total_count": result.source_total_count,
        "source_url": result.source_url,
    }
~~~

- [ ] **Step 2: Add the main entry point with a closed HTTP client**

~~~python
def main(
    argv: Sequence[str] | None = None,
    *,
    api_client_factory: Callable[[httpx.Client], SupportsDatasetRecords] = MelbourneApiClient,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        with httpx.Client(timeout=30.0) as http_client:
            result = ingest_sensor_locations(
                api_client=api_client_factory(http_client=http_client),
                raw_root_dir=args.raw_root,
                manifest_root_dir=args.manifest_root,
                page_limit=args.page_limit,
            )
    except (MelbourneApiError, OSError, SensorLocationParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result_summary(result), sort_keys=True))
    return 0
~~~

Do not catch \`Exception\`: unexpected programming failures should retain their traceback. \`FileExistsError\` is already covered by \`OSError\`.

- [ ] **Step 3: Run focused tests and formatting checks**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/ingestion/sensor_location_cli.py tests/unit/ingestion/test_sensor_location_cli.py --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/ingestion/sensor_location_cli.py tests/unit/ingestion/test_sensor_location_cli.py
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_location_cli.py -v
~~~

Expected: Ruff reports no diagnostics and all CLI tests pass.

- [ ] **Step 4: Commit the implementation and tests**

~~~powershell
git add src/urbanflow/ingestion/sensor_location_cli.py tests/unit/ingestion/test_sensor_location_cli.py
git commit -m "feat: add sensor location runner"
~~~

### Task 3: Add the executable wrapper and local-data documentation

**Files:**
- Create: \`scripts/ingest_sensor_locations.py\`
- Modify: \`.gitignore\`
- Modify: \`README.md\`

- [ ] **Step 1: Create the executable wrapper**

~~~python
from urbanflow.ingestion.sensor_location_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
~~~

- [ ] **Step 2: Protect generated manifests from version control**

Add this line beneath the existing \`data/raw/\` rule in \`.gitignore\`:

~~~gitignore
data/manifests/
~~~

- [ ] **Step 3: Document the command in README**

Add this section after \`## Local development\`:

~~~markdown
## Run sensor-location ingestion locally

~~~powershell
python scripts/ingest_sensor_locations.py
~~~

The command fetches the current City of Melbourne sensor-location dataset and
prints a JSON summary. By default it writes an immutable snapshot below
\`data/raw/\` and a matching manifest below \`data/manifests/\`; both are ignored
by Git. Use \`--raw-root\`, \`--manifest-root\`, or \`--page-limit\` to override
the defaults.
~~~

Use backtick fences rather than the displayed tilde fence when editing the README
if nesting needs to be avoided.

- [ ] **Step 4: Run the wrapper's help command**

Run:

~~~powershell
& .\.venv\Scripts\python.exe scripts/ingest_sensor_locations.py --help
~~~

Expected: usage text lists \`--raw-root\`, \`--manifest-root\`, and \`--page-limit\`.

- [ ] **Step 5: Commit wrapper, ignore rule, and documentation**

~~~powershell
git add scripts/ingest_sensor_locations.py .gitignore README.md
git commit -m "docs: document sensor location runner"
~~~

### Task 4: Run the full verification gate and one live smoke command

**Files:**
- Verify: \`src/urbanflow/ingestion/sensor_location_cli.py\`
- Verify: \`scripts/ingest_sensor_locations.py\`
- Verify: \`tests/unit/ingestion/test_sensor_location_cli.py\`

- [ ] **Step 1: Run the repository quality gate**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git status --short --branch
~~~

Expected: Ruff has zero diagnostics, all files are formatted, all tests pass, and only the intentional local branch state is shown.

- [ ] **Step 2: Run a live City of Melbourne smoke command**

Run:

~~~powershell
& .\.venv\Scripts\python.exe scripts/ingest_sensor_locations.py
~~~

Expected: exit code \`0\`, JSON stdout with a nonzero \`record_count\`, and paths under \`data/raw/melbourne/sensor_locations/\` and \`data/manifests/sensor_locations/\`.

- [ ] **Step 3: Verify generated data remains outside Git**

Run:

~~~powershell
git check-ignore data/raw data/manifests
git status --short --ignored data/raw data/manifests
~~~

Expected: both directories are reported as ignored; no generated snapshot or manifest appears in staged changes.

- [ ] **Step 4: Integrate only after a final self-evaluation**

Check this release checklist before merging the local \`codex/\` branch:

- The CLI calls \`ingest_sensor_locations()\` rather than reimplementing it.
- Tests use only fake API clients; the sole live request was a manual smoke run.
- Default output directories are ignored by Git.
- Operational failures return \`1\`; invalid arguments return \`2\`.
- The quality gate has fresh passing output.

If every item is satisfied, merge the local Codex branch into \`main\`, rerun the
quality gate on \`main\`, then push only \`origin/main\`. Never push the local
\`codex/\` branch.
