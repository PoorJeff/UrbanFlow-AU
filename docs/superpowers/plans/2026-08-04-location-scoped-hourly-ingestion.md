# Location-Scoped Hourly-Count Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Let a user export a bounded hourly-count snapshot for one Melbourne sensor through the existing ingestion CLI, while preserving exact source-query provenance in the result and manifest.

**Architecture:** Extend the existing date-range domain helper with an optional, strictly validated location identifier. Thread that identifier through the pipeline and its existing source query so the count and CSV export always use one identical Socrata where clause. Expose the capability through one CLI option and document the safe local workflow. Do not add database reads, model loading, prediction logic, or custom raw where clauses.

**Tech Stack:** Python 3.12, argparse, existing Melbourne Socrata client, CSV snapshot storage, schema-v1 JSON manifests, pytest, Ruff.

## Global Constraints

- Work only in D:/Github项目/UrbanFlow-AU/.worktrees/location-scoped-hourly-ingestion on branch codex/location-scoped-hourly-ingestion.
- Preserve current date-only behaviour byte-for-byte at the public query-contract level.
- Accept location identifiers only as positive integers. Reject booleans, non-integers, zero, and negatives with the domain message "location_id must be a positive integer".
- Construct no source query in the CLI. The domain helper remains the single query-construction boundary.
- Use the same generated where clause for source counting and CSV export.
- Validate invalid location identifiers before network calls, storage writes, or directory creation.
- Store sensor_filter as "all" when unfiltered, otherwise as an object containing the selected location_id.
- Do not implement an arbitrary --where option, automatic sensor discovery, batching, database ingestion, model training, or forecast serving in this slice.
- Do not call the live Melbourne API in tests or quality gates.
- Use apply_patch for all source and documentation edits.

---

## Task 1: Make the isolated worktree environment executable before TDD

**Files:**

- Verify only. Do not commit .venv, downloaded data, or package caches.

- [ ] **Step 1: Check whether the local virtual environment imports this worktree.**

Run:

~~~powershell
./.venv/Scripts/python.exe -c "import urbanflow; print(urbanflow.__file__)"
~~~

Expected: the printed path starts with D:/Github项目/UrbanFlow-AU/.worktrees/location-scoped-hourly-ingestion/src.

- [ ] **Step 2: Repair the editable install only when the import fails or resolves outside this worktree.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pip install --prefer-binary -e ".[dev]"
./.venv/Scripts/python.exe -c "import urbanflow; print(urbanflow.__file__)"
~~~

Expected: the second command imports urbanflow from this worktree. If installation fails, capture the first actionable error and stop before running a misleading test command. Do not use PYTHONPATH as a workaround.

- [ ] **Step 3: Confirm the basic local test runner is available.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest --version
./.venv/Scripts/python.exe -m ruff --version
~~~

Expected: both commands succeed using tools installed in this isolated environment.

---

## Task 2: Add a validated optional location filter to the hourly-count query domain

**Files:**

- Modify: src/urbanflow/ingestion/hourly_counts.py
- Modify: tests/unit/ingestion/test_hourly_counts.py

- [ ] **Step 1: Write the focused failing domain tests.**

Add coverage that constructs HourlyCountDateRange for 2025-01-01 through 2025-05-31 and verifies:

1. build_hourly_counts_where(date_range, location_id=101) returns exactly:
   sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-05-31' AND location_id = 101
2. build_hourly_counts_where(date_range) still returns only the existing date predicate.
3. validate_location_id(None) returns None and validate_location_id(101) returns 101.
4. validate_location_id rejects 0, -1, True, False, "101", and 101.0 by raising HourlyCountIngestionError with a message containing "positive integer".

Suggested test shape:

~~~python
@pytest.mark.parametrize("location_id", [0, -1, True, False, "101", 101.0])
def test_validate_location_id_rejects_invalid_values(location_id: object) -> None:
    with pytest.raises(HourlyCountIngestionError, match="positive integer"):
        validate_location_id(location_id)
~~~

- [ ] **Step 2: Run the focused test to establish RED.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest tests/unit/ingestion/test_hourly_counts.py -q
~~~

Expected: the new tests fail because validate_location_id and the location_id keyword parameter do not exist yet. Existing date-range tests should still pass.

- [ ] **Step 3: Implement the smallest domain extension.**

In hourly_counts.py:

1. Add validate_location_id(location_id: int | None) -> int | None.
2. Return None unchanged.
3. Check type(location_id) is int rather than relying only on isinstance so bool is rejected.
4. Raise HourlyCountIngestionError("location_id must be a positive integer") for every invalid value.
5. Change build_hourly_counts_where to accept a keyword-only location_id: int | None = None.
6. Build the current date expression first. Validate the optional identifier once and append " AND location_id = {value}" only when it is present.

Keep HourlyCountDateRange and its validation unchanged.

- [ ] **Step 4: Run the focused test to establish GREEN.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest tests/unit/ingestion/test_hourly_counts.py -q
./.venv/Scripts/python.exe -m ruff check src/urbanflow/ingestion/hourly_counts.py tests/unit/ingestion/test_hourly_counts.py
~~~

Expected: all focused tests pass and Ruff reports no diagnostics.

- [ ] **Step 5: Commit the standalone domain change.**

~~~powershell
git add src/urbanflow/ingestion/hourly_counts.py tests/unit/ingestion/test_hourly_counts.py
git diff --cached --check
git commit -m "feat(ingestion): add location-scoped source query"
~~~

## Task 3: Carry the filter through the hourly-count ingestion pipeline and manifest

**Files:**

- Modify: src/urbanflow/ingestion/hourly_count_pipeline.py
- Modify: tests/unit/ingestion/test_hourly_count_pipeline.py

- [ ] **Step 1: Write focused failing pipeline tests using the existing fake API client.**

Add a scoped-success test with a stable extracted_at value and a temporary raw/manifests root. It must verify all of the following after calling ingest_hourly_counts with location_id=51:

1. The fake client receives the exact same scoped where clause in its count call and CSV export call.
2. HourlyCountIngestionResult.location_id equals 51.
3. The parsed manifest has sensor_filter equal to {"location_id": 51}.
4. The manifest source_where equals the expected scoped clause.
5. The stored snapshot and manifest paths exist.

Add an invalid-input test calling ingest_hourly_counts with location_id=0. Verify it raises HourlyCountIngestionError, the fake API client has no calls, and no raw or manifest output is created.

Do not weaken the existing unfiltered assertions. Add or retain an assertion that unfiltered manifests continue to use sensor_filter: "all".

- [ ] **Step 2: Run the focused pipeline test to establish RED.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest tests/unit/ingestion/test_hourly_count_pipeline.py -q
~~~

Expected: the new scoped test fails because the pipeline has no location_id argument/result field, and the invalid-input test fails before any I/O validation exists.

- [ ] **Step 3: Implement propagation with one shared query value.**

In hourly_count_pipeline.py:

1. Import validate_location_id alongside the existing date-range/query helpers.
2. Add location_id: int | None as a final field on HourlyCountIngestionResult.
3. Add location_id: int | None = None as a keyword-only ingest_hourly_counts parameter, after existing optional arguments to preserve call compatibility.
4. At the start of ingest_hourly_counts, validate the value before asking the API client for metadata, counting records, exporting CSV, creating output directories, or writing files.
5. Build source_where exactly once with build_hourly_counts_where(date_range, location_id=validated_location_id).
6. Pass that one source_where value to both count_records and export_csv.
7. Emit "sensor_filter": "all" for an absent identifier and "sensor_filter": {"location_id": validated_location_id} otherwise.
8. Return the validated identifier in HourlyCountIngestionResult.

Do not change the snapshot naming layout, collision behaviour, manifest schema version, selected column list, or source URL construction.

- [ ] **Step 4: Run focused regression tests to establish GREEN.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest tests/unit/ingestion/test_hourly_count_pipeline.py tests/unit/ingestion/test_hourly_counts.py -q
./.venv/Scripts/python.exe -m ruff check src/urbanflow/ingestion/hourly_count_pipeline.py tests/unit/ingestion/test_hourly_count_pipeline.py
~~~

Expected: all focused pipeline and domain tests pass without any network access.

- [ ] **Step 5: Commit the pipeline and provenance change.**

~~~powershell
git add src/urbanflow/ingestion/hourly_count_pipeline.py tests/unit/ingestion/test_hourly_count_pipeline.py
git diff --cached --check
git commit -m "feat(ingestion): record scoped snapshot provenance"
~~~

## Task 4: Expose location-scoped export through the CLI and local documentation

**Files:**

- Modify: src/urbanflow/ingestion/hourly_count_cli.py
- Modify: tests/unit/ingestion/test_hourly_count_cli.py
- Modify: README.md

- [ ] **Step 1: Write focused failing CLI tests.**

Extend the existing parser and main-path tests to prove:

1. --location-id 101 parses to the integer 101.
2. --year 2025 --location-id 101 is accepted together and reaches the ingestion call with the expected full-year date range and location_id=101.
3. --location-id 0, --location-id -1, and --location-id abc fail during argparse parsing with exit code 2.
4. A successful mocked CLI invocation passes location_id=101 to ingest_hourly_counts and includes "location_id": 101 in its one-line JSON output.
5. The existing unfiltered successful invocation includes "location_id": null in its JSON output.
6. Existing pipeline/network failure behaviour remains exit code 1 rather than being converted to parser errors.

Use the existing monkeypatch/fake-result conventions in the file. Do not invoke a live export.

- [ ] **Step 2: Run the focused CLI test to establish RED.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest tests/unit/ingestion/test_hourly_count_cli.py -q
~~~

Expected: tests referring to --location-id and the JSON field fail.

- [ ] **Step 3: Implement strict parser conversion and result reporting.**

In hourly_count_cli.py:

1. Add a small positive_location_id(value: str) -> int converter.
2. Convert to int, then call the domain validator. Convert ValueError and HourlyCountIngestionError into argparse.ArgumentTypeError using the domain message.
3. Add parser option --location-id with metavar LOCATION_ID and this converter.
4. Pass args.location_id to ingest_hourly_counts.
5. Add "location_id": result.location_id to result_summary.
6. Broaden result_summary's return annotation to dict[str, object] if needed because the JSON field can be an integer or None.

Keep date selection validation, error JSON shape, and exit-code conventions unchanged.

- [ ] **Step 4: Update the README for a bounded local export.**

In the existing "Run hourly-count ingestion locally" section:

1. Retain the current year-wide command as the unfiltered option.
2. Add a copyable example using --start-date 2025-01-01 --end-date 2025-05-31 --location-id 101.
3. Explain that --location-id restricts the source query to exactly one sensor while the date arguments bound the historical window.
4. State plainly that a successful export proves only snapshot acquisition and provenance; it does not prove 168 contiguous hourly observations or a valid forecast.

Do not document the feature as a production prediction workflow.

- [ ] **Step 5: Run focused CLI/documentation verification to establish GREEN.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m pytest tests/unit/ingestion/test_hourly_count_cli.py tests/unit/ingestion/test_hourly_count_pipeline.py tests/unit/ingestion/test_hourly_counts.py -q
./.venv/Scripts/python.exe -m ruff check src/urbanflow/ingestion/hourly_count_cli.py tests/unit/ingestion/test_hourly_count_cli.py
./.venv/Scripts/python.exe scripts/ingest_hourly_counts.py --help
~~~

Expected: all focused tests pass, lint passes, and help lists --location-id. The help command must not contact the live API.

- [ ] **Step 6: Commit the CLI and README update.**

~~~powershell
git add src/urbanflow/ingestion/hourly_count_cli.py tests/unit/ingestion/test_hourly_count_cli.py README.md
git diff --cached --check
git commit -m "feat(ingestion): expose location-scoped hourly exports"
~~~

## Task 5: Run the full acceptance gate and integrate only after a clean review

**Files:**

- Verify only.
- Do not commit .venv or generated raw/manifests data.

- [ ] **Step 1: Confirm the worktree virtual environment still imports this worktree's source.**

Run:

~~~powershell
./.venv/Scripts/python.exe -c "import urbanflow; print(urbanflow.__file__)"
~~~

Expected: the printed path starts with D:/Github项目/UrbanFlow-AU/.worktrees/location-scoped-hourly-ingestion/src.

If the import fails or resolves elsewhere, return to Task 1 and repair only the isolated environment:

~~~powershell
./.venv/Scripts/python.exe -m pip install --prefer-binary -e ".[dev]"
./.venv/Scripts/python.exe -c "import urbanflow; print(urbanflow.__file__)"
~~~

If installation fails, capture the first actionable error and stop before running a misleading quality gate. Do not use PYTHONPATH as a workaround.

- [ ] **Step 2: Run formatter, lint, and the full test suite from the isolated worktree.**

Run:

~~~powershell
./.venv/Scripts/python.exe -m ruff format --check .
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m pytest -q
~~~

Expected: formatter check and lint pass; all tests pass without network, PostgreSQL, MLflow, model files, or generated data. Record the actual test count and any pre-existing warning total in the handoff.

- [ ] **Step 3: Review the final diff for scope and contract correctness.**

Run:

~~~powershell
git diff main...HEAD --check
git diff main...HEAD -- src/urbanflow/ingestion/hourly_counts.py src/urbanflow/ingestion/hourly_count_pipeline.py src/urbanflow/ingestion/hourly_count_cli.py tests/unit/ingestion README.md
git status --short --branch
~~~

Verify manually:

1. Date-only public behaviour and unfiltered provenance remain intact.
2. The scoped where string has date predicates first and the location predicate last.
3. Invalid values do not cross the API/storage boundary.
4. Count/export receive identical where values.
5. The CLI emits valid JSON with location_id as an integer or null.
6. No generated snapshot, manifest, virtual-environment, or unrelated refactor is included.

- [ ] **Step 4: Fast-forward integrate only after the branch is clean and all gates pass.**

From the primary repository worktree D:/Github项目/UrbanFlow-AU, inspect main and integrate using the repository's documented fast-forward workflow. Re-run the same full quality gate on main after merging. Do not push to origin unless the user explicitly authorizes a push.

- [ ] **Step 5: Report the verified handoff.**

Report the branch commit range, exact verification commands/results, changed behavioural contract, and explicit next decision: use the location-scoped export to create an offline supervised CSV, or begin a separate slice for a real PostgreSQL/model-backed forecast provider.
