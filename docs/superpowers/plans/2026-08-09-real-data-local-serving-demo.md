# Real-Data Local Serving Demo Implementation Plan

**Goal:** Deliver a reproducible local forecast demonstration for Melbourne
`location_id=1` using five complete months of official hourly observations.

## Task 1: Extend the Prefect flow with location scope

1. Add failing CLI tests for a valid `--location-id`, invalid identifiers, and
   unchanged no-filter behavior.
2. Add failing flow tests proving the value reaches the hourly ingestion task
   and both count/export operations while sensor-location ingestion is unchanged.
3. Add the smallest optional parameter chain through the CLI, flow, and task.
4. Run focused orchestration and ingestion tests, Ruff, and formatter check.
5. Update the README flow example and commit the slice.

## Task 2: Acquire and validate the real snapshot

1. Start the installed Docker Desktop if necessary and create an isolated local
   PostgreSQL container without adding Docker Compose files.
2. Apply the existing Alembic migrations.
3. Run the scoped Prefect flow for 2025-01-01 through 2025-05-31 with
   `location_id=1` and database loading enabled.
4. Verify both manifests, validation reports, row counts, uniqueness, and the
   final 168-hour contiguous serving window.
5. Record paths and hashes without tracking raw data.

## Task 3: Build features and real model evidence

1. Create the local Victorian holiday calendar for 2025-01-01 through
   2025-06-01 from the official 2025 list.
2. Build the manifest-verified supervised CSV.
3. Evaluate Ridge and LightGBM with three validation months and the shared
   final test month; render both Markdown reports.
4. Verify the summaries contain MAE, RMSE, WAPE, split boundaries, and Seasonal
   Naive comparisons.
5. Export the final-fit LightGBM artifact using the real LightGBM summary and
   verify its manifest and model version.

## Task 4: Prove API and Dashboard integration

1. Configure the real database, artifact, and LightGBM metrics summary.
2. Start Uvicorn and verify health, sensors, history, forecast, and metrics over
   loopback HTTP. Confirm the forecast has horizons 1 through 24, non-negative
   values, the expected location, cutoff, and model version.
3. Start Streamlit against that API and verify Today, Explore, and Forecast use
   returned real data.
4. Capture one screenshot of each configured page.

## Task 5: Package evidence and verify

1. Add a compact tracked evidence manifest, real metrics reports, screenshots,
   and an exact local runbook. Do not track raw data, processed rows, database
   files, MLflow state, credentials, or model binaries.
2. Update README status and demo instructions without making production claims.
3. Run focused tests, Ruff check, formatter check, full pytest, configured
   serving smoke, `git diff --check`, and secret/large-file hygiene checks.
4. Review the complete diff, commit locally, and leave remote pushing for
   explicit user authorization.
