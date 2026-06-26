# Sensor Location Runner Design

## Purpose

Provide a small, manually runnable entry point for the existing sensor-location
ingestion pipeline. A developer should be able to fetch the current City of
Melbourne sensor-location dataset and save the immutable local snapshot and
manifest without writing any application-specific orchestration code.

The runner is a local development and smoke-validation tool. It does not make
network access part of the automated test suite and it does not add Prefect or
other scheduling infrastructure.

## Selected Approach

The repository will add a thin executable script:

```powershell
python scripts/ingest_sensor_locations.py
```

The script delegates argument parsing and application behavior to an internal
CLI module under `src/urbanflow/ingestion/`. That module calls the existing
`ingest_sensor_locations()` function; it must not duplicate fetching,
normalization, snapshot, or manifest behavior.

This approach was selected over:

- an import-only function, which already exists but does not provide an easy
  manual operator command;
- a `python -m` public module, which adds a package-level command convention
  before it is needed;
- a Prefect flow, which would add orchestration dependencies before the first
  live local run is proven.

## Command Interface

The runner uses `argparse` and supports:

- `--raw-root PATH`, defaulting to `data/raw` relative to the current working
  directory;
- `--manifest-root PATH`, defaulting to `data/manifests` relative to the
  current working directory;
- `--page-limit POSITIVE_INTEGER`, defaulting to `100`.

The runner uses the current UTC time for `extracted_at`, matching the pipeline
default. It does not expose a timestamp override because deterministic dates
belong in tests, not the normal manual command.

`data/manifests/` will be added to `.gitignore`; both default output trees are
local generated data and must remain untracked.

## Data Flow and Output

```text
arguments -> MelbourneApiClient -> ingest_sensor_locations()
          -> normalized JSON snapshot + manifest
          -> JSON result summary on stdout
```

The CLI creates an `httpx.Client` with the existing 30-second timeout inside a
context manager, passes it to `MelbourneApiClient`, then passes that client to
the pipeline. This closes the HTTP session after the operation while leaving
the reusable API client unchanged.

On success, stdout contains one JSON object with the source dataset and URL,
extraction timestamp, source and normalized record counts, and generated
snapshot and manifest paths. The command returns exit code `0`.

## Failure Handling

- Invalid command-line values are rejected by `argparse` with exit code `2`.
- Expected operational failures from the API, sensor parser, filesystem, or
  immutable-path collision are printed as one `error: ...` message on stderr
  and return exit code `1`.
- Unexpected programming errors are not swallowed, preserving tracebacks for
  diagnosis.

No snapshot or manifest is written when parsing fails because the existing
pipeline normalizes before writing outputs.

## Testing and Verification

Unit tests will import the internal CLI entry point and inject a fake API-client
factory. They will verify:

- defaults call the existing pipeline with `data/raw`, `data/manifests`, and
  page limit `100`;
- supplied options override those values;
- success prints a machine-readable JSON summary and returns `0`;
- expected pipeline errors print a concise stderr error and return `1`.

Tests will never request the live City of Melbourne API. After the full quality
gate passes, one manual command run may call the official endpoint and verify
that local, ignored outputs were created. The generated files are not staged or
committed.

## Scope Boundaries

This slice excludes hourly-count ingestion, Parquet output, data validation,
database writes, Prefect orchestration, scheduled execution, and changes to
the sensor-location pipeline contract.
