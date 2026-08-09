# Real-Data Local Serving Demo

This runbook reproduces the verified single-sensor local demo for City of
Melbourne `location_id=1`, `Bou292_T`, Bourke Street Mall (North). It uses
official hourly observations from 2025-01-01 through 2025-05-31 and produces a
local-only LightGBM forecast for the next 24 hours.

The checked-in evidence is under
[`docs/evidence/real-data-local-serving`](../evidence/real-data-local-serving/)
and the three screenshots are under
[`docs/assets/real-data-local-serving`](../assets/real-data-local-serving/).
Raw snapshots, processed CSVs, the PostgreSQL cluster, MLflow state, and model
binaries are intentionally not tracked.

## 1. Prepare the Python environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 2. Start a loopback-only PostgreSQL database

Any compatible local PostgreSQL instance can be used. The verified run used
EDB's PostgreSQL 17.10 Windows binary archive, installed below the current
user's local application-data directory. EDB documents these archives as a
convenience for expert users.

```powershell
$runtimeRoot = Join-Path $env:LOCALAPPDATA "UrbanFlowAU"
$downloadRoot = Join-Path $runtimeRoot "downloads"
$archive = Join-Path $downloadRoot "postgresql-17.10-2-windows-x64-binaries.zip"
$installRoot = Join-Path $runtimeRoot "postgresql-17.10"
$pgBin = Join-Path $installRoot "pgsql\bin"
$pgData = Join-Path $runtimeRoot "postgres-data-17"

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
curl.exe --fail --location --output $archive `
  "https://get.enterprisedb.com/postgresql/postgresql-17.10-2-windows-x64-binaries.zip"
New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $installRoot

& (Join-Path $pgBin "initdb.exe") `
  --pgdata=$pgData --username=urbanflow --auth=trust --encoding=UTF8 --locale=C
& (Join-Path $pgBin "pg_ctl.exe") `
  --pgdata=$pgData `
  --log=(Join-Path $runtimeRoot "postgres-55432.log") `
  --options="-p 55432 -h 127.0.0.1" start
& (Join-Path $pgBin "createdb.exe") `
  -h 127.0.0.1 -p 55432 -U urbanflow urbanflow
```

The `trust` configuration above is acceptable only for this disposable cluster
because PostgreSQL is bound to `127.0.0.1`. Do not reuse it for a shared or
production database.

Apply the schema:

```powershell
$env:URBANFLOW_DATABASE_URL = `
  "postgresql+psycopg://urbanflow@127.0.0.1:55432/urbanflow"
.\.venv\Scripts\python.exe -m alembic upgrade head
```

## 3. Collect, validate, and load the scoped snapshot

```powershell
.\.venv\Scripts\python.exe scripts\run_ingestion_flow.py `
  --start-date 2025-01-01 `
  --end-date 2025-05-31 `
  --location-id 1 `
  --report-root data\processed\real_demo\data_quality `
  --load-to-database `
  --database-url $env:URBANFLOW_DATABASE_URL
```

The verified snapshot contains exactly 3,624 rows: 151 complete days times 24
hours. Its scoped manifest, source query, and SHA256 are preserved in
[`hourly_counts_manifest.json`](../evidence/real-data-local-serving/hourly_counts_manifest.json).

Resolve the newest local snapshot for the remaining commands:

```powershell
$hourlyManifestPath = Get-ChildItem data\manifests\hourly_counts\*.json |
  Sort-Object LastWriteTime |
  Select-Object -Last 1 -ExpandProperty FullName
$hourlyManifest = Get-Content -Raw $hourlyManifestPath | ConvertFrom-Json
$hourlySnapshotPath = $hourlyManifest.snapshot_path
```

## 4. Build the supervised dataset

The holiday input covers the full observation interval and the 2025-06-01
forecast targets. The dates were taken from the official
[Business Victoria 2025 list](https://business.vic.gov.au/business-information/public-holidays/victorian-public-holidays-2025)
and cross-checked against the
[Fair Work Ombudsman 2025 list](https://www.fairwork.gov.au/employment-conditions/public-holidays/2025-public-holidays).

```powershell
New-Item -ItemType Directory -Force -Path data\processed\real_demo | Out-Null
Copy-Item `
  docs\evidence\real-data-local-serving\victoria_public_holidays_2025.json `
  data\processed\real_demo\holiday_calendar.json

.\.venv\Scripts\python.exe scripts\build_supervised_csv.py `
  $hourlySnapshotPath `
  $hourlyManifestPath `
  data\processed\real_demo\supervised_rows.csv `
  --holiday-calendar data\processed\real_demo\holiday_calendar.json
```

The verified output has 87,000 supervised rows and SHA256
`821e93393a1985a26c0d1837b7b2876b557011cb1ffc398ba400bbd546793abd`.

## 5. Evaluate all three baselines and render reports

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ridge_baseline.py `
  data\processed\real_demo\supervised_rows.csv `
  --validation-months 3 |
  Set-Content -Encoding utf8 data\processed\real_demo\ridge_evaluation_summary.json

.\.venv\Scripts\python.exe scripts\evaluate_lightgbm_baseline.py `
  data\processed\real_demo\supervised_rows.csv `
  --validation-months 3 |
  Set-Content -Encoding utf8 data\processed\real_demo\lightgbm_evaluation_summary.json

.\.venv\Scripts\python.exe scripts\render_ridge_evaluation_report.py `
  data\processed\real_demo\ridge_evaluation_summary.json `
  --output data\processed\real_demo\ridge_evaluation_report.md --force

.\.venv\Scripts\python.exe scripts\render_lightgbm_evaluation_report.py `
  data\processed\real_demo\lightgbm_evaluation_summary.json `
  --output data\processed\real_demo\lightgbm_evaluation_report.md --force
```

The three models use the same February-April validation boundaries and May
final-test boundary. The final-test metrics are preserved in
[`evaluation_summary.json`](../evidence/real-data-local-serving/evaluation_summary.json).
LightGBM won this particular historical final test, but the result is not a
production performance claim.

## 6. Export the serving artifact

```powershell
.\.venv\Scripts\python.exe scripts\export_lightgbm_artifact.py `
  data\processed\real_demo\supervised_rows.csv `
  models\lightgbm\real-demo `
  --holiday-calendar data\processed\real_demo\holiday_calendar.json `
  --evaluation-summary-path data\processed\real_demo\lightgbm_evaluation_summary.json
```

The verified artifact version is `lightgbm-887517f7e42d`. Its manifest records
the supervised CSV hash, training row count, training cutoff, feature contract,
holiday coverage, and model hash; a checked-in copy is available in
[`serving_artifact_manifest.json`](../evidence/real-data-local-serving/serving_artifact_manifest.json).

## 7. Start the API and Dashboard

In one PowerShell terminal:

```powershell
$env:URBANFLOW_DATABASE_URL = `
  "postgresql+psycopg://urbanflow@127.0.0.1:55432/urbanflow"
$env:URBANFLOW_API_MODEL_ARTIFACT_PATH = ".\models\lightgbm\real-demo"
$env:URBANFLOW_API_METRICS_PATH = `
  ".\data\processed\real_demo\lightgbm_evaluation_summary.json"
.\.venv\Scripts\python.exe -m uvicorn urbanflow.api.app:app `
  --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
$env:URBANFLOW_DASHBOARD_API_BASE_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py `
  --server.address 127.0.0.1 --server.port 8501
```

Verify the real serving boundary:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors?active_only=true"
Invoke-RestMethod `
  "http://127.0.0.1:8000/api/v1/sensors/1/history?start=2025-05-25T00:00:00%2B10:00&end=2025-06-01T00:00:00%2B10:00"
Invoke-RestMethod `
  "http://127.0.0.1:8000/api/v1/sensors/1/forecast?horizon=24"
Invoke-RestMethod http://127.0.0.1:8000/api/v1/model/metrics
```

Because no freshness threshold is configured, `/health` honestly returns
`degraded` with HTTP 200 while the database and model provider remain
`available`. The metrics response intentionally keeps `model_version` as JSON
`null` because the evaluation summary predates artifact export; the real
artifact version is returned by `/health` and `/forecast` instead.

## 8. Stop the disposable database

```powershell
& (Join-Path $pgBin "pg_ctl.exe") --pgdata=$pgData stop
```

## Verified evidence

- [Evidence manifest](../evidence/real-data-local-serving/evidence.json)
- [Today screenshot](../assets/real-data-local-serving/today.png)
- [Explore screenshot](../assets/real-data-local-serving/explore.png)
- [Forecast screenshot](../assets/real-data-local-serving/forecast.png)
- [Ridge report](../evidence/real-data-local-serving/ridge_evaluation_report.md)
- [LightGBM report](../evidence/real-data-local-serving/lightgbm_evaluation_report.md)

This demo is deliberately scoped to one historical sensor. It does not add
monitoring, Docker Compose packaging, weather forecasts, multi-sensor training,
online deployment, or any production reliability or accuracy claim.
