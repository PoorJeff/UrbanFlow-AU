# Ridge Evaluation Markdown Report Design

Date: 2026-07-02

## Goal

Add a small local reporting slice that turns a Ridge evaluation JSON summary
into a human-readable Markdown report.

This gives UrbanFlow AU its first lightweight model-evaluation artifact that a
user can open in an editor, view on GitHub, or later convert to HTML/PDF. The
slice should stay deterministic and file-based. It should not add charts,
browser rendering, Streamlit, MLflow, database reads, or model artifacts.

## Current project context

The repository already has:

- `scripts/evaluate_ridge_baseline.py`, which reads an already-built supervised
  feature CSV and prints a Ridge rolling-origin evaluation summary as JSON;
- `urbanflow.modeling.cli.evaluation_summary`, which produces a compact summary
  containing input path, row count, validation windows, final-test metrics, and
  per-horizon metrics;
- `urbanflow.validation.reports`, which demonstrates a small dataclass-first
  reporting pattern with deterministic JSON serialization and explicit file
  writes;
- README guidance for generating the Ridge JSON summary.

The repository does not yet have a user-facing model report file. Users can
inspect JSON, but JSON is not pleasant as a portfolio artifact or model-review
medium.

## Selected approach

Create a Markdown report renderer for the existing Ridge evaluation summary:

- `src/urbanflow/modeling/reports.py` owns validation of the JSON-like summary,
  Markdown table formatting, and report text generation.
- `src/urbanflow/modeling/report_cli.py` owns `argparse`, JSON file reading,
  output path handling, and expected user-error exit code `2`.
- `scripts/render_ridge_evaluation_report.py` is a thin wrapper around
  `urbanflow.modeling.report_cli.main`.
- `README.md` documents the two-step local workflow:
  1. run Ridge evaluation and save JSON;
  2. render Markdown from that JSON.

The renderer consumes the JSON summary already emitted by the Ridge evaluation
CLI. It does not run model evaluation itself. That boundary keeps the report
slice easy to test and prevents report formatting from changing model behavior.

## Alternatives considered

### 1. Markdown report from Ridge JSON summary

This is the selected approach. Markdown is easy to diff, test, and view in the
repository. It also provides a stable bridge to later HTML, PDF, or dashboard
views.

### 2. HTML report

HTML would look more visual immediately, but it would introduce template and
style decisions before the report content contract is stable. HTML can be added
later once the Markdown report defines the information hierarchy.

### 3. Streamlit dashboard

Streamlit would be the most interactive, but it would mix reporting,
application runtime, layout, and model-result parsing. It is too large for this
slice.

## Report input contract

The primary input is a JSON file containing the success payload emitted by:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

The report renderer should require these top-level fields:

- `input_path`;
- `row_count`;
- `validation_window_count`;
- `validation_windows`;
- `final_test`.

Each window object should require:

- `name`;
- `start`;
- `end`;
- `train_end`;
- `training_row_count`;
- `overall`;
- `horizon_metrics`.

Each `overall` metric object should require:

- `row_count`;
- `mae`;
- `rmse`;
- `wape`.

Each horizon metric record should require:

- `forecast_horizon`;
- `row_count`;
- `mae`;
- `rmse`;
- `wape`.

Missing fields should raise a report-specific user-input error with the missing
field path, such as `missing required summary field: final_test.overall.mae`.

## Markdown output

The generated Markdown should be deterministic and compact:

```markdown
# Ridge Evaluation Report

Source: `data/modeling/supervised_rows.csv`

Rows evaluated: 1464
Validation windows: 1

## Final test

Window: `final_test_2025-02`
Period: 2025-02-01T00:00:00+11:00 to 2025-03-01T00:00:00+11:00
Training rows: 1488

| Metric | Value |
| --- | ---: |
| Row count | 672 |
| MAE | 1.2000 |
| RMSE | 1.7000 |
| WAPE | 0.0700 |

## Validation windows

| Window | Period | Training rows | Rows | MAE | RMSE | WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation_2025-01 | 2025-01-01T00:00:00+11:00 to 2025-02-01T00:00:00+11:00 | 744 | 744 | 1.2300 | 1.7500 | 0.0800 |

## Final test by horizon

| Horizon | Rows | MAE | RMSE | WAPE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 672 | 1.2000 | 1.7000 | 0.0700 |
```

Numeric metric values should be formatted to four decimal places. Missing metric
values should render as `n/a`. Row counts and horizons should render as
integers when possible.

The report should end with a single trailing newline.

## CLI contract

The primary command should be:

```powershell
python scripts/render_ridge_evaluation_report.py reports/modeling/ridge_evaluation.json --output reports/modeling/ridge_evaluation.md
```

Arguments:

- positional `summary_json`: path to a local Ridge evaluation JSON summary;
- optional `--output`: Markdown output path. When omitted, write next to the
  input file using the same stem and `.md` suffix;
- optional `--force`: overwrite an existing output file. Without `--force`, an
  existing output path is an error.

On success, print a small JSON payload to stdout and return `0`:

```json
{
  "output_path": "reports/modeling/ridge_evaluation.md"
}
```

The command should create parent directories for the output path.

## Error handling and exit codes

Return code `2` for expected user errors:

- missing input JSON;
- unreadable JSON;
- invalid JSON;
- missing required summary fields;
- output file already exists without `--force`;
- failure to write the output file.

On failure, print a concise message to stderr and no success JSON to stdout.
The command should not show tracebacks for expected user input errors.

No separate code `1` is needed in this slice because the command has no network,
database, or external-service runtime path.

## Testing strategy

Unit tests should cover:

- `render_ridge_evaluation_report(summary)` returns Markdown with the title,
  source path, final-test metric table, validation table, and final-test horizon
  table;
- metric formatting uses four decimals and renders `None` as `n/a`;
- missing required fields raise a report-specific error naming the field path;
- `report_cli.main` writes the Markdown file and prints JSON with
  `output_path`;
- existing output without `--force` returns code `2`;
- `python scripts/render_ridge_evaluation_report.py --help` prints report help
  text.

Tests should use small in-memory summary dictionaries and temporary files. They
should not run Ridge model fitting, read supervised CSVs, or depend on real
`reports/` contents.

## README update

Add a short Markdown-report example after the Ridge evaluation CLI example:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3 > reports/modeling/ridge_evaluation.json
python scripts/render_ridge_evaluation_report.py reports/modeling/ridge_evaluation.json --output reports/modeling/ridge_evaluation.md
```

Clarify that `reports/` remains a local generated-artifact area and is not
required for unit tests.

## Out of scope

This slice intentionally does not add:

- HTML, PDF, or image rendering;
- charts;
- Streamlit or dashboard views;
- direct Ridge evaluation execution inside the report command;
- comparison against Seasonal Naive;
- MLflow logging;
- PostgreSQL reads;
- model artifact persistence;
- new dependencies.

Those can be layered on after the Markdown report establishes the model-review
content contract.

## Success criteria

The implementation is successful when:

- a local Ridge evaluation JSON summary can be rendered into a Markdown report;
- the Markdown includes final-test overall metrics, validation-window metrics,
  and final-test per-horizon metrics;
- expected user input errors return code `2` without tracebacks;
- the script wrapper has tested `--help` behavior;
- README documents the JSON-to-Markdown flow;
- the full repository quality gate passes before merge and again on `main`.
