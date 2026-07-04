# Seasonal Naive Comparison Evaluation Design

Date: 2026-07-04

## Goal

Add a fair model-comparison slice that reports Ridge Regression metrics beside
Seasonal Naive metrics for the same rolling-origin validation and final-test
windows.

This makes the existing Ridge evaluation more honest and useful: readers can
see whether the trainable Ridge baseline improves over the one-week-prior
seasonal baseline before the project moves on to LightGBM or MLflow.

## Current project context

The repository already has:

- `src/urbanflow/modeling/baselines.py`, with
  `add_seasonal_naive_predictions(supervised_frame, panel_frame)`, which joins
  the same location's value from 168 hours before each target timestamp;
- `src/urbanflow/modeling/evaluation.py`, which evaluates Ridge over
  rolling-origin windows and stores predictions plus overall and per-horizon
  Ridge metrics;
- `src/urbanflow/modeling/cli.py`, which reads supervised feature rows, builds
  rolling-origin splits, evaluates Ridge, and emits a JSON summary;
- `src/urbanflow/modeling/reports.py`, which renders the JSON summary as a
  Markdown report with final-test metrics, validation-window metrics,
  per-horizon metrics, and Mermaid metric charts;
- unit tests for Seasonal Naive predictions, Ridge evaluation, CLI JSON output,
  and checked-in report drift.

The missing piece is that Seasonal Naive is not yet included in the main
evaluation summary or Markdown report. That makes the report look like a Ridge
scorecard rather than a model-comparison artifact.

## Selected approach

Extend the existing Ridge evaluation flow to compute a Seasonal Naive baseline
for every evaluated row and include its metrics in the same JSON summary and
Markdown report.

The implementation should:

1. derive Seasonal Naive predictions from the supervised rows themselves;
2. evaluate Ridge and Seasonal Naive on exactly the same validation and
   final-test windows;
3. keep Ridge as the trainable model under evaluation;
4. add comparison fields to the JSON summary without changing the CLI command
   name or requiring a second input file;
5. render a compact Markdown comparison table for validation windows and final
   test.

The command remains:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

It continues to emit JSON to stdout. The report command continues to consume
that JSON.

## Alternatives considered

### 1. Add Seasonal Naive into the existing Ridge evaluation summary

This is the selected approach. It is the smallest useful comparison slice and
keeps all evaluated rows, splits, and report artifacts aligned. It also reuses
the existing Seasonal Naive function and report renderer.

### 2. Create a separate Seasonal Naive evaluation CLI

A separate command would be cleaner in isolation, but it would force users to
run two commands and then mentally compare two JSON outputs. That would delay
the project's model-comparison story.

### 3. Jump directly to LightGBM comparison

LightGBM is an important next model, but adding it before the Seasonal Naive
comparison would expand the scope into dependency management, model
configuration, and likely artifact tracking. The fair baseline comparison
should land first.

## Data contract

The input remains the existing supervised feature CSV. It must contain the
fields already required by Ridge evaluation, including:

- `location_id`;
- `forecast_origin_at`;
- `target_observed_at`;
- `forecast_horizon`;
- `target`;
- feature columns used by Ridge.

For Seasonal Naive, the implementation should derive a compact panel from the
same CSV:

- use `location_id`;
- use `target_observed_at` as the observed timestamp;
- use `target` as the observed pedestrian count.

Rows should then be passed through `add_seasonal_naive_predictions`. This avoids
requiring a raw panel CSV and keeps the evaluation CLI single-input and
reproducible.

If duplicate `location_id + target_observed_at` pairs exist because the
supervised frame has multiple forecast horizons for the same target timestamp,
the derived panel should de-duplicate those pairs before joining. If duplicate
pairs disagree on `target`, raise a clear expected user error because the CSV is
internally inconsistent.

## Evaluation behavior

Each `ModelWindowEvaluation` should include:

- Ridge predictions and metrics, as it does today;
- Seasonal Naive predictions for the same evaluation rows;
- Seasonal Naive overall metrics;
- Seasonal Naive per-horizon metrics;
- a model-comparison summary that can compute relative WAPE improvement when
  both Ridge and Seasonal Naive WAPE are numeric.

The relative WAPE improvement should use:

```text
(seasonal_naive_wape - ridge_wape) / seasonal_naive_wape
```

If either WAPE is missing, or the Seasonal Naive WAPE denominator is `0`, render
the improvement as `null` in JSON and `n/a` in Markdown.

Seasonal Naive rows with missing one-week-prior history should not crash the
evaluation. They should be included in the Ridge metrics, while Seasonal Naive
metrics should naturally drop rows with missing baseline predictions through the
existing `calculate_regression_metrics` behavior. The JSON should make this
visible through the Seasonal Naive metric `row_count`.

## JSON output design

Keep the existing top-level fields:

- `input_path`;
- `row_count`;
- `validation_window_count`;
- `validation_windows`;
- `final_test`.

For each window, keep existing fields and add:

```json
{
  "overall": {
    "row_count": 672,
    "mae": 1.2,
    "rmse": 1.7,
    "wape": 0.07
  },
  "seasonal_naive_overall": {
    "row_count": 672,
    "mae": 1.8,
    "rmse": 2.3,
    "wape": 0.10
  },
  "model_comparison": {
    "ridge_wape": 0.07,
    "seasonal_naive_wape": 0.10,
    "relative_wape_improvement": 0.30
  },
  "horizon_metrics": [
    {
      "forecast_horizon": 1,
      "row_count": 336,
      "mae": 1.1,
      "rmse": 1.6,
      "wape": 0.06
    }
  ],
  "seasonal_naive_horizon_metrics": [
    {
      "forecast_horizon": 1,
      "row_count": 336,
      "mae": 1.7,
      "rmse": 2.2,
      "wape": 0.09
    }
  ]
}
```

The existing `overall` and `horizon_metrics` names remain Ridge metrics to avoid
breaking the current report renderer and checked-in example. New fields carry
the baseline comparison explicitly.

## Markdown report design

The report should add one compact section after the final-test overall metric
table and before validation-window details:

```markdown
## Model comparison

| Window | Model | Rows | MAE | RMSE | WAPE | Relative WAPE improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| final_test_2025-02 | Ridge | 672 | 1.2000 | 1.7000 | 0.0700 | 30.00% |
| final_test_2025-02 | Seasonal Naive | 672 | 1.8000 | 2.3000 | 0.1000 | n/a |
| validation_2025-01 | Ridge | 744 | 1.2345 | 1.7654 | 0.0812 | 12.34% |
| validation_2025-01 | Seasonal Naive | 744 | 1.4100 | 1.9500 | 0.0926 | n/a |
```

Rules:

- Ridge rows show the relative WAPE improvement versus Seasonal Naive for that
  same window.
- Seasonal Naive rows show `n/a` for relative improvement because they are the
  comparison baseline.
- The table should list final test first, then validation windows in existing
  order. This keeps the most important score at the top.
- Existing final-test, validation-window, Mermaid, and horizon sections remain.

The current Mermaid charts can remain Ridge-only in this slice. A later slice
can add model-comparison charts if needed.

## Error handling

Expected user errors should return CLI exit code `2` without tracebacks, matching
the existing CLI pattern.

Add expected user errors for:

- missing required Seasonal Naive derivation columns;
- conflicting duplicate `location_id + target_observed_at` targets in the input
  CSV;
- failure to create any Seasonal Naive metric rows for all windows. This should
  not be fatal if only some early windows lack one-week-prior history, but it
  should be reported if the baseline is entirely unavailable.

Report rendering should remain backward-compatible with older JSON summaries
that do not yet contain Seasonal Naive fields. In that case, omit the
`## Model comparison` section rather than failing. This keeps older checked or
locally generated Ridge JSON files renderable.

## Testing strategy

Use test-driven implementation in the next phase:

- add unit tests for deriving a Seasonal Naive panel from supervised rows,
  including duplicate-consistent and duplicate-conflicting cases;
- add evaluation tests proving each window includes Seasonal Naive overall and
  per-horizon metrics computed on the same evaluation rows;
- add CLI tests proving JSON contains `seasonal_naive_overall`,
  `seasonal_naive_horizon_metrics`, and `model_comparison`;
- add report renderer tests proving the model-comparison table appears when
  fields are present and is omitted for older Ridge-only summaries;
- update the checked-in synthetic example JSON and Markdown report and keep the
  drift test as the final artifact guard;
- run targeted modeling tests after each implementation step and the full
  repository quality gate before merging to `main`.

## Out of scope

This slice intentionally does not add:

- LightGBM training or evaluation;
- MLflow tracking;
- model artifact persistence;
- API forecast fallback behavior;
- Streamlit dashboard changes;
- new dependencies;
- comparison charts beyond the Markdown table;
- new metrics beyond the existing MAE, RMSE, WAPE, and relative WAPE
  improvement.

## Success criteria

The implementation will be successful when:

- the Ridge evaluation JSON includes Seasonal Naive metrics for every evaluated
  window where one-week-prior history is available;
- Ridge and Seasonal Naive metrics use the same rolling-origin windows;
- relative WAPE improvement is reported when meaningful and `null` / `n/a`
  otherwise;
- the Markdown report contains a compact model-comparison table while remaining
  backward-compatible with older Ridge-only summaries;
- the checked-in example report matches the renderer output;
- targeted modeling tests and the full repository quality gate pass before
  merge, and the same gate is re-run on `main` after merge.

## Self-review

- Placeholder scan: no unresolved placeholder language or incomplete
  implementation decisions.
- Internal consistency: JSON fields, Markdown output, error handling, and tests
  all describe the same Ridge-versus-Seasonal-Naive comparison slice.
- Scope check: one evaluation/reporting slice only; LightGBM, MLflow, API, and
  dashboard work remain separate.
- Ambiguity check: input derivation, duplicate handling, metric row counts,
  relative improvement, report placement, and backward compatibility are
  explicit.
