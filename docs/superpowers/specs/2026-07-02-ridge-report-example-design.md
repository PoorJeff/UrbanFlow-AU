# Ridge Report Example Artifact Design

Date: 2026-07-02

## Goal

Add a small repository-tracked example that shows what the Ridge evaluation
Markdown report looks like without requiring a user to generate local model
artifacts first.

This makes the recently added Ridge report renderer visible as a portfolio
artifact. A user should be able to open the example Markdown file in GitHub or
an editor and understand the shape of the model-evaluation output.

## Current project context

The repository already has:

- `scripts/evaluate_ridge_baseline.py`, which prints a Ridge evaluation JSON
  summary from an already-built supervised feature CSV;
- `scripts/render_ridge_evaluation_report.py`, which renders that JSON summary
  into Markdown;
- `src/urbanflow/modeling/reports.py`, which owns deterministic Markdown
  rendering for Ridge evaluation summaries;
- `tests/unit/modeling/test_modeling_reports.py`, which covers renderer and
  report CLI behavior;
- README guidance for generating a local Ridge JSON summary and Markdown
  report under `reports/modeling/`.

What is still missing is a checked-in example report that a reviewer can inspect
without running any local commands.

## Selected approach

Commit a deterministic example JSON summary and its rendered Markdown report
under `docs/examples/modeling/`:

- `docs/examples/modeling/ridge_evaluation_summary.json` stores a compact,
  synthetic Ridge evaluation summary that matches the existing report input
  contract.
- `docs/examples/modeling/ridge_evaluation_report.md` stores the Markdown output
  rendered from that JSON by `scripts/render_ridge_evaluation_report.py`.
- README links to the example Markdown report from the Ridge baseline section.
- A unit test verifies that the example JSON can still be rendered and that the
  checked-in Markdown stays in sync with the renderer.

The example should use clearly synthetic values. It should not claim to be a
production benchmark or a real Melbourne result.

## Alternatives considered

### 1. Commit JSON plus rendered Markdown

This is the selected approach. Keeping both files makes the data contract and
the visible output easy to inspect, while the sync test prevents drift.

### 2. Commit only the Markdown report

This would be slightly smaller, but it would not exercise the CLI input
contract or prove that the example is reproducible from a valid Ridge summary.

### 3. Generate the example from a CSV fixture during tests

This would be closer to the full modeling path, but it would expand the scope
into feature data and model fitting. The example report slice should stay about
report visibility, not model training.

## Example JSON contract

The example JSON should be small enough to read in one screen and should include:

- top-level `input_path`, `row_count`, `validation_window_count`,
  `validation_windows`, and `final_test`;
- one validation window;
- one final-test window;
- one or two horizon metric rows;
- the existing metrics `row_count`, `mae`, `rmse`, and `wape`.

Use a source path such as:

```text
docs/examples/modeling/synthetic_supervised_rows.csv
```

That path is illustrative only. The implementation should not add or require a
synthetic CSV for this slice.

Metric values should be plausible and deterministic, for example:

- final-test MAE around `1.2`;
- final-test RMSE around `1.7`;
- final-test WAPE around `0.07`;
- validation-window metrics close to, but not identical to, final-test metrics.

## Example Markdown output

The checked-in Markdown report should be produced by the existing renderer. It
should contain the same sections as runtime-generated reports:

- `# Ridge Evaluation Report`;
- source path;
- row and validation-window counts;
- final-test overall metrics;
- validation-window table;
- final-test horizon table.

The example report should not include manual prose that the renderer does not
produce. Keeping it renderer-owned ensures the drift test can compare exact file
contents.

## README update

In the Ridge baseline section, after the local JSON-to-Markdown commands, add a
short sentence linking to:

```text
docs/examples/modeling/ridge_evaluation_report.md
```

The wording should make clear that the file is a checked-in synthetic example,
not a result generated from the user's local data.

## Testing strategy

Extend `tests/unit/modeling/test_modeling_reports.py` with one repository-file
test:

- load `docs/examples/modeling/ridge_evaluation_summary.json`;
- render it with `render_ridge_evaluation_report`;
- compare the result exactly to
  `docs/examples/modeling/ridge_evaluation_report.md`.

This test proves:

- the example JSON still satisfies the report input contract;
- the example Markdown is reproducible from the renderer;
- future renderer changes intentionally update the example report.

The test should not run Ridge model fitting, read supervised CSVs, or write to
the local `reports/` directory.

## Error handling

No new user-facing error path is needed for this slice. The existing renderer
and CLI already handle invalid summary JSON and output-file errors.

If the example JSON drifts from the renderer contract, the unit test should fail
with the existing `RidgeReportError` or with a Markdown diff assertion.

## Out of scope

This slice intentionally does not add:

- real training data;
- generated files under `reports/`;
- model fitting in tests;
- charts, HTML, PDF, screenshots, or image assets;
- Streamlit or dashboard views;
- MLflow logging;
- PostgreSQL reads;
- model artifact persistence;
- new dependencies.

## Success criteria

The implementation is successful when:

- `docs/examples/modeling/ridge_evaluation_summary.json` exists and is valid
  input for the Ridge report renderer;
- `docs/examples/modeling/ridge_evaluation_report.md` exists and is readable on
  GitHub;
- a unit test proves the Markdown exactly matches the renderer output for the
  example JSON;
- README links to the example report and labels it synthetic;
- the full repository quality gate passes before merge and again on `main`.
