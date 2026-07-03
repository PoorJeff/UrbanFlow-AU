from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class RidgeReportError(ValueError):
    """Raised when a Ridge evaluation summary cannot be rendered."""


_METRIC_CHARTS = (
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("wape", "WAPE"),
)


def _field_path(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _required(mapping: Mapping[str, Any], key: str, *, path: str = "") -> Any:
    if key not in mapping:
        raise RidgeReportError(f"missing required summary field: {_field_path(path, key)}")
    return mapping[key]


def _required_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RidgeReportError(f"summary field must be an object: {path}")
    return value


def _required_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RidgeReportError(f"summary field must be a list: {path}")
    return value


def _metric_text(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(metric):
        return "n/a"
    return f"{metric:.4f}"


def _count_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cell_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_mapping(window: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    overall_path = _field_path(path, "overall")
    overall = _required_mapping(_required(window, "overall", path=path), path=overall_path)
    for key in ("row_count", "mae", "rmse", "wape"):
        _required(overall, key, path=overall_path)
    return overall


def _window_mapping(summary: Mapping[str, Any], key: str, *, path: str = "") -> Mapping[str, Any]:
    window_path = _field_path(path, key)
    window = _required_mapping(_required(summary, key, path=path), path=window_path)
    for field in ("name", "start", "end", "train_end", "training_row_count", "horizon_metrics"):
        _required(window, field, path=window_path)
    _metric_mapping(window, path=window_path)
    return window


def _horizon_records(window: Mapping[str, Any], *, path: str) -> Sequence[Any]:
    records_path = _field_path(path, "horizon_metrics")
    records = _required_sequence(_required(window, "horizon_metrics", path=path), path=records_path)
    for index, record in enumerate(records):
        record_path = f"{records_path}.{index}"
        mapping = _required_mapping(record, path=record_path)
        for field in ("forecast_horizon", "row_count", "mae", "rmse", "wape"):
            _required(mapping, field, path=record_path)
    return records


def _period(window: Mapping[str, Any]) -> str:
    return f"{_cell_text(window['start'])} to {_cell_text(window['end'])}"


def _validation_row(window: Mapping[str, Any]) -> str:
    overall = _metric_mapping(window, path=str(window["name"]))
    return (
        f"| {_cell_text(window['name'])} | {_period(window)} | "
        f"{_count_text(window['training_row_count'])} | "
        f"{_count_text(overall['row_count'])} | "
        f"{_metric_text(overall['mae'])} | "
        f"{_metric_text(overall['rmse'])} | "
        f"{_metric_text(overall['wape'])} |"
    )


def _horizon_row(record: Mapping[str, Any]) -> str:
    return (
        f"| {_count_text(record['forecast_horizon'])} | "
        f"{_count_text(record['row_count'])} | "
        f"{_metric_text(record['mae'])} | "
        f"{_metric_text(record['rmse'])} | "
        f"{_metric_text(record['wape'])} |"
    )


def _numeric_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(metric) or math.isinf(metric):
        return None
    return metric


def _mermaid_label(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return f'"{text}"'


def _chart_axis_upper_bound(values: Sequence[float]) -> float:
    if max(values) == 0:
        return 1.0
    return max(values) * 1.1


def _metric_chart_points(
    validation_windows: Sequence[Mapping[str, Any]],
    final_test: Mapping[str, Any],
    metric_key: str,
) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for window in [*validation_windows, final_test]:
        overall = _metric_mapping(window, path=str(window["name"]))
        metric = _numeric_metric_value(overall[metric_key])
        if metric is None:
            continue
        points.append((str(window["name"]), metric))
    return points


def _mermaid_metric_chart(metric_label: str, points: Sequence[tuple[str, float]]) -> str:
    labels = ", ".join(_mermaid_label(label) for label, _ in points)
    values = ", ".join(f"{value:.4f}" for _, value in points)
    upper_bound = _chart_axis_upper_bound([value for _, value in points])
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{metric_label} by evaluation window"',
            f"    x-axis [{labels}]",
            f'    y-axis "{metric_label}" 0 --> {upper_bound:.4f}',
            f"    bar [{values}]",
            "```",
        ]
    )


def _metric_comparison_chart_lines(
    validation_windows: Sequence[Mapping[str, Any]],
    final_test: Mapping[str, Any],
) -> list[str]:
    chart_blocks = []
    for metric_key, metric_label in _METRIC_CHARTS:
        points = _metric_chart_points(validation_windows, final_test, metric_key)
        if not points:
            continue
        chart_blocks.append(_mermaid_metric_chart(metric_label, points))

    if not chart_blocks:
        return []

    lines = ["", "## Metric comparison charts", ""]
    for index, chart_block in enumerate(chart_blocks):
        if index > 0:
            lines.append("")
        lines.extend(chart_block.splitlines())
    return lines


def render_ridge_evaluation_report(summary: Mapping[str, Any]) -> str:
    for field in (
        "input_path",
        "row_count",
        "validation_window_count",
        "validation_windows",
        "final_test",
    ):
        _required(summary, field)

    validation_windows = _required_sequence(
        summary["validation_windows"],
        path="validation_windows",
    )
    validation_window_mappings = [
        _required_mapping(window, path=f"validation_windows.{index}")
        for index, window in enumerate(validation_windows)
    ]
    for index, window in enumerate(validation_window_mappings):
        window_path = f"validation_windows.{index}"
        for field in ("name", "start", "end", "train_end", "training_row_count", "horizon_metrics"):
            _required(window, field, path=window_path)
        _metric_mapping(window, path=window_path)

    final_test = _window_mapping(summary, "final_test")
    final_overall = _metric_mapping(final_test, path="final_test")
    final_horizons = [
        _required_mapping(record, path=f"final_test.horizon_metrics.{index}")
        for index, record in enumerate(_horizon_records(final_test, path="final_test"))
    ]

    lines = [
        "# Ridge Evaluation Report",
        "",
        f"Source: `{_cell_text(summary['input_path'])}`",
        "",
        f"Rows evaluated: {_count_text(summary['row_count'])}",
        f"Validation windows: {_count_text(summary['validation_window_count'])}",
        "",
        "## Final test",
        "",
        f"Window: `{_cell_text(final_test['name'])}`",
        f"Period: {_period(final_test)}",
        f"Training rows: {_count_text(final_test['training_row_count'])}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Row count | {_count_text(final_overall['row_count'])} |",
        f"| MAE | {_metric_text(final_overall['mae'])} |",
        f"| RMSE | {_metric_text(final_overall['rmse'])} |",
        f"| WAPE | {_metric_text(final_overall['wape'])} |",
        "",
        "## Validation windows",
        "",
        "| Window | Period | Training rows | Rows | MAE | RMSE | WAPE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_validation_row(window) for window in validation_window_mappings)
    lines.extend(_metric_comparison_chart_lines(validation_window_mappings, final_test))
    lines.extend(
        [
            "",
            "## Final test by horizon",
            "",
            "| Horizon | Rows | MAE | RMSE | WAPE |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_horizon_row(record) for record in final_horizons)
    return "\n".join(lines) + "\n"
