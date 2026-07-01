from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    row_count: int
    mae: float | None
    rmse: float | None
    wape: float | None


def _valid_metric_rows(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    return frame.dropna(subset=["actual", "predicted"])


def calculate_regression_metrics(actual: pd.Series, predicted: pd.Series) -> RegressionMetrics:
    rows = _valid_metric_rows(actual, predicted)
    if rows.empty:
        return RegressionMetrics(row_count=0, mae=None, rmse=None, wape=None)

    absolute_errors = (rows["actual"] - rows["predicted"]).abs()
    squared_errors = (rows["actual"] - rows["predicted"]) ** 2
    denominator = rows["actual"].abs().sum()
    return RegressionMetrics(
        row_count=len(rows),
        mae=float(absolute_errors.mean()),
        rmse=float(math.sqrt(squared_errors.mean())),
        wape=None if denominator == 0 else float(absolute_errors.sum() / denominator),
    )


def summarize_by_group(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    actual_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metrics = calculate_regression_metrics(group[actual_column], group[prediction_column])
        record = dict(zip(group_columns, group_key, strict=True))
        record.update(
            {
                "row_count": metrics.row_count,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "wape": metrics.wape,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def peak_top_decile_mae(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
) -> float | None:
    rows = frame.dropna(subset=[actual_column, prediction_column])
    if rows.empty:
        return None
    peak_count = max(1, math.ceil(len(rows) * 0.10))
    peak_rows = rows.nlargest(peak_count, actual_column)
    metrics = calculate_regression_metrics(peak_rows[actual_column], peak_rows[prediction_column])
    return metrics.mae
