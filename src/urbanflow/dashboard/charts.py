from datetime import datetime

import plotly.graph_objects as go

from urbanflow.api.schemas import ForecastResponse, HistoryResponse
from urbanflow.dashboard.time_utils import MELBOURNE_TIME_ZONE


def build_history_figure(history: HistoryResponse) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(_observed_trace(history))
    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Pedestrian count",
    )
    return figure


def build_forecast_figure(
    *,
    history: HistoryResponse | None,
    forecast: ForecastResponse,
) -> go.Figure:
    figure = go.Figure()
    if history is not None:
        figure.add_trace(_observed_trace(history))
    figure.add_trace(
        go.Scatter(
            x=[_melbourne_timestamp(prediction.target_at) for prediction in forecast.predictions],
            y=[prediction.predicted_count for prediction in forecast.predictions],
            mode="lines+markers",
            name="Forecast",
            line={"dash": "dash"},
        )
    )
    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Pedestrian count",
    )
    return figure


def _observed_trace(history: HistoryResponse) -> go.Scatter:
    return go.Scatter(
        x=[_melbourne_timestamp(point.observed_at) for point in history.data],
        y=[point.pedestrian_count for point in history.data],
        mode="lines+markers",
        name="Observed",
        line={"dash": "solid"},
    )


def _melbourne_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Chart timestamps must be offset-aware.")
    return value.astimezone(MELBOURNE_TIME_ZONE)
