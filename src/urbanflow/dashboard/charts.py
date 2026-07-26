import plotly.graph_objects as go

from urbanflow.api.schemas import ForecastResponse, HistoryResponse


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
            x=[prediction.target_at for prediction in forecast.predictions],
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
        x=[point.observed_at for point in history.data],
        y=[point.pedestrian_count for point in history.data],
        mode="lines+markers",
        name="Observed",
        line={"dash": "solid"},
    )
