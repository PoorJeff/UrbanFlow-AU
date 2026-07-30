from datetime import UTC, datetime, timedelta

import pytest

from urbanflow.api.schemas import (
    ForecastPredictionResponse,
    ForecastResponse,
    HistoryPoint,
    HistoryResponse,
)
from urbanflow.dashboard.charts import build_forecast_figure, build_history_figure
from urbanflow.dashboard.time_utils import MELBOURNE_TIME_ZONE


@pytest.fixture
def history() -> HistoryResponse:
    returned_order = [
        HistoryPoint(
            observed_at=datetime(2026, 7, 12, 9, tzinfo=UTC),
            pedestrian_count=31,
        ),
        HistoryPoint(
            observed_at=datetime(2026, 7, 12, 8, tzinfo=UTC),
            pedestrian_count=24,
        ),
    ]
    return HistoryResponse(
        location_id=101,
        start=datetime(2026, 7, 12, 8, tzinfo=UTC),
        end=datetime(2026, 7, 12, 10, tzinfo=UTC),
        data=returned_order,
    )


@pytest.fixture
def forecast() -> ForecastResponse:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC)
    returned_order = [
        ForecastPredictionResponse(
            forecast_horizon=2,
            target_at=cutoff + timedelta(hours=2),
            predicted_count=12.75,
        ),
        ForecastPredictionResponse(
            forecast_horizon=1,
            target_at=cutoff + timedelta(hours=1),
            predicted_count=3.125,
        ),
    ]
    return ForecastResponse(
        location_id=101,
        model_name="lightgbm",
        model_version="test-v1",
        generated_at=cutoff + timedelta(minutes=1),
        forecast_origin_at=cutoff,
        data_cutoff_at=cutoff,
        horizon_hours=2,
        predictions=returned_order,
    )


def test_build_history_figure_converts_to_melbourne_and_preserves_response_order(
    history: HistoryResponse,
) -> None:
    figure = build_history_figure(history)

    assert len(figure.data) == 1
    observed = figure.data[0]
    assert observed.name == "Observed"
    assert list(observed.x) == [
        datetime(2026, 7, 12, 19, tzinfo=MELBOURNE_TIME_ZONE),
        datetime(2026, 7, 12, 18, tzinfo=MELBOURNE_TIME_ZONE),
    ]
    assert all(value.tzinfo is MELBOURNE_TIME_ZONE for value in observed.x)
    assert list(observed.y) == [31, 24]
    assert observed.line.dash == "solid"
    assert figure.layout.showlegend is not False
    assert figure.layout.xaxis.title.text == "Time"
    assert figure.layout.yaxis.title.text == "Pedestrian count"


def test_build_forecast_figure_labels_and_styles_observed_and_forecast(
    history: HistoryResponse,
    forecast: ForecastResponse,
) -> None:
    figure = build_forecast_figure(history=history, forecast=forecast)

    assert [trace.name for trace in figure.data] == ["Observed", "Forecast"]
    observed, predicted = figure.data
    assert observed.line.dash == "solid"
    assert predicted.line.dash == "dash"
    assert figure.layout.showlegend is not False
    assert figure.layout.xaxis.title.text == "Time"
    assert figure.layout.yaxis.title.text == "Pedestrian count"
    assert list(predicted.x) == [
        datetime(2026, 7, 12, 22, tzinfo=MELBOURNE_TIME_ZONE),
        datetime(2026, 7, 12, 21, tzinfo=MELBOURNE_TIME_ZONE),
    ]
    assert all(value.tzinfo is MELBOURNE_TIME_ZONE for value in predicted.x)
    assert list(predicted.y) == [12.75, 3.125]


def test_build_forecast_figure_supports_a_forecast_only_result(
    forecast: ForecastResponse,
) -> None:
    figure = build_forecast_figure(history=None, forecast=forecast)

    assert [trace.name for trace in figure.data] == ["Forecast"]
    assert figure.data[0].line.dash == "dash"
    assert figure.layout.showlegend is not False
    assert figure.layout.xaxis.title.text == "Time"
    assert figure.layout.yaxis.title.text == "Pedestrian count"
    assert list(figure.data[0].x) == [
        datetime(2026, 7, 12, 22, tzinfo=MELBOURNE_TIME_ZONE),
        datetime(2026, 7, 12, 21, tzinfo=MELBOURNE_TIME_ZONE),
    ]
    assert list(figure.data[0].y) == [12.75, 3.125]


def test_build_history_figure_rejects_a_naive_observed_timestamp(
    history: HistoryResponse,
) -> None:
    history.data[0].observed_at = datetime(2026, 7, 12, 9)

    with pytest.raises(ValueError, match="Chart timestamps must be offset-aware"):
        build_history_figure(history)


def test_build_forecast_figure_rejects_a_naive_target_timestamp(
    forecast: ForecastResponse,
) -> None:
    forecast.predictions[0].target_at = datetime(2026, 7, 12, 12)

    with pytest.raises(ValueError, match="Chart timestamps must be offset-aware"):
        build_forecast_figure(history=None, forecast=forecast)
