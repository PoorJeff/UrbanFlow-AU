import json
from pathlib import Path

import pytest

from tests.unit.api.helpers import api_get
from urbanflow.api.app import create_app


def evaluation_summary(model_name: str) -> dict[str, object]:
    model_key = f"{model_name}_wape"
    return {
        "final_test": {
            "name": "final_test_2025-02",
            "start": "2025-02-01T00:00:00+11:00",
            "end": "2025-03-01T00:00:00+11:00",
            "overall": {"mae": 1.2, "rmse": 1.7, "wape": 0.07},
            "seasonal_naive_overall": {"wape": 0.095},
            "model_comparison": {
                model_key: 0.07,
                "seasonal_naive_wape": 0.095,
                "relative_wape_improvement": 0.2631578947368421,
            },
        }
    }


def configure_metrics_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: object,
) -> Path:
    path = tmp_path / "evaluation-summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("URBANFLOW_API_METRICS_PATH", str(path))
    return path


def test_model_metrics_returns_ridge_summary_with_honest_null_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_metrics_path(monkeypatch, tmp_path, evaluation_summary("ridge"))

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "ridge",
        "model_version": None,
        "evaluation_source": "evaluation_summary",
        "final_test_window": {
            "name": "final_test_2025-02",
            "start": "2025-02-01T00:00:00+11:00",
            "end": "2025-03-01T00:00:00+11:00",
        },
        "metrics": {
            "mae": 1.2,
            "rmse": 1.7,
            "wape": 0.07,
            "seasonal_naive_wape": 0.095,
            "relative_wape_improvement": 0.2631578947368421,
        },
        "mlflow_run_id": None,
        "mlflow_tracking_uri": None,
        "report_path": None,
    }


def test_model_metrics_returns_lightgbm_summary_and_optional_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = evaluation_summary("lightgbm")
    summary.update(
        {
            "model_version": "lightgbm-demo-v1",
            "mlflow_run_id": "run-123",
            "mlflow_tracking_uri": "file:///tmp/mlruns",
            "report_path": "reports/lightgbm.md",
        }
    )
    configure_metrics_path(monkeypatch, tmp_path, summary)

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "lightgbm"
    assert payload["model_version"] == "lightgbm-demo-v1"
    assert payload["mlflow_run_id"] == "run-123"
    assert payload["mlflow_tracking_uri"] == "file:///tmp/mlruns"
    assert payload["report_path"] == "reports/lightgbm.md"


def test_model_metrics_without_a_configured_path_returns_project_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("URBANFLOW_API_METRICS_PATH", raising=False)

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "metrics_unavailable",
        "message": "Model metrics are unavailable.",
        "details": [],
    }


@pytest.mark.parametrize("path_name", ["missing-summary.json", "."])
def test_model_metrics_returns_project_error_for_unreadable_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path_name: str,
) -> None:
    path = tmp_path / path_name
    monkeypatch.setenv("URBANFLOW_API_METRICS_PATH", str(path))
    application = create_app()

    health_response = api_get(application, "/health")
    metrics_response = api_get(application, "/api/v1/model/metrics")

    assert health_response.status_code == 200
    assert metrics_response.status_code == 503
    assert metrics_response.json()["error"]["code"] == "metrics_unavailable"


def test_model_metrics_returns_project_error_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("URBANFLOW_API_METRICS_PATH", str(path))

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "metrics_unavailable"


def test_model_metrics_returns_project_error_for_invalid_json_encoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-encoding.json"
    path.write_bytes(b"\xff")
    monkeypatch.setenv("URBANFLOW_API_METRICS_PATH", str(path))

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "metrics_unavailable"


def test_model_metrics_returns_project_error_for_missing_required_summary_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_metrics_path(monkeypatch, tmp_path, {"final_test": {}})

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "metrics_unavailable"


def test_model_metrics_returns_project_error_for_unrepresentable_metric_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = evaluation_summary("ridge")
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    overall = final_test["overall"]
    assert isinstance(overall, dict)
    overall["mae"] = int("9" * 400)
    configure_metrics_path(monkeypatch, tmp_path, summary)

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "metrics_unavailable"


@pytest.mark.parametrize(
    "comparison",
    [
        {
            "ridge_wape": 0.07,
            "lightgbm_wape": 0.06,
            "seasonal_naive_wape": 0.095,
            "relative_wape_improvement": 0.3,
        },
        {
            "seasonal_naive_wape": 0.095,
            "relative_wape_improvement": 0.3,
        },
    ],
    ids=["ambiguous-model", "unsupported-model"],
)
def test_model_metrics_requires_exactly_one_supported_model_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    comparison: dict[str, float],
) -> None:
    summary = evaluation_summary("ridge")
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    final_test["model_comparison"] = comparison
    configure_metrics_path(monkeypatch, tmp_path, summary)

    response = api_get(create_app(), "/api/v1/model/metrics")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "metrics_unavailable"
