from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime

import pytest

from urbanflow.api import lightgbm_forecast_smoke
from urbanflow.modeling.lightgbm_artifact import LightGBMArtifactError


def test_validate_smoke_schema_name_accepts_generated_lowercase_name() -> None:
    schema_name = lightgbm_forecast_smoke._temporary_schema_name()

    assert lightgbm_forecast_smoke.validate_smoke_schema_name(schema_name) == schema_name


@pytest.mark.parametrize(
    "schema_name",
    [
        "UrbanFlow_smoke",
        "urbanflow-smoke",
        "urbanflow_smoke;drop schema public",
        "1urbanflow_smoke",
        "urbanflow smoke",
        "a" * 64,
    ],
)
def test_validate_smoke_schema_name_rejects_dangerous_identifiers(schema_name: str) -> None:
    with pytest.raises(ValueError, match="safe PostgreSQL identifier"):
        lightgbm_forecast_smoke.validate_smoke_schema_name(schema_name)


def test_lightgbm_forecast_smoke_cli_returns_two_when_database_url_is_missing(capsys) -> None:
    exit_code = lightgbm_forecast_smoke.main([], environ={})

    assert exit_code == 2
    assert lightgbm_forecast_smoke.SMOKE_DATABASE_URL_ENV_VAR in capsys.readouterr().err


def test_lightgbm_forecast_smoke_parser_accepts_optional_schema_name() -> None:
    args = lightgbm_forecast_smoke.build_parser().parse_args(
        [
            "--database-url",
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            "--schema-name",
            "urbanflow_lightgbm_smoke_test",
        ]
    )

    assert args.schema_name == "urbanflow_lightgbm_smoke_test"
    assert vars(args) == {
        "database_url": "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
        "schema_name": "urbanflow_lightgbm_smoke_test",
    }


def test_lightgbm_forecast_smoke_result_is_json_safe() -> None:
    cutoff = datetime(2026, 7, 12, 10, tzinfo=UTC).isoformat()
    result = lightgbm_forecast_smoke.LightGBMForecastSmokeResult(
        schema_name="urbanflow_lightgbm_smoke_test",
        location_id=999001,
        data_cutoff_at=cutoff,
        forecast_horizons=list(range(1, 25)),
        model_version="lightgbm-0123456789ab",
    )

    encoded = json.dumps(asdict(result), sort_keys=True)

    assert json.loads(encoded)["data_cutoff_at"] == cutoff


def test_lightgbm_forecast_smoke_cli_returns_one_for_artifact_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_with_artifact_error(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> lightgbm_forecast_smoke.LightGBMForecastSmokeResult:
        raise LightGBMArtifactError(f"invalid artifact for {database_url} {schema_name}")

    monkeypatch.setattr(
        lightgbm_forecast_smoke,
        "run_lightgbm_forecast_smoke",
        fail_with_artifact_error,
    )

    exit_code = lightgbm_forecast_smoke.main(
        ["--database-url", "postgresql+psycopg://localhost/urbanflow"],
        environ={},
    )

    assert exit_code == 1
    assert "LightGBM forecast smoke test failed" in capsys.readouterr().err


def test_lightgbm_forecast_smoke_cli_returns_two_for_configuration_value_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_with_value_error(
        database_url: str,
        *,
        schema_name: str | None = None,
    ) -> lightgbm_forecast_smoke.LightGBMForecastSmokeResult:
        raise ValueError(f"bad configuration for {database_url} {schema_name}")

    monkeypatch.setattr(
        lightgbm_forecast_smoke,
        "run_lightgbm_forecast_smoke",
        fail_with_value_error,
    )

    exit_code = lightgbm_forecast_smoke.main(
        ["--database-url", "postgresql+psycopg://localhost/urbanflow"],
        environ={},
    )

    assert exit_code == 2
    assert "bad configuration" in capsys.readouterr().err
