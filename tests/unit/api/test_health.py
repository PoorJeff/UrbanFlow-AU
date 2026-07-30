from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

import urbanflow
from tests.unit.api.helpers import api_get
from urbanflow.api.app import create_app
from urbanflow.api.schemas import ComponentHealth, HealthComponents, HealthResult
from urbanflow.api.services import (
    API_MAX_DATA_AGE_HOURS_ENV_VAR,
    ApiRuntimeConfigError,
    ApiServices,
    DataStoreUnavailableError,
    RuntimeHealthService,
    parse_max_data_age,
)


class FakeReadinessRepository:
    def __init__(self, cutoff: datetime | None = None, error: Exception | None = None) -> None:
        self.cutoff = cutoff
        self.error = error

    def get_latest_observed_at(self) -> datetime | None:
        if self.error is not None:
            raise self.error
        return self.cutoff


def fixed_now() -> datetime:
    return datetime(2026, 7, 30, 12, tzinfo=UTC)


def components(
    *,
    api_process: str = "available",
    model_provider: str = "unconfigured",
    data_store: str = "unconfigured",
    data_freshness: str = "unconfigured",
) -> HealthComponents:
    return HealthComponents(
        api_process=ComponentHealth(status=api_process),
        model_provider=ComponentHealth(status=model_provider),
        data_store=ComponentHealth(status=data_store),
        data_freshness=ComponentHealth(status=data_freshness),
    )


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, None),
        ({API_MAX_DATA_AGE_HOURS_ENV_VAR: ""}, None),
        ({API_MAX_DATA_AGE_HOURS_ENV_VAR: "   "}, None),
        ({API_MAX_DATA_AGE_HOURS_ENV_VAR: "1"}, timedelta(hours=1)),
        ({API_MAX_DATA_AGE_HOURS_ENV_VAR: "24"}, timedelta(hours=24)),
    ],
)
def test_parse_max_data_age_uses_only_the_passed_mapping(
    environ: dict[str, str],
    expected: timedelta | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_MAX_DATA_AGE_HOURS_ENV_VAR, "48")

    assert parse_max_data_age(environ) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-a-number"])
def test_parse_max_data_age_rejects_invalid_positive_integer_hours(value: str) -> None:
    with pytest.raises(ApiRuntimeConfigError, match="URBANFLOW_API_MAX_DATA_AGE_HOURS"):
        parse_max_data_age({API_MAX_DATA_AGE_HOURS_ENV_VAR: value})


def test_runtime_health_without_a_repository_reports_unconfigured_data_components() -> None:
    health = RuntimeHealthService(
        data_readiness_repository=None,
        model_provider_status="unconfigured",
        model_version=None,
        max_data_age=None,
        now=fixed_now,
    )()

    assert health.status == "degraded"
    assert health.components.data_store.status == "unconfigured"
    assert health.components.data_freshness.status == "unconfigured"
    assert health.data_cutoff_at is None


def test_runtime_health_reports_ok_for_fresh_aware_data_and_an_available_model() -> None:
    cutoff = datetime(2026, 7, 30, 11, tzinfo=UTC)
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(cutoff),
        model_provider_status="available",
        model_version="lightgbm-demo-v1",
        max_data_age=timedelta(hours=2),
        now=fixed_now,
    )()

    assert health.status == "ok"
    assert health.components.data_store.status == "available"
    assert health.components.data_freshness.status == "available"
    assert health.model_version == "lightgbm-demo-v1"
    assert health.data_cutoff_at == cutoff


def test_runtime_health_reports_stale_aware_data_as_unavailable_freshness() -> None:
    cutoff = datetime(2026, 7, 30, 9, tzinfo=UTC)
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(cutoff),
        model_provider_status="available",
        model_version="lightgbm-demo-v1",
        max_data_age=timedelta(hours=2),
        now=fixed_now,
    )()

    assert health.status == "degraded"
    assert health.components.data_store.status == "available"
    assert health.components.data_freshness.status == "unavailable"
    assert health.data_cutoff_at == cutoff


def test_runtime_health_reports_aware_data_without_a_threshold_as_unconfigured_freshness() -> None:
    cutoff = datetime(2026, 7, 30, 11, tzinfo=UTC)
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(cutoff),
        model_provider_status="available",
        model_version="lightgbm-demo-v1",
        max_data_age=None,
        now=fixed_now,
    )()

    assert health.status == "degraded"
    assert health.components.data_store.status == "available"
    assert health.components.data_freshness.status == "unconfigured"
    assert health.data_cutoff_at == cutoff


def test_runtime_health_reports_missing_cutoff_as_unavailable_freshness() -> None:
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(),
        model_provider_status="available",
        model_version="lightgbm-demo-v1",
        max_data_age=timedelta(hours=2),
        now=fixed_now,
    )()

    assert health.status == "degraded"
    assert health.components.data_store.status == "available"
    assert health.components.data_freshness.status == "unavailable"
    assert health.data_cutoff_at is None


def test_runtime_health_reports_data_store_failure_as_unavailable() -> None:
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(
            error=DataStoreUnavailableError("database unavailable")
        ),
        model_provider_status="available",
        model_version="lightgbm-demo-v1",
        max_data_age=timedelta(hours=2),
        now=fixed_now,
    )()

    assert health.status == "unavailable"
    assert health.components.data_store.status == "unavailable"
    assert health.components.data_freshness.status == "unavailable"
    assert health.data_cutoff_at is None


@pytest.mark.parametrize(
    "cutoff",
    [
        datetime(2026, 7, 30, 11),
        datetime(2026, 7, 30, 13, tzinfo=UTC),
    ],
)
def test_runtime_health_does_not_return_naive_or_future_cutoffs(cutoff: datetime) -> None:
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(cutoff),
        model_provider_status="available",
        model_version="lightgbm-demo-v1",
        max_data_age=timedelta(hours=2),
        now=fixed_now,
    )()

    assert health.status == "degraded"
    assert health.components.data_store.status == "available"
    assert health.components.data_freshness.status == "unavailable"
    assert health.data_cutoff_at is None


def test_runtime_health_keeps_readable_data_store_degraded_when_model_is_unavailable() -> None:
    health = RuntimeHealthService(
        data_readiness_repository=FakeReadinessRepository(datetime(2026, 7, 30, 11, tzinfo=UTC)),
        model_provider_status="unavailable",
        model_version=None,
        max_data_age=timedelta(hours=2),
        now=fixed_now,
    )()

    assert health.status == "degraded"
    assert health.components.data_store.status == "available"


def test_default_health_is_degraded_when_optional_components_are_unconfigured() -> None:
    response = api_get(create_app(), "/health")

    assert response.status_code == 200
    payload = response.json()
    generated_at = datetime.fromisoformat(payload.pop("generated_at"))
    assert generated_at.tzinfo is not None
    assert generated_at.utcoffset() == UTC.utcoffset(generated_at)
    assert payload == {
        "status": "degraded",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "components": {
            "api_process": {"status": "available"},
            "model_provider": {"status": "unconfigured"},
            "data_store": {"status": "unconfigured"},
            "data_freshness": {"status": "unconfigured"},
        },
        "model_version": None,
        "data_cutoff_at": None,
    }


def test_health_returns_200_when_injected_result_is_ok() -> None:
    generated_at = datetime(2026, 7, 12, 10, 30, tzinfo=UTC)
    data_cutoff_at = datetime(2026, 7, 12, 9, 0, tzinfo=UTC)
    services = ApiServices(
        health=lambda: HealthResult(
            status="ok",
            service="urbanflow-au-api",
            version=urbanflow.__version__,
            generated_at=generated_at,
            components=components(
                api_process="available",
                model_provider="available",
                data_store="available",
                data_freshness="available",
            ),
            model_version="lightgbm-demo-v1",
            data_cutoff_at=data_cutoff_at,
        )
    )

    response = api_get(create_app(services=services), "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "generated_at": "2026-07-12T10:30:00Z",
        "components": {
            "api_process": {"status": "available"},
            "model_provider": {"status": "available"},
            "data_store": {"status": "available"},
            "data_freshness": {"status": "available"},
        },
        "model_version": "lightgbm-demo-v1",
        "data_cutoff_at": "2026-07-12T09:00:00Z",
    }


def test_health_returns_503_when_injected_result_is_unavailable() -> None:
    services = ApiServices(
        health=lambda: HealthResult(
            status="unavailable",
            service="urbanflow-au-api",
            version=urbanflow.__version__,
            generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=UTC),
            components=components(api_process="unavailable"),
        )
    )

    response = api_get(create_app(services=services), "/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "service": "urbanflow-au-api",
        "version": urbanflow.__version__,
        "generated_at": "2026-07-12T10:30:00Z",
        "components": {
            "api_process": {"status": "unavailable"},
            "model_provider": {"status": "unconfigured"},
            "data_store": {"status": "unconfigured"},
            "data_freshness": {"status": "unconfigured"},
        },
        "model_version": None,
        "data_cutoff_at": None,
    }


def test_health_result_requires_every_component_status() -> None:
    with pytest.raises(ValidationError, match="components"):
        HealthResult(
            status="ok",
            service="urbanflow-au-api",
            version=urbanflow.__version__,
            generated_at=datetime(2026, 7, 12, 10, 30, tzinfo=UTC),
        )
