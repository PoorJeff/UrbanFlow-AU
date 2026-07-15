from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import BinaryExpression

import urbanflow.api.app as app_module
from tests.unit.api.helpers import api_get
from urbanflow.api.app import create_app
from urbanflow.api.errors import UrbanFlowApiError
from urbanflow.api.postgres import PostgresSensorHistoryRepository
from urbanflow.api.services import ApiServices, EmptyHistoryRepository, EmptySensorRepository
from urbanflow.database.config import DATABASE_URL_ENV_VAR, DatabaseConfigError
from urbanflow.database.models import PedestrianHourlyFact, SensorDim


class FakeScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return list(self._rows)

    def one_or_none(self) -> object | None:
        return self._rows[0] if self._rows else None


class StatementAwareSession:
    def __init__(
        self,
        *,
        active_sensors: list[SensorDim],
        inactive_sensors: list[SensorDim],
        facts: list[PedestrianHourlyFact],
    ) -> None:
        self._active_sensors = active_sensors
        self._sensors = [*active_sensors, *inactive_sensors]
        self._facts = facts
        self.closed = False

    def __enter__(self) -> StatementAwareSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def scalars(self, statement: object) -> FakeScalarResult:
        entity = statement.column_descriptions[0]["entity"]
        if entity is SensorDim:
            rows = self._active_sensors if _has_active_sensor_filter(statement) else self._sensors
            return FakeScalarResult(rows)
        if entity is PedestrianHourlyFact:
            return FakeScalarResult(self._facts)
        raise AssertionError(f"Unexpected ORM entity: {entity}")


class StatementAwareSessionFactory:
    def __init__(
        self,
        *,
        active_sensors: list[SensorDim],
        inactive_sensors: list[SensorDim],
        facts: list[PedestrianHourlyFact],
    ) -> None:
        self._active_sensors = active_sensors
        self._inactive_sensors = inactive_sensors
        self._facts = facts
        self.sessions: list[StatementAwareSession] = []

    def __call__(self) -> StatementAwareSession:
        session = StatementAwareSession(
            active_sensors=self._active_sensors,
            inactive_sensors=self._inactive_sensors,
            facts=self._facts,
        )
        self.sessions.append(session)
        return session


def _has_active_sensor_filter(statement: object) -> bool:
    return any(
        isinstance(criterion, BinaryExpression)
        and criterion.operator is operators.eq
        and criterion.left.compare(SensorDim.status.expression)
        and criterion.right.value == "A"
        for criterion in statement._where_criteria
    )


def _sensor(location_id: int, *, status: str) -> SensorDim:
    return SensorDim(
        location_id=location_id,
        sensor_name=f"Sensor {location_id}",
        sensor_description=f"Description {location_id}",
        latitude=-37.8,
        longitude=144.9,
        installation_date=date(2020, 1, 2),
        status=status,
    )


def _fact(observed_at: datetime, *, pedestrian_count: int) -> PedestrianHourlyFact:
    return PedestrianHourlyFact(
        location_id=999001,
        observed_at=observed_at,
        source_sensing_date=observed_at.date(),
        source_hourday=observed_at.hour,
        pedestrian_count=pedestrian_count,
        direction_1_count=pedestrian_count // 2,
        direction_2_count=pedestrian_count - pedestrian_count // 2,
        source_snapshot_path="records.csv",
    )


def test_create_default_services_uses_empty_repositories_without_a_database_url() -> None:
    missing_url_services = app_module.create_default_services(environ={})
    whitespace_url_services = app_module.create_default_services(
        environ={DATABASE_URL_ENV_VAR: " \t "}
    )

    for services in (missing_url_services, whitespace_url_services):
        assert isinstance(services.sensor_repository, EmptySensorRepository)
        assert isinstance(services.history_repository, EmptyHistoryRepository)


def test_create_default_services_builds_one_shared_postgres_repository_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object()
    engine_urls: list[str] = []
    session_factory_engines: list[object] = []
    session_factory_called = False

    def fake_create_database_engine(database_url: str) -> object:
        engine_urls.append(database_url)
        return engine

    def session_factory() -> StatementAwareSession:
        nonlocal session_factory_called
        session_factory_called = True
        raise AssertionError("default service construction must not open a session")

    def fake_create_session_factory(received_engine: object) -> object:
        session_factory_engines.append(received_engine)
        return session_factory

    monkeypatch.setattr(app_module, "create_database_engine", fake_create_database_engine)
    monkeypatch.setattr(app_module, "create_session_factory", fake_create_session_factory)

    services = app_module.create_default_services(
        environ={DATABASE_URL_ENV_VAR: "  postgresql+psycopg://user:pass@db/urbanflow  "}
    )

    assert engine_urls == ["postgresql+psycopg://user:pass@db/urbanflow"]
    assert session_factory_engines == [engine]
    assert isinstance(services.sensor_repository, PostgresSensorHistoryRepository)
    assert services.sensor_repository is services.history_repository
    assert not session_factory_called


def test_default_app_reads_sensor_and_history_data_through_postgres_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_sensor = _sensor(999001, status="A")
    inactive_sensor = _sensor(999002, status="I")
    session_factory = StatementAwareSessionFactory(
        active_sensors=[active_sensor],
        inactive_sensors=[inactive_sensor],
        facts=[
            _fact(
                datetime(2026, 1, 1, 20, tzinfo=timezone(timedelta(hours=10))),
                pedestrian_count=42,
            ),
            _fact(datetime(2026, 1, 1, 2, tzinfo=UTC), pedestrian_count=7),
        ],
    )
    monkeypatch.setenv(
        DATABASE_URL_ENV_VAR,
        "postgresql+psycopg://user:pass@db/urbanflow",
    )
    monkeypatch.setattr(app_module, "create_database_engine", lambda _url: object())
    monkeypatch.setattr(
        app_module,
        "create_session_factory",
        lambda _engine: session_factory,
    )

    application = app_module.create_app()

    assert session_factory.sessions == []
    sensors = api_get(application, "/api/v1/sensors", params={"active_only": "true"})
    history = api_get(
        application,
        "/api/v1/sensors/999001/history",
        params={
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-02T00:00:00+00:00",
        },
    )

    assert sensors.status_code == 200
    assert [row["location_id"] for row in sensors.json()["data"]] == [999001]
    assert history.status_code == 200
    assert [row["pedestrian_count"] for row in history.json()["data"]] == [7, 42]
    assert len(session_factory.sessions) == 3
    assert all(session.closed for session in session_factory.sessions)


def test_create_default_services_rejects_an_invalid_database_url() -> None:
    with pytest.raises(DatabaseConfigError, match=f"Invalid {DATABASE_URL_ENV_VAR} configuration"):
        app_module.create_default_services(environ={DATABASE_URL_ENV_VAR: "not-a-sqlalchemy-url"})


def test_default_app_maps_lazy_database_read_failures_to_a_project_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory_calls = 0

    def failing_session_factory() -> StatementAwareSession:
        nonlocal session_factory_calls
        session_factory_calls += 1
        raise OperationalError("SELECT 1", {}, OSError("database is unavailable"))

    monkeypatch.setenv(
        DATABASE_URL_ENV_VAR,
        "postgresql+psycopg://user:pass@unreachable/urbanflow",
    )
    monkeypatch.setattr(
        app_module,
        "create_session_factory",
        lambda _engine: failing_session_factory,
    )

    application = app_module.create_app()

    assert session_factory_calls == 0
    response = api_get(application, "/api/v1/sensors")

    assert session_factory_calls == 1
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "data_store_unavailable"


def test_explicit_services_bypass_default_database_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected_services = ApiServices()
    monkeypatch.setenv(DATABASE_URL_ENV_VAR, "not-a-sqlalchemy-url")

    application = app_module.create_app(services=injected_services)

    assert application.state.services is injected_services


def test_create_app_exposes_the_first_fastapi_serving_routes() -> None:
    application = create_app()

    assert isinstance(application, FastAPI)
    assert set(application.openapi()["paths"]) == {
        "/health",
        "/api/v1/sensors",
        "/api/v1/sensors/{location_id}/history",
        "/api/v1/sensors/{location_id}/forecast",
        "/api/v1/model/metrics",
    }


def test_project_errors_use_the_standard_error_response() -> None:
    application = create_app()

    @application.get("/_test/project-error")
    def raise_project_error() -> None:
        raise UrbanFlowApiError(
            status_code=503,
            code="model_unavailable",
            message="No model provider is configured.",
        )

    response = api_get(application, "/_test/project-error")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "model_unavailable",
            "message": "No model provider is configured.",
            "details": [],
        }
    }


def test_request_validation_keeps_fastapi_response_shape() -> None:
    application = create_app()

    @application.get("/_test/requires-integer")
    def requires_integer(value: int) -> dict[str, int]:
        return {"value": value}

    response = api_get(application, "/_test/requires-integer", params={"value": "bad"})

    assert response.status_code == 422
    assert "detail" in response.json()
    assert "error" not in response.json()
