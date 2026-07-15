from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from urbanflow.api.postgres import PostgresSensorHistoryRepository
from urbanflow.api.services import DataStoreUnavailableError, HistoryRecord, SensorRecord
from urbanflow.database.models import PedestrianHourlyFact, SensorDim


class FakeScalarResult:
    def __init__(
        self,
        rows: list[object],
        *,
        result_error: SQLAlchemyError | None = None,
    ) -> None:
        self._rows = rows
        self._result_error = result_error
        self.all_called = False
        self.one_or_none_called = False

    def all(self) -> list[object]:
        self.all_called = True
        if self._result_error is not None:
            raise self._result_error
        return list(self._rows)

    def one_or_none(self) -> object | None:
        self.one_or_none_called = True
        if self._result_error is not None:
            raise self._result_error
        return self._rows[0] if self._rows else None


class FakeSession:
    def __init__(
        self,
        rows: list[object] | None = None,
        *,
        scalars_error: SQLAlchemyError | None = None,
        result_error: SQLAlchemyError | None = None,
    ) -> None:
        self._rows = rows or []
        self._scalars_error = scalars_error
        self._result_error = result_error
        self.statements: list[object] = []
        self.scalar_result: FakeScalarResult | None = None

    def scalars(self, statement: object) -> FakeScalarResult:
        self.statements.append(statement)
        if self._scalars_error is not None:
            raise self._scalars_error
        self.scalar_result = FakeScalarResult(self._rows, result_error=self._result_error)
        return self.scalar_result


RepositoryCall = Callable[[PostgresSensorHistoryRepository], object]


def _repository(session: FakeSession) -> PostgresSensorHistoryRepository:
    return PostgresSensorHistoryRepository(lambda: session)


def _compile(statement: object) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
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
        location_id=101,
        observed_at=observed_at,
        source_sensing_date=observed_at.date(),
        source_hourday=observed_at.hour,
        pedestrian_count=pedestrian_count,
        direction_1_count=pedestrian_count // 2,
        direction_2_count=pedestrian_count - pedestrian_count // 2,
        source_snapshot_path="records.csv",
    )


def test_list_sensors_maps_active_and_inactive_rows() -> None:
    active_sensor = _sensor(101, status="A")
    inactive_sensor = _sensor(102, status="I")

    records = _repository(FakeSession([active_sensor, inactive_sensor])).list_sensors(
        active_only=False
    )

    assert records == [
        SensorRecord(
            location_id=101,
            sensor_name="Sensor 101",
            sensor_description="Description 101",
            status="A",
            latitude=-37.8,
            longitude=144.9,
        ),
        SensorRecord(
            location_id=102,
            sensor_name="Sensor 102",
            sensor_description="Description 102",
            status="I",
            latitude=-37.8,
            longitude=144.9,
        ),
    ]


def test_list_sensors_active_query_filters_active_status_and_orders_by_location_id() -> None:
    session = FakeSession()

    _repository(session).list_sensors(active_only=True)

    sql = _compile(session.statements[0])
    assert "sensor_dim.status = 'A'" in sql
    assert "ORDER BY sensor_dim.location_id" in sql


def test_list_sensors_without_active_filter_has_no_status_predicate() -> None:
    session = FakeSession()

    _repository(session).list_sensors(active_only=False)

    sql = _compile(session.statements[0])
    assert "WHERE sensor_dim.status" not in sql


def test_get_sensor_maps_one_row_and_returns_none_for_no_row() -> None:
    sensor = _sensor(101, status="A")
    session = FakeSession([sensor])

    record = _repository(session).get_sensor(101)
    missing_record = _repository(FakeSession()).get_sensor(999)

    assert record == SensorRecord(
        location_id=101,
        sensor_name="Sensor 101",
        sensor_description="Description 101",
        status="A",
        latitude=-37.8,
        longitude=144.9,
    )
    assert missing_record is None
    assert session.scalar_result is not None
    assert session.scalar_result.one_or_none_called
    assert "WHERE sensor_dim.location_id = 101" in _compile(session.statements[0])


def test_get_history_orders_aware_rows_and_compiles_half_open_range_query() -> None:
    start = datetime(2026, 7, 2, 0, tzinfo=UTC)
    end = datetime(2026, 7, 3, 0, tzinfo=UTC)
    source_later = _fact(
        datetime(2026, 7, 2, 12, tzinfo=timezone(timedelta(hours=10))),
        pedestrian_count=25,
    )
    source_earlier = _fact(datetime(2026, 7, 2, 1, tzinfo=UTC), pedestrian_count=10)
    session = FakeSession([source_later, source_earlier])

    records = _repository(session).get_history(101, start, end)

    assert records == [
        HistoryRecord(
            observed_at=source_earlier.observed_at,
            pedestrian_count=10,
        ),
        HistoryRecord(
            observed_at=source_later.observed_at,
            pedestrian_count=25,
        ),
    ]
    for record, source in zip(records, [source_earlier, source_later], strict=True):
        assert record.observed_at.tzinfo is not None
        assert record.observed_at.utcoffset() is not None
        assert record.observed_at.astimezone(UTC) == source.observed_at.astimezone(UTC)
    sql = _compile(session.statements[0])
    assert "pedestrian_hourly_fact.location_id = 101" in sql
    assert "pedestrian_hourly_fact.observed_at >= '2026-07-02 00:00:00+00:00'" in sql
    assert "pedestrian_hourly_fact.observed_at < '2026-07-03 00:00:00+00:00'" in sql
    assert "ORDER BY pedestrian_hourly_fact.observed_at" in sql


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda repository: repository.list_sensors(active_only=True),
            id="list_sensors",
        ),
        pytest.param(lambda repository: repository.get_sensor(101), id="get_sensor"),
        pytest.param(
            lambda repository: repository.get_history(
                101,
                datetime(2026, 7, 2, 0, tzinfo=UTC),
                datetime(2026, 7, 3, 0, tzinfo=UTC),
            ),
            id="get_history",
        ),
    ],
)
def test_public_methods_translate_session_factory_failures(call: RepositoryCall) -> None:
    def failing_session_factory() -> FakeSession:
        raise SQLAlchemyError("session factory failed")

    repository = PostgresSensorHistoryRepository(failing_session_factory)

    with pytest.raises(DataStoreUnavailableError):
        call(repository)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda repository: repository.list_sensors(active_only=True),
            id="list_sensors",
        ),
        pytest.param(lambda repository: repository.get_sensor(101), id="get_sensor"),
        pytest.param(
            lambda repository: repository.get_history(
                101,
                datetime(2026, 7, 2, 0, tzinfo=UTC),
                datetime(2026, 7, 3, 0, tzinfo=UTC),
            ),
            id="get_history",
        ),
    ],
)
def test_public_methods_translate_scalars_failures(call: RepositoryCall) -> None:
    repository = _repository(FakeSession(scalars_error=SQLAlchemyError("query failed")))

    with pytest.raises(DataStoreUnavailableError):
        call(repository)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda repository: repository.list_sensors(active_only=True),
            id="list_sensors",
        ),
        pytest.param(
            lambda repository: repository.get_history(
                101,
                datetime(2026, 7, 2, 0, tzinfo=UTC),
                datetime(2026, 7, 3, 0, tzinfo=UTC),
            ),
            id="get_history",
        ),
    ],
)
def test_all_failures_become_data_store_unavailable(call: RepositoryCall) -> None:
    repository = _repository(FakeSession(result_error=SQLAlchemyError("result failed")))

    with pytest.raises(DataStoreUnavailableError):
        call(repository)


def test_one_or_none_failures_become_data_store_unavailable() -> None:
    repository = _repository(FakeSession(result_error=SQLAlchemyError("result failed")))

    with pytest.raises(DataStoreUnavailableError):
        repository.get_sensor(101)
