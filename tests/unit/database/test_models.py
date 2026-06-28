from sqlalchemy import CheckConstraint, ForeignKeyConstraint, PrimaryKeyConstraint

from urbanflow.database.models import Base, PedestrianHourlyFact, SensorDim


def test_sensor_dim_table_contract() -> None:
    table = SensorDim.__table__

    assert table.name == "sensor_dim"
    assert table.c.location_id.primary_key is True
    assert table.c.location_id.autoincrement is False
    assert table.c.installation_date.nullable is True
    assert table.c.updated_at.nullable is False

    primary_keys = [
        constraint
        for constraint in table.constraints
        if isinstance(constraint, PrimaryKeyConstraint)
    ]
    assert len(primary_keys) == 1

    checks = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert checks == {"ck_sensor_dim_latitude_range", "ck_sensor_dim_longitude_range"}


def test_pedestrian_hourly_fact_table_contract() -> None:
    table = PedestrianHourlyFact.__table__

    assert table.name == "pedestrian_hourly_fact"
    assert [column.name for column in table.primary_key.columns] == [
        "location_id",
        "observed_at",
    ]
    assert table.c.source_snapshot_path.nullable is False

    foreign_keys = [
        constraint
        for constraint in table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]
    assert len(foreign_keys) == 1
    assert list(foreign_keys[0].columns)[0].name == "location_id"

    checks = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert checks == {
        "ck_pedestrian_hourly_fact_pedestrian_count_non_negative",
        "ck_pedestrian_hourly_fact_direction_1_count_non_negative",
        "ck_pedestrian_hourly_fact_direction_2_count_non_negative",
        "ck_pedestrian_hourly_fact_source_hourday_range",
    }


def test_metadata_contains_only_core_tables() -> None:
    assert set(Base.metadata.tables) == {"sensor_dim", "pedestrian_hourly_fact"}
