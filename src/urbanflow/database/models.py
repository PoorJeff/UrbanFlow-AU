from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, Float, ForeignKey, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base metadata for UrbanFlow database models."""


class SensorDim(Base):
    __tablename__ = "sensor_dim"
    __table_args__ = (
        CheckConstraint(
            "latitude >= -90 AND latitude <= 90",
            name="ck_sensor_dim_latitude_range",
        ),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180",
            name="ck_sensor_dim_longitude_range",
        ),
    )

    location_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    sensor_name: Mapped[str] = mapped_column(Text, nullable=False)
    sensor_description: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    installation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PedestrianHourlyFact(Base):
    __tablename__ = "pedestrian_hourly_fact"
    __table_args__ = (
        CheckConstraint(
            "pedestrian_count >= 0",
            name="ck_pedestrian_hourly_fact_pedestrian_count_non_negative",
        ),
        CheckConstraint(
            "direction_1_count >= 0",
            name="ck_pedestrian_hourly_fact_direction_1_count_non_negative",
        ),
        CheckConstraint(
            "direction_2_count >= 0",
            name="ck_pedestrian_hourly_fact_direction_2_count_non_negative",
        ),
        CheckConstraint(
            "source_hourday >= 0 AND source_hourday <= 23",
            name="ck_pedestrian_hourly_fact_source_hourday_range",
        ),
    )

    location_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sensor_dim.location_id"),
        primary_key=True,
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    source_sensing_date: Mapped[date] = mapped_column(Date, nullable=False)
    source_hourday: Mapped[int] = mapped_column(Integer, nullable=False)
    pedestrian_count: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_1_count: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_2_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    source_snapshot_path: Mapped[str] = mapped_column(Text, nullable=False)
