from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260628_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sensor_dim",
        sa.Column("location_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("sensor_name", sa.Text(), nullable=False),
        sa.Column("sensor_description", sa.Text(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("installation_date", sa.Date(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "latitude >= -90 AND latitude <= 90",
            name="ck_sensor_dim_latitude_range",
        ),
        sa.CheckConstraint(
            "longitude >= -180 AND longitude <= 180",
            name="ck_sensor_dim_longitude_range",
        ),
        sa.PrimaryKeyConstraint("location_id"),
    )
    op.create_table(
        "pedestrian_hourly_fact",
        sa.Column("location_id", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_sensing_date", sa.Date(), nullable=False),
        sa.Column("source_hourday", sa.Integer(), nullable=False),
        sa.Column("pedestrian_count", sa.Integer(), nullable=False),
        sa.Column("direction_1_count", sa.Integer(), nullable=False),
        sa.Column("direction_2_count", sa.Integer(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("source_snapshot_path", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "pedestrian_count >= 0",
            name="ck_pedestrian_hourly_fact_pedestrian_count_non_negative",
        ),
        sa.CheckConstraint(
            "direction_1_count >= 0",
            name="ck_pedestrian_hourly_fact_direction_1_count_non_negative",
        ),
        sa.CheckConstraint(
            "direction_2_count >= 0",
            name="ck_pedestrian_hourly_fact_direction_2_count_non_negative",
        ),
        sa.CheckConstraint(
            "source_hourday >= 0 AND source_hourday <= 23",
            name="ck_pedestrian_hourly_fact_source_hourday_range",
        ),
        sa.ForeignKeyConstraint(["location_id"], ["sensor_dim.location_id"]),
        sa.PrimaryKeyConstraint("location_id", "observed_at"),
    )


def downgrade() -> None:
    op.drop_table("pedestrian_hourly_fact")
    op.drop_table("sensor_dim")
