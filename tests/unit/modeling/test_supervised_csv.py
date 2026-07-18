from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from urbanflow.modeling.supervised_csv import SupervisedCsvError, read_supervised_csv, sha256_file


def test_read_supervised_csv_preserves_offset_aware_instants_across_dst(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.csv"
    path.write_text(
        "forecast_origin_at,target_observed_at,value\n"
        "2025-04-06T01:30:00+11:00,2025-04-06T02:30:00+11:00,1\n"
        "2025-04-06T02:30:00+10:00,2025-04-06T03:30:00+10:00,2\n",
        encoding="utf-8",
    )

    frame = read_supervised_csv(path)

    assert str(frame["forecast_origin_at"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["target_observed_at"].dtype) == "datetime64[ns, UTC]"
    assert frame["forecast_origin_at"].tolist() == [
        pd.Timestamp("2025-04-05T14:30:00Z"),
        pd.Timestamp("2025-04-05T16:30:00Z"),
    ]
    assert frame["target_observed_at"].tolist() == [
        pd.Timestamp("2025-04-05T15:30:00Z"),
        pd.Timestamp("2025-04-05T17:30:00Z"),
    ]


@pytest.mark.parametrize(
    "contents",
    [
        "forecast_origin_at,target_observed_at\nnot-a-time,2025-01-01T01:00:00+11:00\n",
        "forecast_origin_at,target_observed_at\n2025-01-01T00:00:00,2025-01-01T01:00:00+11:00\n",
    ],
)
def test_read_supervised_csv_rejects_invalid_or_naive_timestamp(
    tmp_path: Path,
    contents: str,
) -> None:
    path = tmp_path / "rows.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(SupervisedCsvError, match="forecast_origin_at"):
        read_supervised_csv(path)


def test_read_supervised_csv_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(SupervisedCsvError, match="CSV file does not exist"):
        read_supervised_csv(tmp_path / "missing.csv")


def test_read_supervised_csv_rejects_malformed_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text('forecast_origin_at,value\n"unterminated,1\n', encoding="utf-8")

    with pytest.raises(SupervisedCsvError, match="could not read supervised CSV"):
        read_supervised_csv(path)


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(b"a,b\r\n1,2\r\n")

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
