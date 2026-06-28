import pytest

from urbanflow.database.config import DatabaseConfigError, get_database_url


def test_get_database_url_prefers_explicit_value() -> None:
    url = get_database_url(
        database_url="postgresql+psycopg://user:pass@localhost:5432/urbanflow",
        environ={"URBANFLOW_DATABASE_URL": "postgresql+psycopg://ignored"},
    )

    assert url == "postgresql+psycopg://user:pass@localhost:5432/urbanflow"


def test_get_database_url_reads_environment_value() -> None:
    url = get_database_url(
        database_url=None,
        environ={"URBANFLOW_DATABASE_URL": "postgresql+psycopg://env"},
    )

    assert url == "postgresql+psycopg://env"


def test_get_database_url_rejects_missing_value() -> None:
    with pytest.raises(DatabaseConfigError, match="Database URL is required"):
        get_database_url(database_url=None, environ={})


def test_get_database_url_rejects_blank_value() -> None:
    with pytest.raises(DatabaseConfigError, match="Database URL is required"):
        get_database_url(database_url="  ", environ={})
