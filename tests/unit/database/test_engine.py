from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from urbanflow.database.engine import create_database_engine, create_session_factory


def test_create_database_engine_returns_sqlalchemy_engine() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")

    assert isinstance(engine, Engine)


def test_create_session_factory_opens_sessions() -> None:
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        assert isinstance(session, Session)
