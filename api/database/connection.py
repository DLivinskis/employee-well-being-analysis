"""
Database connection and session management for the FastAPI service.

One `Database` instance, built from `APIConfig`, is shared by `seed.py` and
(in a later phase) the API's routers — so the engine and session factory
are created once rather than per-call.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import APIConfig
from api.database.models import Base


class Database:
    """Owns the SQLAlchemy engine, session factory, and schema creation.

    Entry points:
        create_all: Create all tables that don't already exist.
        session: Context manager yielding a Session, committed on success.

    Args:
        config: API configuration holding the database URL.
    """

    def __init__(self, config: APIConfig) -> None:
        self._engine: Engine = create_engine(config.database_url)
        self._session_factory = sessionmaker(bind=self._engine)

    def create_all(self) -> None:
        """Create the `employees` and `models` tables if they don't exist."""
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a Session, committing on success and rolling back on error.

        Yields:
            An open SQLAlchemy Session, closed automatically on exit.
        """
        # Context manager: callers get `with database.session() as session:`,
        # which guarantees the session is committed/rolled-back and closed
        # even if the caller's code raises, without repeating that
        # try/except/finally at every call site.
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
