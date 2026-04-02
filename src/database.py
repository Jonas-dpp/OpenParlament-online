"""
Database engine and session management for OpenParlament.

Usage
─────
    from src.database import get_session, init_db

    init_db()  # creates all tables (idempotent)

    with get_session() as session:
        session.add(some_object)
        # auto-commit on exit, rollback on exception
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.models import Base

# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "openparlament.db"


def _get_db_url() -> str:
    """Return the database URL.

    Reads ``OPENPARLAMENT_DB_URL`` from the environment if set, otherwise falls
    back to the default SQLite file path inside the ``data/`` directory.
    """
    env_url = os.environ.get("OPENPARLAMENT_DB_URL")
    if env_url:
        return env_url
    _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_DB_PATH}"


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory
# ─────────────────────────────────────────────────────────────────────────────

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return (and lazily create) the singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        db_url = _get_db_url()
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(db_url, connect_args=connect_args, echo=False)
        # Enable WAL mode and foreign-key enforcement for SQLite.
        if db_url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, _connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────────────────────────────────────────

_SessionFactory: sessionmaker | None = None


def _get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autoflush=True, autocommit=False)
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session.

    Commits automatically on successful exit; rolls back on any exception.

    Example::

        with get_session() as session:
            session.add(obj)
    """
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all database tables (idempotent – safe to call multiple times)."""
    Base.metadata.create_all(bind=get_engine())


def drop_db() -> None:
    """Drop all tables (DANGER: destroys all data – for testing only)."""
    Base.metadata.drop_all(bind=get_engine())


def reset_db_state() -> None:
    """Reset the cached engine and session factory (for testing only).

    Call this after changing ``OPENPARLAMENT_DB_URL`` (e.g. via
    ``monkeypatch.setenv``) so that the next call to :func:`get_engine`
    picks up the new URL instead of reusing the previously cached engine.
    """
    global _engine, _SessionFactory
    _engine = None
    _SessionFactory = None
