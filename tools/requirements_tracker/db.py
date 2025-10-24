from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "requirements.db"


def _build_database_url() -> str:
    """Derive the SQLite URL, allowing overrides for testing."""
    override = os.getenv("CALL_ANALYTICS_REQUIREMENTS_DB")
    if override:
        return override
    return f"sqlite:///{DEFAULT_DB_PATH}"


def _create_engine():
    return create_engine(_build_database_url(), connect_args={"check_same_thread": False})


engine = _create_engine()


def reset_engine() -> None:
    """Rebuild the SQLAlchemy engine (handy for tests with temp DBs)."""
    global engine  # noqa: PLW0603
    engine = _create_engine()


def init_db() -> None:
    """Create database tables if they do not exist."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency providing a database session."""
    with Session(engine) as session:
        yield session
