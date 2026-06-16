import sqlite3
from pathlib import Path

from app.config import settings


def get_sqlite_path() -> Path:
    """
    Returns the configured SQLite DB path and ensures parent directory exists.
    """
    db_path = Path(settings.SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection.

    check_same_thread=False allows usage from FastAPI request handlers.
    For this MVP, this is fine.
    """
    conn = sqlite3.connect(
        get_sqlite_path(),
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn