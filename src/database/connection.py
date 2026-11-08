"""
Database connection factory for the Portfolio Analytics Agent.

Responsibilities:
- Resolve the database file path relative to the project root.
- Return configured sqlite3 connections with foreign key enforcement.
- Provide a context-manager helper for safe, auto-closing connections.

Design notes:
- We use a persistent file-based SQLite database (portfolio_database.db)
  so that the loader step only needs to run once and the file can be
  inspected externally with any SQLite browser.
- We intentionally do NOT use a module-level singleton connection.
  SQLite connections are not thread-safe by default, so callers should
  obtain a connection when they need one and close it when they are done.
  The get_connection() factory makes this easy.
- Foreign key enforcement is OFF by default in SQLite; we enable it on
  every new connection via PRAGMA foreign_keys = ON.
"""

import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    This file lives at <root>/src/database/connection.py, so the root is
    two levels up.  Using Path(__file__) makes it reliable regardless of
    the working directory from which the process is launched.
    """
    return Path(__file__).resolve().parent.parent.parent


def get_db_path() -> Path:
    """Return the absolute path to the SQLite database file."""
    return get_project_root() / "portfolio_database.db"


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """
    Open and return a new sqlite3 connection.

    Parameters
    ----------
    db_path:
        Path to the database file.  Defaults to portfolio_database.db in
        the project root.  Pass ':memory:' (as a string wrapped in Path)
        or an explicit path during testing.

    Returns
    -------
    sqlite3.Connection
        A configured connection with:
        - row_factory set to sqlite3.Row so columns are accessible by name.
        - foreign_keys PRAGMA enabled.
        - journal_mode set to WAL.
        - A 5-second busy timeout to handle concurrent readers gracefully.

    Raises
    ------
    RuntimeError
        If the connection cannot be established.
    """
    if db_path is None:
        db_path = get_db_path()

    resolved = str(db_path)

    try:
        conn = sqlite3.connect(resolved, timeout=5.0)
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Failed to open database at '{resolved}': {exc}"
        ) from exc

    # Return rows as dict-like objects (access columns by name).
    conn.row_factory = sqlite3.Row

    # Enable foreign key constraint enforcement.  Must be done per connection.
    # PRAGMA statements do not open a transaction, so no commit is needed here.
    conn.execute("PRAGMA foreign_keys = ON;")

    # Write-ahead logging gives better concurrent read performance.
    conn.execute("PRAGMA journal_mode = WAL;")

    return conn


# ---------------------------------------------------------------------------
# Context manager for safe usage
# ---------------------------------------------------------------------------

@contextmanager
def managed_connection(db_path: Path | None = None):
    """
    Context manager that opens a connection and guarantees it is closed.

    Usage
    -----
    with managed_connection() as conn:
        rows = conn.execute("SELECT * FROM portfolios").fetchall()

    The connection is committed and closed on normal exit; rolled back and
    closed on exception.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Quick connectivity check
# ---------------------------------------------------------------------------

def check_connection(db_path: Path | None = None) -> bool:
    """
    Return True if a connection can be opened and a trivial query executed.

    Useful for startup health checks without raising an exception.
    """
    try:
        with managed_connection(db_path) as conn:
            conn.execute("SELECT 1;")
        return True
    except Exception:
        return False
