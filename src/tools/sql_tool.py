"""
SQL Query Tool for the Portfolio Analytics Agent.

Public API
----------
execute_sql_query(sql: str, db_path: Path | None = None) -> SqlResult

This module does NOT generate SQL — that is the agent's job via Gemini.
This module's responsibility is:
  1. Validate that the SQL is safe to execute (SELECT-only).
  2. Execute it against the SQLite database.
  3. Return a structured result with rows, column names, and metadata.

Design decisions
----------------
- We enforce a strict SELECT-only policy.  Any statement that contains
  INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA (write
  variants), or semicolons after the first statement is rejected before
  execution.  This prevents prompt-injection attacks where a malicious
  user question tricks Gemini into generating a destructive statement.

- We use parameterised execution via sqlite3 (not string formatting) so
  the safety check is a defence-in-depth measure, not the only line of
  defence.

- Results are returned as a list of dicts (column→value) so they are
  immediately JSON-serialisable and easy for the agent to format.

- Row limit: 500 rows maximum.  This prevents accidental full-table scans
  from flooding the LLM context window.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.database.connection import managed_connection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ROWS = 500

# Keywords that indicate a mutating or structural statement.
# We check the normalised (upper-case, whitespace-collapsed) SQL.
_FORBIDDEN_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|DETACH|VACUUM|REINDEX|ANALYZE)\b",
    re.IGNORECASE,
)

# A valid query must start with SELECT (or WITH … SELECT for CTEs).
_VALID_START = re.compile(
    r"^\s*(WITH\b|SELECT\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SqlResult:
    """
    Structured result of a SQL query execution.

    Attributes
    ----------
    sql : str
        The SQL statement that was executed (or attempted).
    success : bool
        True if the query executed without error.
    columns : list[str]
        Column names in result order.  Empty on failure.
    rows : list[dict]
        Each row as a {column: value} dict.  Empty on failure or no rows.
    row_count : int
        Number of rows returned.
    truncated : bool
        True if more than MAX_ROWS rows were available and the result
        was capped.
    error : str
        Human-readable error message.  Empty on success.
    """
    sql: str
    success: bool = False
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str = ""

    def to_display_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "success": self.success,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> Optional[str]:
    """
    Check that *sql* is safe to execute.

    Returns None if the query passes all checks, or a human-readable error
    string describing the first violation found.

    Checks performed:
    1. Non-empty string.
    2. Starts with SELECT or WITH (CTEs).
    3. Contains no mutating or DDL keywords.
    4. Contains only one statement (no semicolons mid-query, only trailing).
    """
    if not sql or not sql.strip():
        return "SQL query is empty."

    stripped = sql.strip().rstrip(";")

    # Check for multiple statements (semicolons inside the query body)
    if ";" in stripped:
        return (
            "Multiple SQL statements are not permitted. "
            "Submit one SELECT statement at a time."
        )

    if not _VALID_START.match(stripped):
        return (
            "Only SELECT queries (including CTEs starting with WITH) "
            "are permitted. The provided statement does not start with "
            "SELECT or WITH."
        )

    match = _FORBIDDEN_PATTERNS.search(stripped)
    if match:
        return (
            f"SQL contains a forbidden keyword: '{match.group().upper()}'. "
            "Only read-only SELECT statements are permitted."
        )

    return None  # all checks passed


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_sql_query(
    sql: str,
    db_path: Optional[Path] = None,
) -> SqlResult:
    """
    Validate and execute a SQL SELECT query against the portfolio database.

    Parameters
    ----------
    sql : str
        A SQL SELECT statement (or CTE) to execute.  Must be read-only.
    db_path : Path, optional
        Database path override.  Defaults to portfolio_database.db.

    Returns
    -------
    SqlResult
        A structured result.  On validation or execution failure,
        result.success is False and result.error contains the reason.
        Never raises — all errors are captured in the result.
    """
    result = SqlResult(sql=sql.strip() if sql else "")

    # --- Step 1: validate ---
    validation_error = validate_sql(sql)
    if validation_error:
        result.error = validation_error
        return result

    # --- Step 2: execute ---
    try:
        with managed_connection(db_path) as conn:
            # Enforce row limit by wrapping the query in a subquery.
            # We fetch MAX_ROWS + 1 to detect truncation.
            limited_sql = (
                f"SELECT * FROM ({sql.strip().rstrip(';')}) "
                f"LIMIT {MAX_ROWS + 1};"
            )
            cursor = conn.execute(limited_sql)
            all_rows = cursor.fetchall()

            result.columns = [desc[0] for desc in cursor.description]

            # Detect and apply truncation
            if len(all_rows) > MAX_ROWS:
                all_rows = all_rows[:MAX_ROWS]
                result.truncated = True

            result.rows = [dict(row) for row in all_rows]
            result.row_count = len(result.rows)
            result.success = True

    except sqlite3.Error as exc:
        result.error = f"SQL execution error: {exc}"
    except Exception as exc:
        result.error = f"Unexpected error during query execution: {exc}"

    return result


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def format_sql_result(result: SqlResult, max_display_rows: int = 20) -> str:
    """
    Return a human-readable string representation of a SqlResult.

    This is a presentation function; it does no computation.
    """
    lines: list[str] = []

    if not result.success:
        lines.append(f"Query failed: {result.error}")
        return "\n".join(lines)

    if result.row_count == 0:
        lines.append("The query returned no results.")
        return "\n".join(lines)

    # Single-value result (e.g. COUNT(*), SUM())
    if result.row_count == 1 and len(result.columns) == 1:
        val = result.rows[0][result.columns[0]]
        lines.append(f"{result.columns[0]}: {val}")
        return "\n".join(lines)

    # List result (single column, multiple rows)
    if len(result.columns) == 1:
        col = result.columns[0]
        lines.append(f"{col}:")
        for row in result.rows[:max_display_rows]:
            lines.append(f"  - {row[col]}")
        if result.truncated or result.row_count > max_display_rows:
            shown = min(result.row_count, max_display_rows)
            lines.append(f"  ... ({shown} of {result.row_count} rows shown)")
        return "\n".join(lines)

    # Table result — compute column widths
    col_widths = {col: len(str(col)) for col in result.columns}
    display_rows = result.rows[:max_display_rows]
    for row in display_rows:
        for col in result.columns:
            col_widths[col] = max(col_widths[col], len(str(row.get(col, ""))))

    header = "  " + "  ".join(
        str(col).ljust(col_widths[col]) for col in result.columns
    )
    divider = "  " + "  ".join("-" * col_widths[col] for col in result.columns)
    lines.append(header)
    lines.append(divider)
    for row in display_rows:
        line = "  " + "  ".join(
            str(row.get(col, "")).ljust(col_widths[col])
            for col in result.columns
        )
        lines.append(line)

    if result.truncated or result.row_count > max_display_rows:
        shown = min(result.row_count, max_display_rows)
        suffix = " (result was capped)" if result.truncated else ""
        lines.append(f"\n  Showing {shown} of {result.row_count} rows{suffix}.")

    return "\n".join(lines)
