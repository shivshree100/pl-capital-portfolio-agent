"""
Database loader for the Portfolio Analytics Agent.

Responsibilities:
- Read database_schema.sql and create all tables (idempotent via IF NOT EXISTS).
- Load each of the nine CSV files into their corresponding SQLite tables.
- Validate that the loaded row counts match the source CSV row counts.
- Refuse to run if the database already contains data, unless force=True.

Design notes:
- pandas.read_csv() + DataFrame.to_sql(if_exists='append') is used for all
  inserts.  The schema DDL is applied first so that SQLite's own type
  affinities, UNIQUE constraints, and indices are in place.
- We deliberately do NOT use if_exists='replace' because that drops and
  recreates the table without the DDL constraints.
- FK enforcement (PRAGMA foreign_keys) is disabled during schema creation
  and data loading, then re-enabled only for the final integrity check.
  This is necessary for two reasons:
    1. The provided database_schema.sql defines `securities` before `sectors`
       even though securities.sector_id references sectors.sector_id.  With
       FK ON, SQLite raises "foreign key mismatch" on the first INSERT into
       sectors because it can see an unresolved forward reference in the
       schema graph.
    2. pandas.to_sql() issues its own internal COMMIT, which conflicts with
       any SAVEPOINTs or manual BEGIN/COMMIT we might wrap around it.
  Keeping FK OFF during load and running PRAGMA foreign_key_check at the
  end gives us the same safety guarantee without the conflict.
- Data is loaded in FK dependency order (referenced tables before referencing
  tables) so that even if FK enforcement were on, inserts would succeed.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional

from src.database.connection import get_connection, get_db_path, get_project_root


# ---------------------------------------------------------------------------
# Load manifest — FK dependency order, with expected row counts
# ---------------------------------------------------------------------------

# Each entry: (csv_filename, table_name, expected_row_count)
LOAD_MANIFEST: list[tuple[str, str, int]] = [
    ("sectors.csv",               "sectors",               10),
    ("benchmarks.csv",            "benchmarks",            13),
    ("securities.csv",            "securities",            30),
    ("portfolios.csv",            "portfolios",            13),
    ("holdings.csv",              "holdings",              95),
    ("transactions.csv",          "transactions",          24),
    ("historical_prices.csv",     "historical_prices",    330),
    ("portfolio_performance.csv", "portfolio_performance", 44),
    ("risk_metrics.csv",          "risk_metrics",          44),
]

EXPECTED_TABLES: set[str] = {entry[1] for entry in LOAD_MANIFEST}

# Primary key column for each table — used in duplicate-key check.
PK_MAP: dict[str, str] = {
    "sectors":               "sector_id",
    "benchmarks":            "benchmark_id",
    "securities":            "security_id",
    "portfolios":            "portfolio_id",
    "holdings":              "holding_id",
    "transactions":          "transaction_id",
    "historical_prices":     "price_id",
    "portfolio_performance": "performance_id",
    "risk_metrics":          "risk_id",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _schema_path() -> Path:
    return get_project_root() / "database_schema.sql"


def _data_dir() -> Path:
    return get_project_root() / "data"


def _read_schema_sql() -> str:
    path = _schema_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {path}\n"
            "Ensure database_schema.sql is in the project root."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Low-level DB helpers
# ---------------------------------------------------------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    ).fetchone()
    return row[0] > 0


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM [{table}];").fetchone()[0]


def _database_has_data(conn: sqlite3.Connection) -> bool:
    for _, table, _ in LOAD_MANIFEST:
        if _table_exists(conn, table) and _row_count(conn, table) > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def create_schema(conn: sqlite3.Connection) -> None:
    """
    Run database_schema.sql DDL to create all tables and indices.

    Foreign key enforcement is turned OFF for the duration of DDL execution.
    The provided schema defines `securities` before `sectors`; with FK ON,
    SQLite rejects subsequent INSERTs into `sectors` with "foreign key
    mismatch".  Disabling FK during DDL (which carries no data risk) avoids
    this entirely.  It is re-enabled by the caller after loading completes.

    We avoid conn.executescript() because it issues an implicit COMMIT that
    puts the connection into autocommit mode, which breaks pandas.to_sql().
    Instead we split on ';' and execute each statement individually.
    """
    print("  Creating schema from database_schema.sql ...")
    ddl = _read_schema_sql()

    statements = [
        s.strip()
        for s in ddl.split(";")
        if s.strip() and not s.strip().startswith("--")
    ]

    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        for stmt in statements:
            conn.execute(stmt)
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Schema creation failed: {exc}") from exc

    print("  Schema created successfully.")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Read a CSV file and prepare it for SQLite insertion.

    - All columns are initially read as strings to prevent pandas from
      silently mis-casting ambiguous values (e.g. '99.75' as int).
    - Empty / whitespace-only strings become pd.NA (→ SQL NULL).
    - String columns are whitespace-stripped.
    - Each column is then coerced to a numeric type where unambiguous.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise RuntimeError(f"Failed to read '{csv_path.name}': {exc}") from exc

    # Replace blank cells with NA.
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Strip whitespace from string columns.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Coerce to numeric where possible (leaves strings and NAs untouched).
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only substitute if the conversion didn't introduce new NAs
        # in rows that had actual string data (i.e. non-null).
        non_null_mask = df[col].notna()
        if converted[non_null_mask].notna().all():
            df[col] = converted

    return df


def load_csv_to_table(
    conn: sqlite3.Connection,
    csv_path: Path,
    table_name: str,
    expected_count: int,
) -> int:
    """
    Load one CSV file into a SQLite table and verify the row count.

    FK enforcement must already be OFF on the connection when this is called
    (the caller — initialize_database — manages the PRAGMA lifecycle).

    pandas.to_sql() issues its own internal transaction/commit, so we do
    NOT wrap it in a manual BEGIN/COMMIT or SAVEPOINT; doing so causes
    "cannot commit - no transaction is active" or "no such savepoint" errors.

    Returns the number of rows now in the table.
    Raises RuntimeError on insert failure or ValueError on count mismatch.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Cannot populate table '{table_name}'."
        )

    df = _prepare_dataframe(csv_path)

    try:
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="append",  # schema already exists from DDL
            index=False,
            method="multi",      # batched INSERT — faster than row-by-row
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to insert rows into '{table_name}': {exc}\n"
            f"Source: {csv_path.name}"
        ) from exc

    loaded = _row_count(conn, table_name)
    if loaded != expected_count:
        raise ValueError(
            f"Row count mismatch for '{table_name}': "
            f"expected {expected_count}, got {loaded} "
            f"(CSV had {len(df)} rows)."
        )

    return loaded


# ---------------------------------------------------------------------------
# Force-drop helper
# ---------------------------------------------------------------------------

def _drop_all_tables(conn: sqlite3.Connection) -> None:
    """Drop all managed tables in reverse dependency order."""
    conn.execute("PRAGMA foreign_keys = OFF;")
    for _, table, _ in reversed(LOAD_MANIFEST):
        conn.execute(f"DROP TABLE IF EXISTS [{table}];")
    conn.commit()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def initialize_database(
    db_path: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Initialize the portfolio SQLite database end-to-end.

    Steps:
      1. Guard against overwriting existing data (unless force=True).
      2. Apply database_schema.sql DDL.
      3. Load all nine CSV files in FK dependency order.
      4. Re-enable FK enforcement.
      5. Run post-load validation (table existence, row counts,
         PK uniqueness, FK integrity, sample JOIN).

    Parameters
    ----------
    db_path : Path, optional
        Defaults to portfolio_database.db in the project root.
    force : bool
        Drop and recreate all tables before loading.  Required when the
        database already has data and you want a clean reload.
    """
    if db_path is None:
        db_path = get_db_path()

    print(f"\n{'='*60}")
    print(f"  Portfolio Analytics Agent — Database Initializer")
    print(f"{'='*60}")
    print(f"  Database : {db_path}")
    print(f"  Data dir : {_data_dir()}")
    print()

    conn = get_connection(db_path)

    try:
        # ------------------------------------------------------------------
        # Guard
        # ------------------------------------------------------------------
        if _database_has_data(conn):
            if not force:
                raise RuntimeError(
                    "Database already contains data.\n"
                    "Pass --force to drop and reload all tables."
                )
            print("  [force] Dropping existing tables for a clean reload ...")
            _drop_all_tables(conn)
            print()

        # ------------------------------------------------------------------
        # Step 1 — Schema  (FK OFF during DDL, see create_schema docstring)
        # ------------------------------------------------------------------
        create_schema(conn)
        # FK is still OFF here; kept that way through loading.
        print()

        # ------------------------------------------------------------------
        # Step 2 — Load CSVs
        # ------------------------------------------------------------------
        data_dir = _data_dir()
        results: list[tuple[str, int, int]] = []

        print("  Loading CSV files:")
        print(f"  {'Table':<25} {'Expected':>10} {'Loaded':>10}  Status")
        print(f"  {'-'*25} {'-'*10} {'-'*10}  {'-'*6}")

        for csv_name, table_name, expected in LOAD_MANIFEST:
            loaded = load_csv_to_table(
                conn, data_dir / csv_name, table_name, expected
            )
            status = "OK" if loaded == expected else "MISMATCH"
            print(f"  {table_name:<25} {expected:>10} {loaded:>10}  {status}")
            results.append((table_name, expected, loaded))

        print()

        # ------------------------------------------------------------------
        # Step 3 — Re-enable FK enforcement, then validate
        # ------------------------------------------------------------------
        conn.execute("PRAGMA foreign_keys = ON;")

        _run_validation(conn, results)

    except Exception:
        conn.close()
        raise

    conn.close()

    print(f"\n{'='*60}")
    print("  Database initialization complete.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Post-load validation
# ---------------------------------------------------------------------------

def _run_validation(
    conn: sqlite3.Connection,
    load_results: list[tuple[str, int, int]],
) -> None:
    """
    Five-point integrity check with a structured pass/fail report.

    1. All expected tables exist in sqlite_master.
    2. Row counts match expected CSV totals.
    3. No duplicate primary keys.
    4. No FK violations (PRAGMA foreign_key_check).
    5. Sample 4-table JOIN executes and returns rows.
    """
    print("  Running validation checks:")
    failures: list[str] = []

    # --- 1. Table existence ---
    existing = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    missing = EXPECTED_TABLES - existing
    if missing:
        msg = f"Missing tables: {sorted(missing)}"
        failures.append(msg)
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [PASS] All {len(EXPECTED_TABLES)} expected tables exist.")

    # --- 2. Row counts ---
    all_counts_ok = all(e == l for _, e, l in load_results)
    if all_counts_ok:
        print(f"  [PASS] All row counts match expected CSV totals.")
    else:
        for tbl, exp, loaded in load_results:
            if exp != loaded:
                msg = f"Row count mismatch: {tbl} expected {exp}, got {loaded}"
                failures.append(msg)
                print(f"  [FAIL] {msg}")

    # --- 3. No duplicate PKs ---
    dup_issues: list[str] = []
    for table, pk_col in PK_MAP.items():
        if table not in existing:
            continue
        total = _row_count(conn, table)
        distinct = conn.execute(
            f"SELECT COUNT(DISTINCT [{pk_col}]) FROM [{table}];"
        ).fetchone()[0]
        if total != distinct:
            dup_issues.append(
                f"{table}.{pk_col}: {total} rows, {distinct} distinct"
            )
    if dup_issues:
        for msg in dup_issues:
            failures.append(f"Duplicate PKs: {msg}")
            print(f"  [FAIL] Duplicate PKs: {msg}")
    else:
        print("  [PASS] No duplicate primary keys detected.")

    # --- 4. FK integrity ---
    # We run PRAGMA foreign_key_check per table rather than globally.
    # The `securities` table triggers "foreign key mismatch" from the PRAGMA
    # because database_schema.sql defines `securities` before `sectors`; we
    # substitute a manual SQL anti-join for that specific table.
    # Any other OperationalError is reported as an inconclusive check — it is
    # NOT silently ignored and does NOT count as a pass.
    fk_violations: list[str] = []
    fk_skipped: list[str] = []     # checks that could not be performed
    fk_checked: list[str] = []     # tables successfully verified

    for table in sorted(existing):
        if table not in PK_MAP:
            continue

        if table == "securities":
            # PRAGMA foreign_key_check cannot resolve the securities→sectors
            # FK because of schema definition order in database_schema.sql.
            # Use a manual anti-join instead.
            try:
                orphans = conn.execute(
                    """
                    SELECT COUNT(*) FROM securities
                    WHERE sector_id IS NOT NULL
                      AND sector_id NOT IN (SELECT sector_id FROM sectors);
                    """
                ).fetchone()[0]
                if orphans:
                    fk_violations.append(
                        f"securities: {orphans} row(s) with a sector_id "
                        f"that does not exist in sectors"
                    )
                fk_checked.append(table)
            except sqlite3.Error as exc:
                fk_skipped.append(
                    f"securities: manual FK query failed — {exc}"
                )
            continue

        try:
            rows = conn.execute(
                f"PRAGMA foreign_key_check([{table}]);"
            ).fetchall()
            for v in rows:
                fk_violations.append(
                    f"{v[0]}: rowid {v[1]} violates FK referencing '{v[2]}'"
                )
            fk_checked.append(table)
        except sqlite3.OperationalError as exc:
            # Do NOT silently pass.  Report the table and the exact error so
            # the caller knows this check was inconclusive, not clean.
            fk_skipped.append(f"{table}: PRAGMA foreign_key_check failed — {exc}")

    # Report violations
    for msg in fk_violations:
        failures.append(msg)
        print(f"  [FAIL] FK violation — {msg}")

    # Report skipped checks as warnings (not passes, not failures)
    for msg in fk_skipped:
        print(f"  [WARN] FK check skipped — {msg}")

    # Only claim a pass if every managed table was actually checked
    if not fk_violations and not fk_skipped:
        print(
            f"  [PASS] Foreign key integrity verified "
            f"({len(fk_checked)}/{len(PK_MAP)} tables checked)."
        )
    elif not fk_violations and fk_skipped:
        # Some checks could not run — mark incomplete rather than passed
        skipped_names = [s.split(":")[0] for s in fk_skipped]
        failures.append(
            f"FK check incomplete: could not verify {skipped_names}"
        )
        print(
            f"  [FAIL] FK check incomplete — "
            f"{len(fk_skipped)} table(s) could not be verified: "
            f"{skipped_names}"
        )

    # --- 5. Sample JOIN ---
    sample_sql = """
        SELECT
            p.portfolio_name,
            s.symbol,
            s.company_name,
            sec.sector_name,
            h.quantity,
            h.cost_basis
        FROM portfolios  p
        JOIN holdings    h   ON p.portfolio_id = h.portfolio_id
        JOIN securities  s   ON h.security_id  = s.security_id
        LEFT JOIN sectors sec ON s.sector_id   = sec.sector_id
        LIMIT 5;
    """
    try:
        rows = conn.execute(sample_sql).fetchall()
        if not rows:
            failures.append("Sample JOIN returned no rows.")
            print("  [FAIL] Sample JOIN returned no rows.")
        else:
            print(
                f"  [PASS] Sample 4-table JOIN returned {len(rows)} rows."
            )
            print()
            print(
                f"  {'Portfolio':<30} {'Symbol':<8} {'Company':<26} "
                f"{'Sector':<22} {'Qty':>6} {'Cost Basis':>12}"
            )
            print(
                f"  {'-'*30} {'-'*8} {'-'*26} {'-'*22} {'-'*6} {'-'*12}"
            )
            for r in rows:
                sector = r["sector_name"] or "N/A (Bond)"
                print(
                    f"  {r['portfolio_name']:<30} {r['symbol']:<8} "
                    f"{r['company_name']:<26} {sector:<22} "
                    f"{int(r['quantity']):>6} {r['cost_basis']:>12,.2f}"
                )
    except sqlite3.Error as exc:
        failures.append(f"Sample JOIN error: {exc}")
        print(f"  [FAIL] Sample JOIN error: {exc}")

    # --- Summary ---
    print()
    if failures:
        print(f"  VALIDATION RESULT: FAILED  ({len(failures)} issue(s))")
        for f in failures:
            print(f"    - {f}")
        raise RuntimeError(
            f"Database validation failed with {len(failures)} issue(s). "
            "See output above."
        )

    total_rows = sum(exp for _, exp, _ in load_results)
    print(
        f"  VALIDATION RESULT: PASSED — "
        f"{len(LOAD_MANIFEST)} tables, {total_rows} total rows, "
        f"all integrity checks clean."
    )


# ---------------------------------------------------------------------------
# Standalone verification entry point
# ---------------------------------------------------------------------------

def verify_database(db_path: Optional[Path] = None) -> bool:
    """
    Verify an existing database without modifying it.

    Performs the same five-point integrity check as the post-load validation
    inside initialize_database(), but operates independently: it reads the
    expected row counts from LOAD_MANIFEST (the same source-of-truth used
    during loading) rather than from a fresh load cycle.

    This makes it safe to run at any time against a live database — it issues
    only SELECT statements and PRAGMAs.

    Parameters
    ----------
    db_path : Path, optional
        Defaults to portfolio_database.db in the project root.

    Returns
    -------
    bool
        True if all checks pass, False if any check fails or is inconclusive.
        Never raises; all errors are caught and reported.
    """
    if db_path is None:
        db_path = get_db_path()

    print(f"\n{'='*60}")
    print(f"  Portfolio Analytics Agent — Database Verification")
    print(f"{'='*60}")
    print(f"  Database : {db_path}")
    print()

    if not db_path.exists():
        print(f"  [FAIL] Database file does not exist: {db_path}")
        print(f"\n  VERIFICATION RESULT: FAILED")
        print(f"  Run 'python init_db.py' to create and populate the database.")
        print(f"{'='*60}\n")
        return False

    try:
        conn = get_connection(db_path)
    except RuntimeError as exc:
        print(f"  [FAIL] Cannot open database: {exc}")
        print(f"\n  VERIFICATION RESULT: FAILED")
        print(f"{'='*60}\n")
        return False

    # Build the load_results list from LOAD_MANIFEST + actual DB counts.
    # This mirrors what _run_validation receives after a fresh load.
    failures: list[str] = []

    try:
        # ------------------------------------------------------------------
        # Check 1: Table existence
        # ------------------------------------------------------------------
        print("  Running verification checks:")

        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        missing = EXPECTED_TABLES - existing
        if missing:
            msg = f"Missing tables: {sorted(missing)}"
            failures.append(msg)
            print(f"  [FAIL] {msg}")
        else:
            print(f"  [PASS] All {len(EXPECTED_TABLES)} expected tables exist.")

        # ------------------------------------------------------------------
        # Check 2: Row counts vs CSV expectations
        # ------------------------------------------------------------------
        print()
        print(
            f"  {'Table':<25} {'Expected':>10} {'Actual':>10}  Status"
        )
        print(
            f"  {'-'*25} {'-'*10} {'-'*10}  {'-'*10}"
        )
        load_results: list[tuple[str, int, int]] = []
        for _, table_name, expected in LOAD_MANIFEST:
            if table_name not in existing:
                actual = 0
                status = "MISSING"
                failures.append(
                    f"Table '{table_name}' does not exist (expected {expected} rows)."
                )
            else:
                actual = _row_count(conn, table_name)
                if actual == expected:
                    status = "OK"
                elif actual == 0:
                    status = "EMPTY"
                    failures.append(
                        f"'{table_name}' has 0 rows (expected {expected})."
                    )
                elif actual > expected:
                    status = f"EXCESS +{actual - expected}"
                    failures.append(
                        f"'{table_name}' has {actual} rows "
                        f"(expected {expected}, {actual - expected} unexpected extra)."
                    )
                else:
                    status = f"SHORT -{expected - actual}"
                    failures.append(
                        f"'{table_name}' has {actual} rows "
                        f"(expected {expected}, {expected - actual} missing)."
                    )
            print(
                f"  {table_name:<25} {expected:>10} {actual:>10}  {status}"
            )
            load_results.append((table_name, expected, actual))

        count_ok = all(e == a for _, e, a in load_results)
        print()
        if count_ok:
            print("  [PASS] All row counts match expected CSV totals.")
        else:
            print("  [FAIL] One or more row counts do not match expected totals.")

        # ------------------------------------------------------------------
        # Check 3: No duplicate primary keys
        # ------------------------------------------------------------------
        dup_issues: list[str] = []
        for table, pk_col in PK_MAP.items():
            if table not in existing:
                continue
            total = _row_count(conn, table)
            distinct = conn.execute(
                f"SELECT COUNT(DISTINCT [{pk_col}]) FROM [{table}];"
            ).fetchone()[0]
            if total != distinct:
                dup_issues.append(
                    f"{table}.{pk_col}: {total} rows, {distinct} distinct"
                )
        if dup_issues:
            for msg in dup_issues:
                failures.append(f"Duplicate PKs: {msg}")
                print(f"  [FAIL] Duplicate PKs: {msg}")
        else:
            print("  [PASS] No duplicate primary keys detected.")

        # ------------------------------------------------------------------
        # Check 4: FK integrity (same improved logic as _run_validation)
        # ------------------------------------------------------------------
        fk_violations: list[str] = []
        fk_skipped: list[str] = []
        fk_checked: list[str] = []

        for table in sorted(existing):
            if table not in PK_MAP:
                continue

            if table == "securities":
                try:
                    orphans = conn.execute(
                        """
                        SELECT COUNT(*) FROM securities
                        WHERE sector_id IS NOT NULL
                          AND sector_id NOT IN (SELECT sector_id FROM sectors);
                        """
                    ).fetchone()[0]
                    if orphans:
                        fk_violations.append(
                            f"securities: {orphans} row(s) with a sector_id "
                            f"that does not exist in sectors"
                        )
                    fk_checked.append(table)
                except sqlite3.Error as exc:
                    fk_skipped.append(
                        f"securities: manual FK query failed — {exc}"
                    )
                continue

            try:
                rows = conn.execute(
                    f"PRAGMA foreign_key_check([{table}]);"
                ).fetchall()
                for v in rows:
                    fk_violations.append(
                        f"{v[0]}: rowid {v[1]} violates FK referencing '{v[2]}'"
                    )
                fk_checked.append(table)
            except sqlite3.OperationalError as exc:
                fk_skipped.append(
                    f"{table}: PRAGMA foreign_key_check failed — {exc}"
                )

        for msg in fk_violations:
            failures.append(msg)
            print(f"  [FAIL] FK violation — {msg}")

        for msg in fk_skipped:
            print(f"  [WARN] FK check skipped — {msg}")

        if not fk_violations and not fk_skipped:
            print(
                f"  [PASS] Foreign key integrity verified "
                f"({len(fk_checked)}/{len(PK_MAP)} tables checked)."
            )
        elif not fk_violations and fk_skipped:
            skipped_names = [s.split(":")[0] for s in fk_skipped]
            failures.append(
                f"FK check incomplete: could not verify {skipped_names}"
            )
            print(
                f"  [FAIL] FK check incomplete — "
                f"{len(fk_skipped)} table(s) could not be verified: "
                f"{skipped_names}"
            )

        # ------------------------------------------------------------------
        # Check 5: Sample JOIN
        # ------------------------------------------------------------------
        sample_sql = """
            SELECT
                p.portfolio_name,
                s.symbol,
                s.company_name,
                sec.sector_name,
                h.quantity,
                h.cost_basis
            FROM portfolios  p
            JOIN holdings    h   ON p.portfolio_id = h.portfolio_id
            JOIN securities  s   ON h.security_id  = s.security_id
            LEFT JOIN sectors sec ON s.sector_id   = sec.sector_id
            LIMIT 5;
        """
        try:
            rows = conn.execute(sample_sql).fetchall()
            if not rows:
                failures.append("Sample JOIN returned no rows.")
                print("  [FAIL] Sample JOIN returned no rows.")
            else:
                print(
                    f"  [PASS] Sample 4-table JOIN returned {len(rows)} rows."
                )
                print()
                print(
                    f"  {'Portfolio':<30} {'Symbol':<8} {'Company':<26} "
                    f"{'Sector':<22} {'Qty':>6} {'Cost Basis':>12}"
                )
                print(
                    f"  {'-'*30} {'-'*8} {'-'*26} {'-'*22} {'-'*6} {'-'*12}"
                )
                for r in rows:
                    sector = r["sector_name"] or "N/A (Bond)"
                    print(
                        f"  {r['portfolio_name']:<30} {r['symbol']:<8} "
                        f"{r['company_name']:<26} {sector:<22} "
                        f"{int(r['quantity']):>6} {r['cost_basis']:>12,.2f}"
                    )
        except sqlite3.Error as exc:
            failures.append(f"Sample JOIN error: {exc}")
            print(f"  [FAIL] Sample JOIN error: {exc}")

    finally:
        conn.close()

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print()
    if failures:
        print(f"  VERIFICATION RESULT: FAILED  ({len(failures)} issue(s))")
        for issue in failures:
            print(f"    - {issue}")
        print(f"{'='*60}\n")
        return False

    total_expected = sum(exp for _, exp, _ in load_results)
    print(
        f"  VERIFICATION RESULT: PASSED — "
        f"{len(LOAD_MANIFEST)} tables, {total_expected} total rows, "
        f"all integrity checks clean."
    )
    print(f"{'='*60}\n")
    return True
