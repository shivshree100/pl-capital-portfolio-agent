# Database Setup

## Overview

The Portfolio Analytics Agent uses a local SQLite database (`portfolio_database.db`)
that is generated once from the nine source CSV files in `data/`.  The database file
is not committed to version control (it is in `.gitignore`); it must be created by
running the initializer script.

---

## How the Database Is Initialized

Run the following command from the **project root**:

```bash
python init_db.py
```

This single command:

1. Reads `database_schema.sql` and creates all eight tables plus their indices.
2. Loads each CSV in dependency order (referenced tables before referencing tables).
3. Validates row counts, primary key uniqueness, foreign key integrity, and a
   four-table JOIN.
4. Prints a structured pass/fail report.

To force a **clean reload** (drop all tables and reimport):

```bash
python init_db.py --force
```

To **verify** an existing database without modifying it:

```bash
python init_db.py --verify
```

Exit codes: `0` = success / verification passed, `1` = error / verification failed.
`--force` and `--verify` are mutually exclusive.

---

## Table Load Order

Tables are loaded in foreign-key dependency order so constraint checks are
satisfied at insert time:

| Order | CSV file                  | Table                 | Rows |
|------:|---------------------------|-----------------------|-----:|
|     1 | sectors.csv               | sectors               |   10 |
|     2 | benchmarks.csv            | benchmarks            |   13 |
|     3 | securities.csv            | securities            |   30 |
|     4 | portfolios.csv            | portfolios            |   13 |
|     5 | holdings.csv              | holdings              |   95 |
|     6 | transactions.csv          | transactions          |   24 |
|     7 | historical_prices.csv     | historical_prices     |  330 |
|     8 | portfolio_performance.csv | portfolio_performance |   44 |
|     9 | risk_metrics.csv          | risk_metrics          |   44 |

**Total: 607 rows across 9 tables.**

---

## How CSV Files Are Loaded

```
src/database/loader.py
```

- All columns are initially read as strings by `pandas.read_csv()`.
- Empty/whitespace values are replaced with `NaN` (→ SQL `NULL`).
- Each column is then coerced to a numeric type where possible; string
  columns (names, dates, enums) remain as text.
- Rows are inserted with `DataFrame.to_sql(if_exists='append')` so the
  schema created by the DDL file is preserved (types, constraints, indices).
- Each table load is wrapped in an explicit `BEGIN` / `COMMIT` / `ROLLBACK`
  so a failure in one table does not corrupt others.

---

## How Connections Are Managed

```
src/database/connection.py
```

Two public interfaces are provided:

### `get_connection(db_path=None) -> sqlite3.Connection`

Opens and returns a new connection.  Callers are responsible for closing it.
Suitable for long-lived use inside a class or service.

```python
from src.database.connection import get_connection

conn = get_connection()
rows = conn.execute("SELECT * FROM portfolios").fetchall()
conn.close()
```

### `managed_connection(db_path=None)` (context manager)

Opens a connection, commits on success, rolls back on exception, and always
closes.  Preferred for short operations.

```python
from src.database.connection import managed_connection

with managed_connection() as conn:
    rows = conn.execute("SELECT COUNT(*) FROM holdings").fetchall()
```

Every connection is configured with:

| Setting               | Value            | Reason                                      |
|-----------------------|------------------|---------------------------------------------|
| `row_factory`         | `sqlite3.Row`    | Columns accessible by name, not just index  |
| `foreign_keys`        | `ON`             | Enforce FK constraints at runtime           |
| `journal_mode`        | `WAL`            | Better concurrent read performance          |
| `timeout`             | 5 seconds        | Graceful handling of busy database          |

---

## How Validation Is Performed

After loading, `loader.py` runs five checks automatically:

| # | Check                          | What it verifies                                      |
|---|--------------------------------|-------------------------------------------------------|
| 1 | Table existence                | All 9 expected tables are present in `sqlite_master`  |
| 2 | Row counts                     | Each table's count matches the source CSV row count; reports OK / EMPTY / EXCESS +N / SHORT -N |
| 3 | Primary key uniqueness         | `COUNT(*) == COUNT(DISTINCT pk)` for every table      |
| 4 | Foreign key integrity          | `PRAGMA foreign_key_check` per table; `securities→sectors` uses a manual anti-join; any `OperationalError` is reported as `[WARN]` and causes the overall FK check to fail rather than pass silently |
| 5 | Sample 4-table JOIN            | `portfolios → holdings → securities → sectors` works  |

A validation failure raises a `RuntimeError` with a detailed message and
exits with code 1 so CI pipelines can detect broken imports.

---

## File Locations

```
portfolio_database.db      ← generated, gitignored
init_db.py                 ← CLI entry point
src/
  database/
    connection.py          ← connection factory + context manager
    loader.py              ← schema creation + CSV loading + validation
database_schema.sql        ← original DDL (do not modify)
data/                      ← original CSV files (do not modify)
```
