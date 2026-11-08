"""
init_db.py — Database initializer and verifier for the Portfolio Analytics Agent.

Usage
-----
# First-time setup (create schema + load all CSV data):
    python init_db.py

# Force a clean reload (drops all tables and rebuilds from CSV):
    python init_db.py --force

# Verify an existing database without modifying it:
    python init_db.py --verify

This script can be run from any directory; all paths are resolved from its
own location using Path(__file__).

Exit codes
----------
0 — success (initialization completed, or verification passed)
1 — failure (initialization error, or verification found issues)
"""

import sys
import argparse
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable regardless
# of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.loader import initialize_database, verify_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize or verify the Portfolio Analytics SQLite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python init_db.py           # first-time setup\n"
            "  python init_db.py --force   # drop and reload all data\n"
            "  python init_db.py --verify  # check existing database\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--force",
        action="store_true",
        help=(
            "Drop and recreate all tables, then reload from CSV files. "
            "Requires explicit confirmation that existing data will be lost."
        ),
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Verify the existing database without modifying it. "
            "Checks table existence, row counts, PK uniqueness, "
            "FK integrity, and a sample JOIN. "
            "Exits with code 1 if any check fails."
        ),
    )

    args = parser.parse_args()

    if args.verify:
        # Verification path: read-only, returns bool, exits with matching code.
        passed = verify_database()
        sys.exit(0 if passed else 1)

    # Initialization path (default or --force).
    try:
        initialize_database(force=args.force)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] Missing file: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] Data mismatch: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
