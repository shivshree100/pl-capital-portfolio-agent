"""
Tests for src/tools/exposure_tool.py

Test strategy
-------------
- Ground-truth tests (Q9, Q10): validate the calculator against manually
  derived expected values, NOT against hardcoded answers from the JSON.
  The expected values are recomputed from the same formula the tool uses
  (quantity × current_price), so the tests exercise the full pipeline.

- Edge-case tests: use an in-memory SQLite database with controlled data
  so they run without the production database and are fully deterministic.

Running the tests
-----------------
    # From the project root:
    python -m pytest tests/test_exposure_tool.py -v

    # Or run this file directly:
    python tests/test_exposure_tool.py
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup — make `src` importable when pytest is run from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.exposure_tool import calculate_sector_exposure, ExposureResult


# ---------------------------------------------------------------------------
# Helpers for in-memory test databases
# ---------------------------------------------------------------------------

def _build_test_db(scenario: str) -> Path:
    """
    Return the string ':memory:' sentinel used by the in-memory fixture.

    We pass a custom db_path to calculate_sector_exposure for each test.
    Because managed_connection(db_path) resolves str paths directly,
    we use the string ':memory:' wrapped as a Path-like string object
    by patching the connection via a fixture-populated in-memory DB.

    Instead of monkey-patching, each edge-case test creates a real
    temporary SQLite file, populates it with controlled data, runs the
    tool, and then cleans up.  This avoids mocking complexity while
    keeping the tests isolated from the production database.
    """
    raise NotImplementedError("Use _InMemoryDB fixture instead.")


class _InMemoryDB:
    """
    Minimal in-memory SQLite database for edge-case tests.

    Creates the four tables the exposure tool needs:
    portfolios, holdings, securities, sectors.
    """

    DDL = """
        CREATE TABLE sectors (
            sector_id   INTEGER PRIMARY KEY,
            sector_name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE portfolios (
            portfolio_id   INTEGER PRIMARY KEY,
            portfolio_name TEXT NOT NULL UNIQUE,
            status         TEXT
        );
        CREATE TABLE securities (
            security_id   INTEGER PRIMARY KEY,
            symbol        TEXT NOT NULL UNIQUE,
            company_name  TEXT NOT NULL,
            asset_type    TEXT NOT NULL,
            sector_id     INTEGER,
            current_price REAL,
            FOREIGN KEY (sector_id) REFERENCES sectors(sector_id)
        );
        CREATE TABLE holdings (
            holding_id   INTEGER PRIMARY KEY,
            portfolio_id INTEGER NOT NULL,
            security_id  INTEGER NOT NULL,
            quantity     REAL NOT NULL,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
            FOREIGN KEY (security_id)  REFERENCES securities(security_id),
            UNIQUE(portfolio_id, security_id)
        );
    """

    def __init__(self, tmp_path: Path, scenario_name: str):
        self.db_path = tmp_path / f"{scenario_name}.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(self.DDL)
        conn.commit()
        self._conn = conn  # keep alive for inserts

    def execute(self, sql: str, params=()):
        self._conn.execute(sql, params)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prod_db_path() -> Path:
    """Path to the production database — must already exist."""
    p = PROJECT_ROOT / "portfolio_database.db"
    if not p.exists():
        pytest.skip(
            "Production database not found. Run 'python init_db.py' first."
        )
    return p


# ---------------------------------------------------------------------------
# Ground-truth tests — Q9 and Q10
# ---------------------------------------------------------------------------

class TestGroundTruth:
    """
    Tests that reproduce the manual calculations verified before implementation.

    Expected values were derived by hand:
      market_value = quantity × securities.current_price (from the CSV/DB)
    They are NOT copied from ground_truth_dataset.json; they are independent
    arithmetic results using the same source data.
    """

    # -----------------------------------------------------------------------
    # Q9 — Tech Innovation Fund (portfolio_id = 3)
    #
    # Holdings and prices (from securities.current_price):
    #   AAPL  qty=1800  price=185.50  mv=333,900.00   sector=Technology
    #   MSFT  qty=600   price=415.26  mv=249,156.00   sector=Technology
    #   GOOGL qty=80    price=2750.80 mv=220,064.00   sector=Technology
    #   NVDA  qty=300   price=875.30  mv=262,590.00   sector=Technology
    #   AVGO  qty=150   price=875.15  mv=131,272.50   sector=Technology
    #   CRM   qty=800   price=210.90  mv=168,720.00   sector=Technology
    #   ORCL  qty=600   price=115.85  mv= 69,510.00   sector=Technology
    #   META  qty=400   price=325.20  mv=130,080.00   sector=Technology
    #
    # Total equity MV = 1,565,292.50
    # Technology = 100.00% (all holdings are Technology)
    # -----------------------------------------------------------------------

    TECH_FUND_NAME = "Tech Innovation Fund"

    # Tolerances: percentages rounded to 4 dp inside the tool.
    # We check to 2 dp in assertions to allow for future price updates.
    PCT_TOLERANCE = 0.01   # ± 0.01 percentage points
    MV_TOLERANCE  = 0.01   # ± $0.01

    def test_tech_fund_found(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        assert result.portfolio_found, (
            f"Portfolio '{self.TECH_FUND_NAME}' should be found in the DB."
        )

    def test_tech_fund_no_warnings(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        assert result.warnings == [], (
            f"Expected no data-quality warnings, got: {result.warnings}"
        )

    def test_tech_fund_equity_count(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        # 8 equity holdings, 0 bonds
        assert result.equity_holding_count == 8, (
            f"Expected 8 equity holdings, got {result.equity_holding_count}"
        )
        assert result.excluded_bond_count == 0, (
            f"Expected 0 bond holdings, got {result.excluded_bond_count}"
        )

    def test_tech_fund_total_market_value(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        expected_total = 1_565_292.50
        assert abs(result.total_equity_market_value - expected_total) < self.MV_TOLERANCE, (
            f"Total equity MV: expected ~{expected_total:,.2f}, "
            f"got {result.total_equity_market_value:,.4f}"
        )

    def test_tech_fund_single_sector(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        assert "Technology" in result.sector_exposures, (
            "Technology sector should be present in the breakdown."
        )
        assert len(result.sector_exposures) == 1, (
            f"Expected exactly 1 sector, got: {list(result.sector_exposures.keys())}"
        )

    def test_tech_fund_technology_100pct(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        tech_pct = result.sector_exposures.get("Technology", 0.0)
        assert abs(tech_pct - 100.0) < self.PCT_TOLERANCE, (
            f"Technology should be ~100.00%, got {tech_pct:.4f}%"
        )

    def test_tech_fund_market_values_per_sector(self, prod_db_path):
        result = calculate_sector_exposure(
            self.TECH_FUND_NAME, db_path=prod_db_path
        )
        expected_tech_mv = 1_565_292.50
        actual_tech_mv = result.sector_market_values.get("Technology", 0.0)
        assert abs(actual_tech_mv - expected_tech_mv) < self.MV_TOLERANCE, (
            f"Technology market value: expected ~{expected_tech_mv:,.2f}, "
            f"got {actual_tech_mv:,.4f}"
        )

    # -----------------------------------------------------------------------
    # Q10 — International Equity Fund (portfolio_id = 7)
    #
    # Holdings and prices:
    #   AAPL  qty=600   price=185.50  mv=111,300.00   sector=Technology
    #   MSFT  qty=200   price=415.26  mv= 83,052.00   sector=Technology
    #   GOOGL qty=30    price=2750.80 mv= 82,524.00   sector=Technology
    #   AVGO  qty=80    price=875.15  mv= 70,012.00   sector=Technology
    #   TSLA  qty=400   price=248.50  mv= 99,400.00   sector=Automotive
    #   XOM   qty=1000  price=98.75   mv= 98,750.00   sector=Energy
    #   CVX   qty=800   price=158.40  mv=126,720.00   sector=Energy
    #   HD    qty=200   price=315.85  mv= 63,170.00   sector=Consumer Staples
    #
    # Sector totals:
    #   Technology      = 111,300 + 83,052 + 82,524 + 70,012 = 346,888.00
    #   Energy          = 98,750 + 126,720                   = 225,470.00
    #   Automotive      = 99,400.00
    #   Consumer Staples= 63,170.00
    #
    # Total equity MV  = 734,928.00
    # Exposures:
    #   Technology       = 346,888 / 734,928 × 100 ≈ 47.1999%
    #   Energy           = 225,470 / 734,928 × 100 ≈ 30.6793%
    #   Automotive       = 99,400 / 734,928 × 100  ≈ 13.5252%
    #   Consumer Staples = 63,170 / 734,928 × 100  ≈  8.5955%
    # -----------------------------------------------------------------------

    INTL_FUND_NAME = "International Equity Fund"

    INTL_EXPECTED_SECTORS = {
        "Technology":       {"mv": 346_888.00, "pct": 47.1999},
        "Energy":           {"mv": 225_470.00, "pct": 30.6793},
        "Automotive":       {"mv":  99_400.00, "pct": 13.5252},
        "Consumer Staples": {"mv":  63_170.00, "pct":  8.5955},
    }
    INTL_EXPECTED_TOTAL = 734_928.00

    def test_intl_fund_found(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        assert result.portfolio_found

    def test_intl_fund_no_warnings(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        assert result.warnings == [], (
            f"Expected no warnings, got: {result.warnings}"
        )

    def test_intl_fund_equity_count(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        assert result.equity_holding_count == 8
        assert result.excluded_bond_count == 0

    def test_intl_fund_total_market_value(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        assert abs(result.total_equity_market_value - self.INTL_EXPECTED_TOTAL) < self.MV_TOLERANCE, (
            f"Total equity MV: expected ~{self.INTL_EXPECTED_TOTAL:,.2f}, "
            f"got {result.total_equity_market_value:,.4f}"
        )

    def test_intl_fund_sector_count(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        assert len(result.sector_exposures) == 4, (
            f"Expected 4 sectors, got: {list(result.sector_exposures.keys())}"
        )

    def test_intl_fund_sector_exposures(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        for sector, expected in self.INTL_EXPECTED_SECTORS.items():
            actual_pct = result.sector_exposures.get(sector)
            assert actual_pct is not None, (
                f"Sector '{sector}' missing from result. "
                f"Got: {list(result.sector_exposures.keys())}"
            )
            assert abs(actual_pct - expected["pct"]) < self.PCT_TOLERANCE, (
                f"{sector}: expected {expected['pct']:.4f}%, "
                f"got {actual_pct:.4f}%"
            )

    def test_intl_fund_sector_market_values(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        for sector, expected in self.INTL_EXPECTED_SECTORS.items():
            actual_mv = result.sector_market_values.get(sector, 0.0)
            assert abs(actual_mv - expected["mv"]) < self.MV_TOLERANCE, (
                f"{sector} market value: expected {expected['mv']:,.2f}, "
                f"got {actual_mv:,.4f}"
            )

    def test_intl_fund_exposures_sum_to_100(self, prod_db_path):
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        total_pct = sum(result.sector_exposures.values())
        assert abs(total_pct - 100.0) < 0.01, (
            f"Sector percentages should sum to ~100%, got {total_pct:.4f}%"
        )

    def test_intl_fund_sorted_by_mv_descending(self, prod_db_path):
        """Sectors should be returned highest market value first."""
        result = calculate_sector_exposure(
            self.INTL_FUND_NAME, db_path=prod_db_path
        )
        mvs = list(result.sector_market_values.values())
        assert mvs == sorted(mvs, reverse=True), (
            f"Sectors not sorted by market value descending: {mvs}"
        )


# ---------------------------------------------------------------------------
# Edge-case tests — use isolated in-memory databases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """
    Edge-case tests using isolated temporary SQLite databases.
    These do not depend on the production database.
    """

    def test_portfolio_not_found(self, tmp_path):
        """Querying a name that doesn't exist returns portfolio_found=False."""
        with _InMemoryDB(tmp_path, "empty") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Real Fund', 'Active')"
            )
            db.commit()

        result = calculate_sector_exposure(
            "Nonexistent Fund", db_path=tmp_path / "empty.db"
        )
        assert not result.portfolio_found
        assert result.total_equity_market_value == 0.0
        assert result.sector_exposures == {}
        assert len(result.warnings) == 1
        assert "not found" in result.warnings[0].lower()

    def test_portfolio_with_no_holdings(self, tmp_path):
        """Portfolio exists but has no holdings at all."""
        with _InMemoryDB(tmp_path, "noholdings") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Empty Fund', 'Active')"
            )
            db.commit()

        result = calculate_sector_exposure(
            "Empty Fund", db_path=tmp_path / "noholdings.db"
        )
        assert result.portfolio_found
        assert result.equity_holding_count == 0
        assert result.total_equity_market_value == 0.0
        assert result.sector_exposures == {}
        assert len(result.warnings) >= 1
        assert "no holdings" in result.warnings[0].lower()

    def test_portfolio_bonds_only(self, tmp_path):
        """Portfolio with only bond holdings — no equity exposure to report."""
        with _InMemoryDB(tmp_path, "bondsonly") as db:
            # No sectors needed — bonds have NULL sector_id
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Bond Fund', 'Active')"
            )
            db.execute(
                "INSERT INTO securities VALUES "
                "(1, 'UST-10Y', 'Treasury 10Y', 'Bond', NULL, 102.50)"
            )
            db.execute(
                "INSERT INTO securities VALUES "
                "(2, 'UST-5Y', 'Treasury 5Y', 'Bond', NULL, 100.25)"
            )
            db.execute(
                "INSERT INTO holdings VALUES (1, 1, 1, 500)"
            )
            db.execute(
                "INSERT INTO holdings VALUES (2, 1, 2, 300)"
            )
            db.commit()

        result = calculate_sector_exposure(
            "Bond Fund", db_path=tmp_path / "bondsonly.db"
        )
        assert result.portfolio_found
        assert result.equity_holding_count == 0
        assert result.excluded_bond_count == 2
        assert result.total_equity_market_value == 0.0
        assert result.sector_exposures == {}
        assert len(result.warnings) >= 1
        # Warning must mention bond
        assert any("bond" in w.lower() for w in result.warnings)

    def test_holding_with_null_current_price(self, tmp_path):
        """Equity with NULL current_price is excluded with a warning."""
        with _InMemoryDB(tmp_path, "nullprice") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute("INSERT INTO sectors VALUES (2, 'Energy')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Mixed Fund', 'Active')"
            )
            # Good security
            db.execute(
                "INSERT INTO securities VALUES "
                "(1, 'AAPL', 'Apple', 'Stock', 1, 185.50)"
            )
            # Security with NULL price
            db.execute(
                "INSERT INTO securities VALUES "
                "(2, 'BROKEN', 'Broken Co', 'Stock', 2, NULL)"
            )
            db.execute("INSERT INTO holdings VALUES (1, 1, 1, 100)")   # qty=100
            db.execute("INSERT INTO holdings VALUES (2, 1, 2, 200)")   # qty=200, NULL price
            db.commit()

        result = calculate_sector_exposure(
            "Mixed Fund", db_path=tmp_path / "nullprice.db"
        )
        assert result.portfolio_found
        # Only the good holding counted
        assert result.equity_holding_count == 1
        assert result.excluded_no_price_count == 1
        # Technology: 100 × 185.50 = 18,550.00
        assert abs(result.total_equity_market_value - 18_550.00) < 0.01
        assert "Technology" in result.sector_exposures
        assert abs(result.sector_exposures["Technology"] - 100.0) < 0.01
        # Warning about the excluded holding
        assert len(result.warnings) >= 1
        assert any("null" in w.lower() or "price" in w.lower()
                   for w in result.warnings)

    def test_holding_with_null_sector_id_on_stock(self, tmp_path):
        """
        Stock with NULL sector_id is excluded as anomalous — reported as
        a warning and not counted in the sector breakdown.
        """
        with _InMemoryDB(tmp_path, "nosector") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Odd Fund', 'Active')"
            )
            # Normal stock
            db.execute(
                "INSERT INTO securities VALUES "
                "(1, 'AAPL', 'Apple', 'Stock', 1, 185.50)"
            )
            # Stock without a sector (anomalous)
            db.execute(
                "INSERT INTO securities VALUES "
                "(2, 'NOSEC', 'No Sector Co', 'Stock', NULL, 50.00)"
            )
            db.execute("INSERT INTO holdings VALUES (1, 1, 1, 100)")
            db.execute("INSERT INTO holdings VALUES (2, 1, 2, 50)")
            db.commit()

        result = calculate_sector_exposure(
            "Odd Fund", db_path=tmp_path / "nosector.db"
        )
        assert result.portfolio_found
        assert result.excluded_no_sector_count == 1
        # Only the valid stock counts
        assert result.equity_holding_count == 1
        # Technology: 100 × 185.50 = 18,550
        assert abs(result.total_equity_market_value - 18_550.00) < 0.01
        assert "Technology" in result.sector_exposures
        assert len(result.warnings) >= 1
        assert any("sector" in w.lower() or "null" in w.lower()
                   for w in result.warnings)

    def test_empty_portfolio_name(self, tmp_path):
        """Empty string or whitespace-only name is rejected immediately."""
        with _InMemoryDB(tmp_path, "emptyname") as db:
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'A Fund', 'Active')"
            )
            db.commit()

        for bad_name in ["", "   ", "\t"]:
            result = calculate_sector_exposure(
                bad_name, db_path=tmp_path / "emptyname.db"
            )
            assert not result.portfolio_found, (
                f"Expected portfolio_found=False for name={repr(bad_name)}"
            )
            assert len(result.warnings) >= 1

    def test_mixed_equity_and_bonds(self, tmp_path):
        """
        Portfolio with both stocks and bonds: only stocks contribute
        to sector exposure; bonds are counted in excluded_bond_count.
        """
        with _InMemoryDB(tmp_path, "mixed") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute("INSERT INTO sectors VALUES (4, 'Financials')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Mixed Fund', 'Active')"
            )
            # Two stocks
            db.execute(
                "INSERT INTO securities VALUES "
                "(1, 'AAPL', 'Apple', 'Stock', 1, 200.00)"
            )
            db.execute(
                "INSERT INTO securities VALUES "
                "(2, 'JPM', 'JPMorgan', 'Stock', 4, 150.00)"
            )
            # One bond
            db.execute(
                "INSERT INTO securities VALUES "
                "(3, 'UST', 'Treasury', 'Bond', NULL, 100.00)"
            )
            db.execute("INSERT INTO holdings VALUES (1, 1, 1, 100)")  # AAPL
            db.execute("INSERT INTO holdings VALUES (2, 1, 2, 200)")  # JPM
            db.execute("INSERT INTO holdings VALUES (3, 1, 3, 300)")  # Bond
            db.commit()

        result = calculate_sector_exposure(
            "Mixed Fund", db_path=tmp_path / "mixed.db"
        )
        # AAPL: 100 × 200 = 20,000  → Technology
        # JPM:  200 × 150 = 30,000  → Financials
        # Bond: excluded
        assert result.portfolio_found
        assert result.equity_holding_count == 2
        assert result.excluded_bond_count == 1
        assert result.warnings == []   # no data-quality issues

        total = 20_000.0 + 30_000.0
        assert abs(result.total_equity_market_value - total) < 0.01

        assert "Technology" in result.sector_exposures
        assert "Financials" in result.sector_exposures

        assert abs(result.sector_exposures["Technology"] - (20_000 / total * 100)) < 0.01
        assert abs(result.sector_exposures["Financials"] - (30_000 / total * 100)) < 0.01

    def test_result_is_valid_flags(self, tmp_path):
        """ExposureResult.is_valid() returns True only when usable data exists."""
        with _InMemoryDB(tmp_path, "valid") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Good Fund', 'Active')"
            )
            db.execute(
                "INSERT INTO securities VALUES "
                "(1, 'AAPL', 'Apple', 'Stock', 1, 100.00)"
            )
            db.execute("INSERT INTO holdings VALUES (1, 1, 1, 10)")
            db.commit()

        good = calculate_sector_exposure(
            "Good Fund", db_path=tmp_path / "valid.db"
        )
        assert good.is_valid()

        bad = calculate_sector_exposure(
            "Nonexistent", db_path=tmp_path / "valid.db"
        )
        assert not bad.is_valid()

    def test_to_display_dict_structure(self, tmp_path):
        """to_display_dict() returns all expected keys with correct types."""
        with _InMemoryDB(tmp_path, "display") as db:
            db.execute("INSERT INTO sectors VALUES (1, 'Technology')")
            db.execute(
                "INSERT INTO portfolios VALUES (1, 'Display Fund', 'Active')"
            )
            db.execute(
                "INSERT INTO securities VALUES "
                "(1, 'AAPL', 'Apple', 'Stock', 1, 150.00)"
            )
            db.execute("INSERT INTO holdings VALUES (1, 1, 1, 20)")
            db.commit()

        result = calculate_sector_exposure(
            "Display Fund", db_path=tmp_path / "display.db"
        )
        d = result.to_display_dict()

        required_keys = {
            "portfolio_name", "portfolio_found",
            "total_equity_market_value", "equity_holding_count",
            "excluded_bond_count", "sector_exposures",
            "sector_market_values", "warnings",
        }
        assert required_keys.issubset(d.keys()), (
            f"Missing keys: {required_keys - d.keys()}"
        )
        assert isinstance(d["sector_exposures"], dict)
        assert isinstance(d["sector_market_values"], dict)
        assert isinstance(d["warnings"], list)


# ---------------------------------------------------------------------------
# Direct runner — print a formatted summary when run as a script
# ---------------------------------------------------------------------------

def _print_summary(label: str, result: ExposureResult) -> None:
    from src.tools.exposure_tool import format_exposure_result
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    print(format_exposure_result(result))
    d = result.to_display_dict()
    print(f"\n  Sector exposures (%):")
    for sector, pct in d["sector_exposures"].items():
        print(f"    {sector:<24} {pct:>7.2f}%")
    print(f"\n  Total equity MV: ${d['total_equity_market_value']:>14,.2f}")
    print(f"  Equity holdings: {d['equity_holding_count']}")
    print(f"  Excluded bonds:  {d['excluded_bond_count']}")
    if d["warnings"]:
        print("\n  Warnings:")
        for w in d["warnings"]:
            print(f"    [WARN] {w}")


if __name__ == "__main__":
    prod_db = PROJECT_ROOT / "portfolio_database.db"
    if not prod_db.exists():
        print("[ERROR] portfolio_database.db not found. Run 'python init_db.py' first.")
        sys.exit(1)

    print("\nRunning ground-truth calculations against production database...")

    r9 = calculate_sector_exposure("Tech Innovation Fund", db_path=prod_db)
    _print_summary("Q9 — Tech Innovation Fund", r9)

    r10 = calculate_sector_exposure("International Equity Fund", db_path=prod_db)
    _print_summary("Q10 — International Equity Fund", r10)

    print("\n\nRunning pytest for full test suite...")
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
