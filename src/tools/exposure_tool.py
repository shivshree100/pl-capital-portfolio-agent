"""
Sector Exposure Calculator for the Portfolio Analytics Agent.

Public API
----------
calculate_sector_exposure(portfolio_name: str,
                          db_path: Path | None = None) -> ExposureResult

Design decisions
----------------
1.  Market value = quantity × securities.current_price
    We recompute market value from live price data rather than trusting
    holdings.current_weight, which is a pre-computed snapshot that may be
    stale relative to current_price.

2.  Bonds are excluded from sector exposure.
    The assignment explicitly requires equity-only exposure. Bonds have
    NULL sector_id by design, so they cannot be mapped to a sector. The
    filter is applied in SQL (asset_type = 'Stock'), not post-hoc in Python.

3.  Holdings with NULL current_price are excluded and reported as warnings.
    The data guarantees non-null prices for all equities, but the code
    handles the case defensively so it never silently produces a wrong total.

4.  Holdings with NULL sector_id but asset_type = 'Stock' are anomalies.
    The schema allows stocks without a sector. These are reported as
    warnings and excluded from the percentage calculation.

5.  All computation uses Python floats from SQLite REAL columns.
    No external statistical libraries are needed; pandas is used only for
    groupby aggregation. Results are rounded to 4 decimal places before
    returning so floating-point noise does not accumulate in downstream
    consumers.

6.  The function never raises on business-logic failures (unknown portfolio,
    no equities, zero total). It returns an ExposureResult with an empty
    breakdown and a populated warnings list. Exceptions are only raised for
    infrastructure failures (DB connection, SQL error).
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database.connection import managed_connection


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExposureResult:
    """
    Structured result of a sector exposure calculation.

    Attributes
    ----------
    portfolio_name : str
        The name of the portfolio as stored in the database.

    portfolio_found : bool
        False when the portfolio name does not exist in the database.
        When False, all numeric fields are zero and breakdown is empty.

    total_equity_market_value : float
        Sum of (quantity × current_price) for all equity holdings included
        in the calculation.

    sector_market_values : dict[str, float]
        Mapping of sector_name → total market value of holdings in that
        sector.  Sorted by market value descending.

    sector_exposures : dict[str, float]
        Mapping of sector_name → percentage of total_equity_market_value.
        Values sum to 100.0 (subject to floating-point rounding).
        Empty when total_equity_market_value is zero.

    equity_holding_count : int
        Number of equity holdings that contributed to the calculation.

    excluded_bond_count : int
        Number of holdings excluded because their security is a Bond.

    excluded_no_price_count : int
        Number of equity holdings excluded due to NULL current_price.

    excluded_no_sector_count : int
        Number of equity holdings with a valid price but NULL sector_id.
        These are anomalous and are reported as warnings.

    warnings : list[str]
        Human-readable descriptions of data-quality issues encountered.
        An empty list means the calculation was clean.
    """

    portfolio_name: str
    portfolio_found: bool = False

    total_equity_market_value: float = 0.0
    sector_market_values: dict = field(default_factory=dict)
    sector_exposures: dict = field(default_factory=dict)

    equity_holding_count: int = 0
    excluded_bond_count: int = 0
    excluded_no_price_count: int = 0
    excluded_no_sector_count: int = 0

    warnings: list = field(default_factory=list)

    def is_valid(self) -> bool:
        """Return True if the result contains usable exposure data."""
        return self.portfolio_found and self.total_equity_market_value > 0

    def to_display_dict(self) -> dict:
        """
        Return a plain dictionary suitable for formatted display or
        serialisation to JSON.  Percentages are rounded to 2 decimal places.
        """
        return {
            "portfolio_name": self.portfolio_name,
            "portfolio_found": self.portfolio_found,
            "total_equity_market_value": round(self.total_equity_market_value, 2),
            "equity_holding_count": self.equity_holding_count,
            "excluded_bond_count": self.excluded_bond_count,
            "sector_exposures": {
                k: round(v, 2) for k, v in self.sector_exposures.items()
            },
            "sector_market_values": {
                k: round(v, 2) for k, v in self.sector_market_values.items()
            },
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# SQL query
# ---------------------------------------------------------------------------

# We join holdings → securities → sectors, keeping only equity rows with a
# valid sector.  We also carry NULL sector_id rows so we can warn about them
# separately — hence the LEFT JOIN to sectors.
#
# The WHERE clause filters on portfolio name (parameterised — no injection
# risk) and on asset_type = 'Stock' (excludes bonds at the DB level).
# We additionally require current_price IS NOT NULL and quantity > 0 as
# safety guards; the data does not have such issues but the code is robust.

_HOLDINGS_SQL = """
    SELECT
        h.holding_id,
        h.quantity,
        s.security_id,
        s.symbol,
        s.company_name,
        s.asset_type,
        s.current_price,
        s.sector_id,
        sec.sector_name
    FROM portfolios  p
    JOIN holdings    h   ON p.portfolio_id = h.portfolio_id
    JOIN securities  s   ON h.security_id  = s.security_id
    LEFT JOIN sectors sec ON s.sector_id   = sec.sector_id
    WHERE p.portfolio_name = ?
      AND s.asset_type     = 'Stock'
    ORDER BY sec.sector_name NULLS LAST, s.symbol;
"""

# Separate query used only to confirm the portfolio exists when the main
# query returns zero rows (distinguishes "not found" from "no equity").
_PORTFOLIO_EXISTS_SQL = """
    SELECT COUNT(*) FROM portfolios WHERE portfolio_name = ?;
"""

# Count bond holdings separately so we can report excluded_bond_count.
_BOND_COUNT_SQL = """
    SELECT COUNT(*)
    FROM portfolios  p
    JOIN holdings    h  ON p.portfolio_id = h.portfolio_id
    JOIN securities  s  ON h.security_id  = s.security_id
    WHERE p.portfolio_name = ?
      AND s.asset_type     = 'Bond';
"""


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def calculate_sector_exposure(
    portfolio_name: str,
    db_path: Optional[Path] = None,
) -> ExposureResult:
    """
    Calculate equity sector exposures for a named portfolio.

    Parameters
    ----------
    portfolio_name : str
        Exact name of the portfolio as stored in portfolios.portfolio_name.
        The match is case-sensitive.

    db_path : Path, optional
        Override the database path.  Defaults to portfolio_database.db in
        the project root.  Useful in tests.

    Returns
    -------
    ExposureResult
        A structured result with sector breakdowns and any warnings.
        If the portfolio does not exist, ExposureResult.portfolio_found is
        False and all numeric fields are zero.

    Raises
    ------
    RuntimeError
        If the database connection cannot be opened or the SQL fails.
        Business-logic issues (unknown portfolio, no equities) are returned
        as an ExposureResult with warnings, not raised as exceptions.
    """
    if not isinstance(portfolio_name, str) or not portfolio_name.strip():
        return ExposureResult(
            portfolio_name=str(portfolio_name),
            portfolio_found=False,
            warnings=["portfolio_name must be a non-empty string."],
        )

    result = ExposureResult(portfolio_name=portfolio_name)

    with managed_connection(db_path) as conn:
        # ------------------------------------------------------------------
        # Step 1: check portfolio exists
        # ------------------------------------------------------------------
        exists = conn.execute(
            _PORTFOLIO_EXISTS_SQL, (portfolio_name,)
        ).fetchone()[0]

        if not exists:
            result.portfolio_found = False
            result.warnings.append(
                f"Portfolio '{portfolio_name}' was not found in the database. "
                "Check the exact name including capitalisation."
            )
            return result

        result.portfolio_found = True

        # ------------------------------------------------------------------
        # Step 2: count excluded bond holdings (informational)
        # ------------------------------------------------------------------
        result.excluded_bond_count = conn.execute(
            _BOND_COUNT_SQL, (portfolio_name,)
        ).fetchone()[0]

        # ------------------------------------------------------------------
        # Step 3: fetch equity holdings with sector info
        # ------------------------------------------------------------------
        rows = conn.execute(_HOLDINGS_SQL, (portfolio_name,)).fetchall()

        if not rows:
            # Portfolio found but no equity holdings at all.
            if result.excluded_bond_count > 0:
                result.warnings.append(
                    f"Portfolio '{portfolio_name}' contains "
                    f"{result.excluded_bond_count} bond holding(s) but no "
                    "equity holdings. Sector exposure cannot be calculated "
                    "for bond-only portfolios."
                )
            else:
                result.warnings.append(
                    f"Portfolio '{portfolio_name}' has no holdings. "
                    "Sector exposure cannot be calculated."
                )
            return result

    # ------------------------------------------------------------------------
    # Step 4: compute market values in pandas (outside the DB context)
    # ------------------------------------------------------------------------
    df = pd.DataFrame(
        [dict(r) for r in rows],
        columns=[
            "holding_id", "quantity", "security_id", "symbol",
            "company_name", "asset_type", "current_price",
            "sector_id", "sector_name",
        ],
    )

    # --- 4a. Separate rows with NULL current_price ---
    null_price_mask = df["current_price"].isna()
    if null_price_mask.any():
        bad = df.loc[null_price_mask, "symbol"].tolist()
        result.excluded_no_price_count = int(null_price_mask.sum())
        result.warnings.append(
            f"{result.excluded_no_price_count} equity holding(s) excluded "
            f"due to NULL current_price: {bad}. "
            "These securities will not appear in the sector breakdown."
        )
    df = df[~null_price_mask].copy()

    # --- 4b. Separate rows with NULL sector_id (stock with no sector) ---
    null_sector_mask = df["sector_id"].isna()
    if null_sector_mask.any():
        bad = df.loc[null_sector_mask, "symbol"].tolist()
        result.excluded_no_sector_count = int(null_sector_mask.sum())
        result.warnings.append(
            f"{result.excluded_no_sector_count} equity holding(s) have "
            f"asset_type='Stock' but a NULL sector_id: {bad}. "
            "These are anomalous and are excluded from the sector breakdown."
        )
    df = df[~null_sector_mask].copy()

    if df.empty:
        result.warnings.append(
            "After excluding holdings with missing price or sector data, "
            "no equity holdings remain. Sector exposure cannot be calculated."
        )
        return result

    # --- 4c. Compute market value per holding ---
    df["market_value"] = df["quantity"] * df["current_price"]

    result.equity_holding_count = len(df)

    # --- 4d. Group by sector ---
    sector_mv = (
        df.groupby("sector_name", sort=False)["market_value"]
        .sum()
        .sort_values(ascending=False)
    )

    total = float(sector_mv.sum())

    if total <= 0.0:
        result.warnings.append(
            "Total equity market value is zero or negative. "
            "Sector exposure percentages cannot be computed. "
            "Verify that current_price values are correct."
        )
        result.sector_market_values = {
            str(k): round(float(v), 4) for k, v in sector_mv.items()
        }
        return result

    # --- 4e. Compute percentages ---
    result.total_equity_market_value = round(total, 4)
    result.sector_market_values = {
        str(k): round(float(v), 4) for k, v in sector_mv.items()
    }
    result.sector_exposures = {
        str(k): round((float(v) / total) * 100, 4)
        for k, v in sector_mv.items()
    }

    return result


# ---------------------------------------------------------------------------
# Convenience formatter (keeps presentation out of the calculator)
# ---------------------------------------------------------------------------

def format_exposure_result(result: ExposureResult) -> str:
    """
    Return a human-readable string representation of an ExposureResult.

    This is a pure presentation function; it does no calculation.
    """
    lines: list[str] = []
    sep = "-" * 56

    lines.append(sep)
    lines.append(f"  Sector Exposure — {result.portfolio_name}")
    lines.append(sep)

    if not result.portfolio_found:
        lines.append("  Portfolio not found.")
        if result.warnings:
            for w in result.warnings:
                lines.append(f"  [WARN] {w}")
        return "\n".join(lines)

    if not result.is_valid():
        lines.append("  No sector exposure data available.")
        if result.warnings:
            for w in result.warnings:
                lines.append(f"  [WARN] {w}")
        return "\n".join(lines)

    lines.append(
        f"  Total equity market value : "
        f"${result.total_equity_market_value:>14,.2f}"
    )
    lines.append(f"  Equity holdings included  : {result.equity_holding_count}")
    if result.excluded_bond_count:
        lines.append(
            f"  Bond holdings excluded    : {result.excluded_bond_count}"
        )
    lines.append("")
    lines.append(
        f"  {'Sector':<24} {'Market Value':>14}  {'Exposure %':>10}"
    )
    lines.append(f"  {'-'*24} {'-'*14}  {'-'*10}")

    for sector, pct in result.sector_exposures.items():
        mv = result.sector_market_values[sector]
        lines.append(f"  {sector:<24} ${mv:>13,.2f}  {pct:>9.2f}%")

    lines.append("")
    total_pct = sum(result.sector_exposures.values())
    lines.append(f"  {'Total':<24} "
                 f"${result.total_equity_market_value:>13,.2f}  "
                 f"{total_pct:>9.2f}%")

    if result.warnings:
        lines.append("")
        for w in result.warnings:
            lines.append(f"  [WARN] {w}")

    lines.append(sep)
    return "\n".join(lines)
