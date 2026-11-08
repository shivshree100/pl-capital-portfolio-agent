# Sector Exposure Calculator

## Purpose

Given a portfolio name, calculate the percentage of the portfolio's equity
market value that is allocated to each sector.  This is used to answer
natural-language questions such as:
- "What are the sector exposures for the Tech Innovation Fund?"
- "Calculate the sector exposure breakdown for the International Equity Fund."

---

## Calculation Formula

For each equity holding in the portfolio:

```
market_value = quantity × securities.current_price
```

Holdings are then grouped by sector:

```
sector_market_value[sector] = Σ market_value  for all holdings in that sector
total_equity_market_value   = Σ sector_market_value
sector_exposure[sector]     = (sector_market_value[sector] / total_equity_market_value) × 100
```

Results are sorted by market value descending (largest sector exposure first).

### Example — International Equity Fund

| Symbol | Sector           | Qty  | Price    | Market Value  |
|--------|------------------|------|----------|---------------|
| AAPL   | Technology       | 600  | $185.50  | $111,300.00   |
| MSFT   | Technology       | 200  | $415.26  | $83,052.00    |
| GOOGL  | Technology       | 30   | $2750.80 | $82,524.00    |
| AVGO   | Technology       | 80   | $875.15  | $70,012.00    |
| TSLA   | Automotive       | 400  | $248.50  | $99,400.00    |
| XOM    | Energy           | 1000 | $98.75   | $98,750.00    |
| CVX    | Energy           | 800  | $158.40  | $126,720.00   |
| HD     | Consumer Staples | 200  | $315.85  | $63,170.00    |

| Sector           | Market Value  | Exposure |
|------------------|---------------|----------|
| Technology       | $346,888.00   | 47.20%   |
| Energy           | $225,470.00   | 30.68%   |
| Automotive       | $99,400.00    | 13.53%   |
| Consumer Staples | $63,170.00    | 8.60%    |
| **Total**        | **$734,928**  | **100%** |

---

## Why Bonds Are Excluded

The assignment specification states: *"Ignore bond holdings for this
calculation, only consider the equities in the portfolio."*

Bonds are excluded for two reasons:

1. **No sector mapping**: Bond securities have `sector_id = NULL` in the
   database by design.  They cannot be assigned to an equity sector such as
   Technology or Energy.

2. **Different risk profile**: Sector exposure is a measure of equity
   concentration risk.  Mixing bond market value into the denominator would
   dilute the sector percentages and make them meaningless as an indicator
   of equity sector concentration.

Bonds are counted in `ExposureResult.excluded_bond_count` so the caller
can see how much of the portfolio's holdings were skipped.

---

## Why `holdings.current_weight` Is Not Used

The `holdings` table contains a `current_weight` column, but the calculator
**does not use it** for the following reasons:

1. **It is a pre-computed snapshot**.  `current_weight` was calculated at
   the time the holding was recorded (all purchase dates are 2023-09-25) and
   reflects prices from that date, not today's `securities.current_price`.

2. **It is a fraction of total portfolio value**, not of equity-only value.
   A portfolio with bonds would have `current_weight` values that sum to 1.0
   across both stocks and bonds.  Using it directly for sector exposure would
   include bond value in the denominator.

3. **Recomputing is exact**.  `quantity × current_price` uses the same
   `current_price` column that the rest of the system uses, producing
   internally consistent results.

4. **The ground-truth SQL for Q7** (the hardest evaluation query) also
   recomputes `SUM(h.quantity * s.current_price)` rather than summing
   `current_weight`, confirming this is the intended approach.

---

## How Missing Data Is Handled

The calculator handles four categories of data-quality issues:

| Issue | Behaviour |
|---|---|
| Portfolio name not found | Returns `portfolio_found=False`, warning with the exact name |
| Portfolio has no holdings | Returns empty breakdown, warning indicating no holdings exist |
| Portfolio has bonds only | Returns empty breakdown, warning noting bond-only portfolio |
| Equity holding with NULL `current_price` | Excluded, warning lists the affected symbols; remaining equities still calculated |
| Stock with NULL `sector_id` (anomalous) | Excluded, warning lists the affected symbols; remaining equities still calculated |
| Total equity market value is zero | Returns empty exposure dict, warning |

In all cases:
- The function **never silently ignores a data problem**.
- The function **never raises** on a business-logic issue (only on
  infrastructure failures such as a missing database file).
- Every exclusion is reflected in a counter field (`excluded_bond_count`,
  `excluded_no_price_count`, `excluded_no_sector_count`) and a warning string.

---

## Module Structure

```
src/tools/exposure_tool.py

  ExposureResult          dataclass   Structured result with all fields
  calculate_sector_exposure()         Main calculation function (pure logic)
  format_exposure_result()            Presentation helper (no calculation)
```

### `calculate_sector_exposure(portfolio_name, db_path=None) -> ExposureResult`

- `portfolio_name`: exact case-sensitive portfolio name
- `db_path`: optional override for testing; defaults to `portfolio_database.db`
- Returns an `ExposureResult` in all cases (never raises for business logic)
- Raises `RuntimeError` only if the database cannot be opened

### `ExposureResult` fields

| Field | Type | Meaning |
|---|---|---|
| `portfolio_name` | str | Name as requested |
| `portfolio_found` | bool | False if not in database |
| `total_equity_market_value` | float | Sum of all equity market values |
| `sector_market_values` | dict[str, float] | Per-sector market value |
| `sector_exposures` | dict[str, float] | Per-sector % (sums to ~100) |
| `equity_holding_count` | int | Holdings included in calculation |
| `excluded_bond_count` | int | Bond holdings skipped |
| `excluded_no_price_count` | int | Equities skipped due to NULL price |
| `excluded_no_sector_count` | int | Equities skipped due to NULL sector |
| `warnings` | list[str] | Data-quality messages (empty = clean) |

---

## How to Run the Tests

### Prerequisites

The production database must exist:

```bash
python init_db.py        # if not already done
```

### Run all tests

```bash
python -m pytest tests/test_exposure_tool.py -v
```

### Run only ground-truth tests (Q9 and Q10)

```bash
python -m pytest tests/test_exposure_tool.py -v -k "TestGroundTruth"
```

### Run only edge-case tests

```bash
python -m pytest tests/test_exposure_tool.py -v -k "TestEdgeCases"
```

### Run as a script (formatted output + pytest)

```bash
python tests/test_exposure_tool.py
```

### Expected test counts

| Suite | Tests |
|---|---|
| `TestGroundTruth` | 12 tests (6 per portfolio) |
| `TestEdgeCases` | 8 tests |
| **Total** | **20 tests** |

All 20 tests should pass when the production database is populated correctly.

---

## Integration with the Agent (Phase 3 Preview)

The agent will call `calculate_sector_exposure(portfolio_name)` when it
detects that a user question requires sector exposure rather than SQL.  The
`ExposureResult.to_display_dict()` method produces a JSON-serialisable
dictionary suitable for inclusion in an LLM response or a Streamlit widget.
