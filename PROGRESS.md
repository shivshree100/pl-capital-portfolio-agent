# Project Progress — PL Capital Portfolio Analytics Agent

## Overview

An AI-powered command-line agent that answers natural-language questions about investment portfolio data using a SQLite database and the Google Gemini API.

---

## What's Built

### Phase 1 — Database Foundation
- SQLite database loaded from 9 CSV files (603 rows, 9 tables)
- Connection factory with FK enforcement and WAL mode
- Loader with 5-point validation (tables, row counts, PKs, FKs, sample JOIN)
- CLI: `python init_db.py` (load), `--force` (reload), `--verify` (check)

### Phase 2 — Sector Exposure Calculator
- Deterministic tool: `market_value = quantity × current_price`, grouped by sector
- Excludes bonds, ignores stale `current_weight` column
- Returns structured `ExposureResult` with warnings for data issues
- 25 tests (ground-truth + edge cases)

### Phase 3 — AI Agent + CLI
- Gemini function-calling agent with 2 tools: `run_sql_query` and `calculate_exposure`
- `mode=AUTO` — Gemini picks a tool for portfolio questions, returns text for off-topic
- SQL safety: SELECT-only validation, 500-row cap, no DDL/DML
- Two-step flow: dispatch (pick tool) → synthesise (compose NL answer)
- Fallback model chain with 503 retry logic
- Interactive CLI: `python main.py`

### Phase 4 — Ground-Truth Evaluator
- Runs all 10 questions from `ground_truth_dataset.json`
- Computes expected results at runtime (no hardcoded answers)
- Compares structured data (not NL text) with type-aware logic
- Metrics: routing accuracy, data accuracy (of routed questions), overall pass rate
- CLI: `python evaluate.py` or `python evaluate.py --question Q1,Q5,Q9`

---

## How to Run

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
cp .env.example .env   # then add your GEMINI_API_KEY

# 3. Initialize database
python init_db.py

# 4. Run the agent
python main.py

# 5. Run evaluation
python evaluate.py

# 6. Run tests
python -m pytest tests/ -v -k "not Integration and not test_api"
```

---

## Test Summary

| Suite | Tests | What it covers |
|---|---|---|
| Exposure calculator | 25 | Ground-truth Q9/Q10 + edge cases |
| Agent routing (mocked) | 8 | SQL/exposure/off-topic/malformed routing |
| Agent dispatch (mocked) | 4 | Structured result return without synthesis |
| Evaluator | 49 | Value/list/table/exposure comparison + scoring |
| Live API test | 1 | Gemini connectivity (skipped without key) |
| Integration tests | 5 | End-to-end with real API (skipped without key) |
| **Total** | **92** | 86 unit tests pass without API key |

---

## Current Status

- All 4 phases implemented and tested
- 86 unit tests pass without a live API key
- Evaluator runs all 10 ground-truth questions against the live agent
- Architecture: 2 tools with `mode=AUTO` (simpler routing, no decline escape hatch)

## Project Structure

```
src/
  database/     connection.py, loader.py
  tools/        sql_tool.py, exposure_tool.py
  agent/        portfolio_agent.py
  evaluation/   evaluator.py
tests/          test_exposure_tool.py, test_agent.py, test_evaluator.py, test_api.py
docs/           database_setup.md, exposure_calculator.md
init_db.py      Database initializer CLI
main.py         Interactive agent CLI
evaluate.py     Ground-truth evaluator CLI
```
