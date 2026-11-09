"""
api.py — REST API for the PL Capital Portfolio Analytics Agent.

Features
--------
- Concurrent query handling: all blocking DB/agent calls run in a thread
  pool via asyncio.to_thread(), keeping the event loop free.
- In-memory TTL caching: SQL read results and exposure calculations are
  cached for 120s/300s respectively (see src/cache.py).
- RAG context injection: relevant database facts are pre-fetched and
  injected into the Gemini prompt for better accuracy (src/rag.py).
- Proper HTTP status codes: 200, 400, 404, 503, 500.
- Automatic docs: http://localhost:8000/docs and /redoc.

Running
-------
    python api.py                         # development server (reload on)
    uvicorn api:app --host 0.0.0.0 --port 8000  # production
"""

import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import managed_connection, get_db_path, check_connection
from src.tools.sql_tool import execute_sql_query, validate_sql
from src.tools.exposure_tool import calculate_sector_exposure
from src.cache import query_cache, exposure_cache, _NOT_FOUND


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    database: str
    agent_available: bool
    cache: dict


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000,
        description="Natural-language question about portfolio data.",
        examples=["How many portfolios do we have?"],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    tool_used: Optional[str] = None
    cached: bool = False


class SqlRequest(BaseModel):
    sql: str = Field(
        ..., min_length=6,
        description="A valid SQLite SELECT statement.",
        examples=["SELECT COUNT(*) FROM portfolios"],
    )


class SqlResponse(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    cached: bool = False


class ExposureResponse(BaseModel):
    portfolio_name: str
    portfolio_found: bool
    total_equity_market_value: float
    equity_holding_count: int
    excluded_bond_count: int
    sector_exposures: dict[str, float]
    sector_market_values: dict[str, float]
    warnings: list[str]
    cached: bool = False


# ---------------------------------------------------------------------------
# Shared agent instance
# ---------------------------------------------------------------------------

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    if not check_connection():
        print("[WARN] Database not found. Run 'python init_db.py' first.")
    try:
        from src.agent.portfolio_agent import PortfolioAgent
        _agent = PortfolioAgent()
        print("[INFO] PortfolioAgent initialised.")
    except RuntimeError as exc:
        _agent = None
        print(f"[WARN] Agent unavailable: {exc}")
    yield
    _agent = None
    query_cache.clear()
    exposure_cache.clear()
    print("[INFO] API shutdown. Caches cleared.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PL Capital — Portfolio Analytics API",
    description=(
        "REST API with concurrent query handling, in-memory caching, "
        "and RAG-augmented natural-language Q&A."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Internal helpers — all synchronous, run in thread pool
# ---------------------------------------------------------------------------

def _run_sql_sync(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a trusted internal SQL query. NOT for user input."""
    with managed_connection() as conn:
        cursor = conn.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _portfolio_id_sync(name: str) -> int:
    rows = _run_sql_sync(
        "SELECT portfolio_id FROM portfolios WHERE portfolio_name = ?", (name,)
    )
    if not rows:
        raise ValueError(f"Portfolio '{name}' not found.")
    return rows[0]["portfolio_id"]


# ---------------------------------------------------------------------------
# Async wrappers — move blocking calls off the event loop
# ---------------------------------------------------------------------------

async def _run_sql(sql: str, params: tuple = ()) -> list[dict]:
    """Run a DB query concurrently in a thread pool."""
    cache_key = f"sql:{sql}:{params}"
    cached = query_cache.get(cache_key)
    if cached is not _NOT_FOUND:
        return cached, True  # (data, from_cache)
    data = await asyncio.to_thread(_run_sql_sync, sql, params)
    query_cache.set(cache_key, data)
    return data, False


async def _get_portfolio_id(name: str) -> int:
    """Resolve portfolio name to ID, with 404 on miss."""
    try:
        return await asyncio.to_thread(_portfolio_id_sync, name)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc) + " Use GET /portfolios to see available names."
        )


# ---------------------------------------------------------------------------
# GET /  — health + cache stats
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check including database status and cache statistics."""
    db_ok = await asyncio.to_thread(check_connection)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="connected" if db_ok else "unavailable — run python init_db.py",
        agent_available=_agent is not None,
        cache={
            "query_cache": query_cache.stats(),
            "exposure_cache": exposure_cache.stats(),
        },
    )


# ---------------------------------------------------------------------------
# GET /portfolios
# ---------------------------------------------------------------------------

@app.get("/portfolios", tags=["Portfolios"])
async def list_portfolios():
    """All portfolios with key attributes."""
    data, _ = await _run_sql(
        "SELECT portfolio_id, portfolio_name, total_aum, target_risk_level, "
        "strategy_type, benchmark_index, status, creation_date "
        "FROM portfolios ORDER BY portfolio_name"
    )
    return data


@app.get("/portfolios/{portfolio_name}", tags=["Portfolios"])
async def get_portfolio(portfolio_name: str):
    """Single portfolio by exact name."""
    data, _ = await _run_sql(
        "SELECT * FROM portfolios WHERE portfolio_name = ?", (portfolio_name,)
    )
    if not data:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_name}' not found.")
    return data[0]


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/exposure
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/exposure",
         response_model=ExposureResponse, tags=["Portfolios"])
async def get_sector_exposure(portfolio_name: str):
    """
    Equity sector exposure for a portfolio.
    Deterministic calculation — no Gemini API call.
    Results cached for 300 seconds.
    """
    cache_key = f"exposure:{portfolio_name}"
    cached = exposure_cache.get(cache_key)
    if cached is not _NOT_FOUND:
        cached["cached"] = True
        return ExposureResponse(**cached)

    result = await asyncio.to_thread(calculate_sector_exposure, portfolio_name)

    if not result.portfolio_found:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_name}' not found.")

    payload = {
        "portfolio_name": result.portfolio_name,
        "portfolio_found": result.portfolio_found,
        "total_equity_market_value": round(result.total_equity_market_value, 2),
        "equity_holding_count": result.equity_holding_count,
        "excluded_bond_count": result.excluded_bond_count,
        "sector_exposures": {k: round(v, 4) for k, v in result.sector_exposures.items()},
        "sector_market_values": {k: round(v, 2) for k, v in result.sector_market_values.items()},
        "warnings": result.warnings,
    }
    exposure_cache.set(cache_key, payload)
    return ExposureResponse(**payload, cached=False)


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/performance
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/performance", tags=["Portfolios"])
async def get_performance(portfolio_name: str):
    """Performance history (NAV, returns, Sharpe, volatility). Most recent first."""
    pid = await _get_portfolio_id(portfolio_name)
    rows, from_cache = await _run_sql(
        "SELECT performance_date, nav, total_return_1m, total_return_3m, "
        "total_return_6m, total_return_1y, volatility, sharpe_ratio, max_drawdown "
        "FROM portfolio_performance WHERE portfolio_id = ? "
        "ORDER BY performance_date DESC",
        (pid,),
    )
    return {"portfolio_name": portfolio_name, "records": rows, "cached": from_cache}


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/risk
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/risk", tags=["Portfolios"])
async def get_risk(portfolio_name: str):
    """Risk metrics (VaR, beta, Sharpe, Sortino etc.). Most recent first."""
    pid = await _get_portfolio_id(portfolio_name)
    rows, from_cache = await _run_sql(
        "SELECT calculation_date, var_95, var_99, cvar_95, beta, "
        "correlation_sp500, tracking_error, information_ratio, sortino_ratio "
        "FROM risk_metrics WHERE portfolio_id = ? "
        "ORDER BY calculation_date DESC",
        (pid,),
    )
    return {"portfolio_name": portfolio_name, "records": rows, "cached": from_cache}


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/holdings
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/holdings", tags=["Portfolios"])
async def get_holdings(portfolio_name: str):
    """All holdings for a portfolio with live market values, sorted by value."""
    pid = await _get_portfolio_id(portfolio_name)
    rows, from_cache = await _run_sql(
        "SELECT h.holding_id, s.symbol, s.company_name, s.asset_type, "
        "sec.sector_name, h.quantity, h.purchase_price, h.cost_basis, "
        "s.current_price, ROUND(h.quantity * s.current_price, 2) AS market_value "
        "FROM holdings h "
        "JOIN securities s ON h.security_id = s.security_id "
        "LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
        "WHERE h.portfolio_id = ? ORDER BY market_value DESC",
        (pid,),
    )
    return {"portfolio_name": portfolio_name, "holdings": rows, "cached": from_cache}


# ---------------------------------------------------------------------------
# GET /securities
# ---------------------------------------------------------------------------

@app.get("/securities", tags=["Securities"])
async def list_securities(
    asset_type: Optional[str] = Query(
        default=None, description="Filter: 'Stock' or 'Bond'"
    )
):
    """All securities with sector info. Filter by asset_type if needed."""
    if asset_type and asset_type not in ("Stock", "Bond"):
        raise HTTPException(status_code=400, detail="asset_type must be 'Stock' or 'Bond'.")

    if asset_type:
        data, _ = await _run_sql(
            "SELECT s.security_id, s.symbol, s.company_name, s.asset_type, "
            "sec.sector_name, s.current_price, s.market_cap, s.exchange, s.country "
            "FROM securities s LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
            "WHERE s.asset_type = ? ORDER BY s.symbol",
            (asset_type,),
        )
    else:
        data, _ = await _run_sql(
            "SELECT s.security_id, s.symbol, s.company_name, s.asset_type, "
            "sec.sector_name, s.current_price, s.market_cap, s.exchange, s.country "
            "FROM securities s LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
            "ORDER BY s.symbol"
        )
    return data


# ---------------------------------------------------------------------------
# GET /sectors
# ---------------------------------------------------------------------------

@app.get("/sectors", tags=["Sectors"])
async def list_sectors():
    """All sectors with security counts."""
    data, _ = await _run_sql(
        "SELECT sec.sector_id, sec.sector_name, sec.industry_group, "
        "COUNT(s.security_id) AS security_count "
        "FROM sectors sec LEFT JOIN securities s ON sec.sector_id = s.sector_id "
        "GROUP BY sec.sector_id ORDER BY sec.sector_name"
    )
    return data


# ---------------------------------------------------------------------------
# GET /cache  — cache inspection
# ---------------------------------------------------------------------------

@app.get("/cache", tags=["Health"])
async def cache_stats():
    """Inspect current cache state (entry counts, TTLs, maxsize)."""
    return {
        "query_cache": query_cache.stats(),
        "exposure_cache": exposure_cache.stats(),
    }


@app.delete("/cache", tags=["Health"])
async def clear_cache():
    """Flush all caches (useful after data updates)."""
    query_cache.clear()
    exposure_cache.clear()
    return {"message": "All caches cleared."}


# ---------------------------------------------------------------------------
# POST /ask  — natural-language Q&A with RAG + caching
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask_question(request: AskRequest):
    """
    Answer a natural-language question using the AI agent.

    - RAG: relevant database facts are pre-fetched and injected into the
      Gemini prompt for better accuracy.
    - Caching: identical questions return cached answers within 120 seconds.
    - Concurrent: runs in a thread pool — does not block other requests.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not available. Ensure GEMINI_API_KEY is set and restart.",
        )

    # Check answer cache first
    from src.cache import _NOT_FOUND
    cache_key = f"agent:answer:{request.question.strip().lower()}"
    cached_answer = query_cache.get(cache_key)
    if cached_answer is not _NOT_FOUND:
        return AskResponse(
            question=request.question,
            answer=cached_answer,
            cached=True,
        )

    try:
        # Run blocking Gemini call in thread pool
        answer = await asyncio.to_thread(_agent.answer, request.question)

        # Cache the result
        query_cache.set(cache_key, answer)

        from src.agent.portfolio_agent import _looks_like_portfolio_question
        tool_hint = "run_sql_query" if _looks_like_portfolio_question(request.question) else None

        return AskResponse(
            question=request.question,
            answer=answer,
            tool_used=tool_hint,
            cached=False,
        )

    except RuntimeError as exc:
        err = str(exc)
        if "429" in err:
            raise HTTPException(
                status_code=503,
                detail="Gemini API quota exceeded. Please try again later."
            )
        if "503" in err:
            raise HTTPException(
                status_code=503,
                detail="Gemini API overloaded. Please try again shortly."
            )
        raise HTTPException(status_code=500, detail=f"Agent error: {err}")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# POST /sql  — validated SQL execution with caching
# ---------------------------------------------------------------------------

@app.post("/sql", response_model=SqlResponse, tags=["Data"])
async def run_sql(request: SqlRequest):
    """
    Execute a validated SELECT query.
    Results are cached for 120 seconds for identical queries.
    """
    err = validate_sql(request.sql)
    if err:
        raise HTTPException(status_code=400, detail=f"SQL validation failed: {err}")

    # Check cache
    cache_key = f"sql_endpoint:{request.sql.strip()}"
    cached = query_cache.get(cache_key)
    if cached is not _NOT_FOUND:
        return SqlResponse(**cached, cached=True)

    result = await asyncio.to_thread(execute_sql_query, request.sql)

    if not result.success:
        raise HTTPException(status_code=400, detail=f"SQL failed: {result.error}")

    payload = {
        "sql": result.sql,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
    }
    query_cache.set(cache_key, payload)
    return SqlResponse(**payload, cached=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
