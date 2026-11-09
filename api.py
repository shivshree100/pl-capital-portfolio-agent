"""
api.py — REST API for the PL Capital Portfolio Analytics Agent.

Wraps the existing agent and tools in a FastAPI application.
All business logic lives in src/; this module only handles HTTP routing,
request validation, and response serialisation.

Endpoints
---------
GET  /                          Health check
GET  /portfolios                List all portfolios
GET  /portfolios/{name}         Get a single portfolio by name
GET  /portfolios/{name}/exposure  Sector exposure for a portfolio
GET  /portfolios/{name}/performance  Performance history
GET  /portfolios/{name}/risk    Risk metrics history
GET  /securities                List all securities
GET  /sectors                   List all sectors with holdings summary
GET  /holdings?portfolio={name} Holdings, optionally filtered by portfolio
POST /ask                       Natural-language question → agent answer
POST /sql                       Execute a validated SQL SELECT query directly

Running
-------
    # Development server:
    python api.py

    # Or with uvicorn directly:
    uvicorn api:app --reload --port 8000

    # Interactive docs (auto-generated):
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import managed_connection, get_db_path, check_connection
from src.tools.sql_tool import execute_sql_query, validate_sql
from src.tools.exposure_tool import calculate_sector_exposure


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    database: str
    agent_available: bool


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about portfolio data.",
        examples=["How many portfolios do we have?"],
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    tool_used: Optional[str] = None


class SqlRequest(BaseModel):
    sql: str = Field(
        ...,
        min_length=6,
        description="A valid SQLite SELECT statement.",
        examples=["SELECT COUNT(*) FROM portfolios"],
    )


class SqlResponse(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool


class ExposureResponse(BaseModel):
    portfolio_name: str
    portfolio_found: bool
    total_equity_market_value: float
    equity_holding_count: int
    excluded_bond_count: int
    sector_exposures: dict[str, float]
    sector_market_values: dict[str, float]
    warnings: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Application lifespan — initialise shared resources once at startup
# ---------------------------------------------------------------------------

_agent = None  # shared PortfolioAgent instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the agent on startup; clean up on shutdown."""
    global _agent

    # Verify the database is accessible
    if not check_connection():
        print(
            "[WARN] portfolio_database.db not found or not accessible.\n"
            "       Run 'python init_db.py' first.\n"
            "       The /ask endpoint will be unavailable."
        )

    # Try to initialise the agent (requires GEMINI_API_KEY)
    try:
        from src.agent.portfolio_agent import PortfolioAgent
        _agent = PortfolioAgent()
        print("[INFO] PortfolioAgent initialised successfully.")
    except RuntimeError as exc:
        _agent = None
        print(f"[WARN] PortfolioAgent not available: {exc}")
        print("       The /ask endpoint will return 503.")

    yield  # application runs here

    _agent = None
    print("[INFO] API shutdown.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PL Capital — Portfolio Analytics API",
    description=(
        "REST API for the PL Capital Portfolio Analytics Agent. "
        "Provides endpoints for portfolio data, sector exposure, "
        "performance metrics, and natural-language Q&A via Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper: run a parameterised SQL query and return rows as dicts
# ---------------------------------------------------------------------------

def _run_sql(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a raw (trusted internal) SQL query. NOT for user input."""
    with managed_connection() as conn:
        cursor = conn.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _portfolio_id_for(name: str) -> int:
    """Return portfolio_id for a given portfolio name, or raise 404."""
    rows = _run_sql(
        "SELECT portfolio_id FROM portfolios WHERE portfolio_name = ?", (name,)
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{name}' not found. "
                   "Use GET /portfolios to see available names."
        )
    return rows[0]["portfolio_id"]


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check. Returns database connectivity and agent availability."""
    db_ok = check_connection()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="connected" if db_ok else f"unavailable — run python init_db.py",
        agent_available=_agent is not None,
    )


# ---------------------------------------------------------------------------
# GET /portfolios
# ---------------------------------------------------------------------------

@app.get("/portfolios", tags=["Portfolios"])
def list_portfolios():
    """Return all portfolios with their key attributes."""
    return _run_sql(
        "SELECT portfolio_id, portfolio_name, total_aum, target_risk_level, "
        "strategy_type, benchmark_index, status, creation_date "
        "FROM portfolios ORDER BY portfolio_name"
    )


@app.get("/portfolios/{portfolio_name}", tags=["Portfolios"])
def get_portfolio(portfolio_name: str):
    """Return a single portfolio by exact name."""
    rows = _run_sql(
        "SELECT * FROM portfolios WHERE portfolio_name = ?", (portfolio_name,)
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_name}' not found."
        )
    return rows[0]


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/exposure
# ---------------------------------------------------------------------------

@app.get(
    "/portfolios/{portfolio_name}/exposure",
    response_model=ExposureResponse,
    tags=["Portfolios"],
)
def get_sector_exposure(portfolio_name: str):
    """
    Calculate equity sector exposure for a portfolio.
    Uses the deterministic exposure calculator (no Gemini API call).
    """
    result = calculate_sector_exposure(portfolio_name)

    if not result.portfolio_found:
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio '{portfolio_name}' not found."
        )

    return ExposureResponse(
        portfolio_name=result.portfolio_name,
        portfolio_found=result.portfolio_found,
        total_equity_market_value=round(result.total_equity_market_value, 2),
        equity_holding_count=result.equity_holding_count,
        excluded_bond_count=result.excluded_bond_count,
        sector_exposures={k: round(v, 4) for k, v in result.sector_exposures.items()},
        sector_market_values={k: round(v, 2) for k, v in result.sector_market_values.items()},
        warnings=result.warnings,
    )


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/performance
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/performance", tags=["Portfolios"])
def get_performance(portfolio_name: str):
    """Return performance history for a portfolio, most recent first."""
    pid = _portfolio_id_for(portfolio_name)
    rows = _run_sql(
        "SELECT performance_date, nav, total_return_1m, total_return_3m, "
        "total_return_6m, total_return_1y, volatility, sharpe_ratio, max_drawdown "
        "FROM portfolio_performance WHERE portfolio_id = ? "
        "ORDER BY performance_date DESC",
        (pid,),
    )
    return {"portfolio_name": portfolio_name, "records": rows}


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/risk
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/risk", tags=["Portfolios"])
def get_risk(portfolio_name: str):
    """Return risk metrics history for a portfolio, most recent first."""
    pid = _portfolio_id_for(portfolio_name)
    rows = _run_sql(
        "SELECT calculation_date, var_95, var_99, cvar_95, beta, "
        "correlation_sp500, tracking_error, information_ratio, sortino_ratio "
        "FROM risk_metrics WHERE portfolio_id = ? "
        "ORDER BY calculation_date DESC",
        (pid,),
    )
    return {"portfolio_name": portfolio_name, "records": rows}


# ---------------------------------------------------------------------------
# GET /portfolios/{name}/holdings
# ---------------------------------------------------------------------------

@app.get("/portfolios/{portfolio_name}/holdings", tags=["Portfolios"])
def get_holdings(portfolio_name: str):
    """Return all holdings for a portfolio with security details."""
    pid = _portfolio_id_for(portfolio_name)
    rows = _run_sql(
        "SELECT h.holding_id, s.symbol, s.company_name, s.asset_type, "
        "sec.sector_name, h.quantity, h.purchase_price, h.cost_basis, "
        "s.current_price, ROUND(h.quantity * s.current_price, 2) AS market_value "
        "FROM holdings h "
        "JOIN securities s ON h.security_id = s.security_id "
        "LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
        "WHERE h.portfolio_id = ? "
        "ORDER BY market_value DESC",
        (pid,),
    )
    return {"portfolio_name": portfolio_name, "holdings": rows}


# ---------------------------------------------------------------------------
# GET /securities
# ---------------------------------------------------------------------------

@app.get("/securities", tags=["Securities"])
def list_securities(
    asset_type: Optional[str] = Query(
        default=None,
        description="Filter by asset type: 'Stock' or 'Bond'",
    )
):
    """Return all securities. Optionally filter by asset_type."""
    if asset_type:
        if asset_type not in ("Stock", "Bond"):
            raise HTTPException(
                status_code=400,
                detail="asset_type must be 'Stock' or 'Bond'."
            )
        rows = _run_sql(
            "SELECT s.security_id, s.symbol, s.company_name, s.asset_type, "
            "sec.sector_name, s.current_price, s.market_cap, s.exchange, s.country "
            "FROM securities s "
            "LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
            "WHERE s.asset_type = ? ORDER BY s.symbol",
            (asset_type,),
        )
    else:
        rows = _run_sql(
            "SELECT s.security_id, s.symbol, s.company_name, s.asset_type, "
            "sec.sector_name, s.current_price, s.market_cap, s.exchange, s.country "
            "FROM securities s "
            "LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
            "ORDER BY s.symbol"
        )
    return rows


# ---------------------------------------------------------------------------
# GET /sectors
# ---------------------------------------------------------------------------

@app.get("/sectors", tags=["Sectors"])
def list_sectors():
    """Return all sectors with a count of securities in each."""
    return _run_sql(
        "SELECT sec.sector_id, sec.sector_name, sec.industry_group, "
        "COUNT(s.security_id) AS security_count "
        "FROM sectors sec "
        "LEFT JOIN securities s ON sec.sector_id = s.sector_id "
        "GROUP BY sec.sector_id ORDER BY sec.sector_name"
    )


# ---------------------------------------------------------------------------
# POST /ask  — natural language Q&A via Gemini agent
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["Agent"])
def ask_question(request: AskRequest):
    """
    Answer a natural-language question about portfolio data using the AI agent.

    The agent routes the question to either the SQL tool or the sector
    exposure calculator.  Requires GEMINI_API_KEY to be configured.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI agent is not available. "
                "Ensure GEMINI_API_KEY is set in the .env file and restart the API."
            ),
        )

    try:
        # answer() handles routing, tool execution, and synthesis.
        # We get the answer directly — no second API call needed.
        answer = _agent.answer(request.question)

        # Infer tool used from the question type (lightweight keyword check)
        from src.agent.portfolio_agent import _looks_like_portfolio_question
        tool_hint = "run_sql_query" if _looks_like_portfolio_question(request.question) else None

        return AskResponse(
            question=request.question,
            answer=answer,
            tool_used=tool_hint,
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
                detail="Gemini API is temporarily overloaded. Please try again."
            )
        raise HTTPException(status_code=500, detail=f"Agent error: {err}")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {exc}"
        )


# ---------------------------------------------------------------------------
# POST /sql  — validated SQL execution (no Gemini)
# ---------------------------------------------------------------------------

@app.post("/sql", response_model=SqlResponse, tags=["Data"])
def run_sql(request: SqlRequest):
    """
    Execute a validated SELECT query directly against the database.

    SQL is validated before execution (SELECT-only, no DDL/DML).
    Results are capped at 500 rows.
    """
    validation_error = validate_sql(request.sql)
    if validation_error:
        raise HTTPException(
            status_code=400,
            detail=f"SQL validation failed: {validation_error}"
        )

    result = execute_sql_query(request.sql)

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=f"SQL execution failed: {result.error}"
        )

    return SqlResponse(
        sql=result.sql,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
