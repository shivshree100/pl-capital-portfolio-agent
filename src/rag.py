"""
src/rag.py — Schema-aware context retrieval for the Portfolio Analytics Agent.

What this is
------------
A lightweight Retrieval-Augmented Generation (RAG) layer that pre-fetches
relevant database facts and injects them into the Gemini prompt context.

This is NOT vector-based RAG (no embeddings, no ChromaDB, no FAISS).
It is deterministic, schema-aware retrieval:
  - Detect which entities the user is asking about (portfolio names, sectors,
    securities, time ranges).
  - Fetch a small, relevant fact sheet from the database.
  - Inject that fact sheet as grounded context into the Gemini prompt.

Why this approach
-----------------
Vector-based RAG would require embedding models and a vector store — heavy
dependencies with no benefit here because our "documents" are structured
relational rows, not unstructured text.  The right retrieval for structured
data is SQL, which we already do.  What we add here is a pre-fetch step that
gives Gemini verified facts before it generates SQL, so it:
  1. Uses exact portfolio names (no hallucinated names).
  2. Knows which sectors, securities, and risk levels exist.
  3. Gets data ranges (e.g. date range of price history) so it generates
     correct date filters.

Public API
----------
build_context(question: str) -> str
    Returns a markdown-formatted context string ready to prepend to the
    Gemini prompt.  Returns an empty string if no relevant context found.
"""

import re
from pathlib import Path
from typing import Optional

from src.database.connection import managed_connection
from src.cache import query_cache


# ---------------------------------------------------------------------------
# Entity detection helpers
# ---------------------------------------------------------------------------

def _detect_portfolio_names(question: str, all_names: list[str]) -> list[str]:
    """Return portfolio names mentioned in the question (case-insensitive)."""
    q_lower = question.lower()
    found = []
    for name in all_names:
        # Match full name or any meaningful word from it (>= 4 chars)
        if name.lower() in q_lower:
            found.append(name)
            continue
        words = [w for w in name.lower().split() if len(w) >= 4]
        if any(w in q_lower for w in words):
            found.append(name)
    return list(dict.fromkeys(found))  # preserve order, deduplicate


def _detect_sectors(question: str, all_sectors: list[str]) -> list[str]:
    """Return sector names mentioned in the question."""
    q_lower = question.lower()
    return [s for s in all_sectors if s.lower() in q_lower]


def _detect_securities(question: str, symbols: list[str]) -> list[str]:
    """Return security symbols mentioned in the question (uppercase match)."""
    q_upper = question.upper()
    return [s for s in symbols if s in q_upper]


# ---------------------------------------------------------------------------
# Database lookups (cached)
# ---------------------------------------------------------------------------

def _all_portfolio_names() -> list[str]:
    from src.cache import _NOT_FOUND
    cached = query_cache.get("rag:portfolio_names")
    if cached is not _NOT_FOUND:
        return cached
    with managed_connection() as conn:
        rows = conn.execute(
            "SELECT portfolio_name FROM portfolios ORDER BY portfolio_name"
        ).fetchall()
    names = [r[0] for r in rows]
    query_cache.set("rag:portfolio_names", names)
    return names


def _all_sectors() -> list[str]:
    from src.cache import _NOT_FOUND
    cached = query_cache.get("rag:sectors")
    if cached is not _NOT_FOUND:
        return cached
    with managed_connection() as conn:
        rows = conn.execute(
            "SELECT sector_name FROM sectors ORDER BY sector_name"
        ).fetchall()
    sectors = [r[0] for r in rows]
    query_cache.set("rag:sectors", sectors)
    return sectors


def _all_symbols() -> list[str]:
    from src.cache import _NOT_FOUND
    cached = query_cache.get("rag:symbols")
    if cached is not _NOT_FOUND:
        return cached
    with managed_connection() as conn:
        rows = conn.execute(
            "SELECT symbol FROM securities ORDER BY symbol"
        ).fetchall()
    symbols = [r[0] for r in rows]
    query_cache.set("rag:symbols", symbols)
    return symbols


def _portfolio_facts(portfolio_name: str) -> dict:
    """Fetch key facts about a portfolio for context injection."""
    from src.cache import _NOT_FOUND
    key = f"rag:portfolio:{portfolio_name}"
    cached = query_cache.get(key)
    if cached is not _NOT_FOUND:
        return cached

    with managed_connection() as conn:
        # Portfolio metadata
        meta = conn.execute(
            "SELECT portfolio_id, portfolio_name, total_aum, target_risk_level, "
            "strategy_type, status, benchmark_index, creation_date "
            "FROM portfolios WHERE portfolio_name = ?",
            (portfolio_name,)
        ).fetchone()

        if not meta:
            return {}

        pid = meta["portfolio_id"]

        # Holdings summary
        holdings = conn.execute(
            "SELECT COUNT(*) as count, "
            "SUM(h.quantity * s.current_price) as total_market_value "
            "FROM holdings h JOIN securities s ON h.security_id = s.security_id "
            "WHERE h.portfolio_id = ?",
            (pid,)
        ).fetchone()

        # Sector breakdown
        sectors_raw = conn.execute(
            "SELECT sec.sector_name, COUNT(*) as holdings_count "
            "FROM holdings h "
            "JOIN securities s ON h.security_id = s.security_id "
            "LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
            "WHERE h.portfolio_id = ? AND s.asset_type = 'Stock' "
            "GROUP BY sec.sector_name ORDER BY holdings_count DESC",
            (pid,)
        ).fetchall()

        # Latest performance
        perf = conn.execute(
            "SELECT performance_date, nav, total_return_1y, sharpe_ratio, volatility "
            "FROM portfolio_performance WHERE portfolio_id = ? "
            "ORDER BY performance_date DESC LIMIT 1",
            (pid,)
        ).fetchone()

    facts = {
        "name": meta["portfolio_name"],
        "aum": meta["total_aum"],
        "risk_level": meta["target_risk_level"],
        "strategy": meta["strategy_type"],
        "status": meta["status"],
        "benchmark": meta["benchmark_index"],
        "created": meta["creation_date"],
        "holdings_count": holdings["count"] if holdings else 0,
        "market_value": round(holdings["total_market_value"] or 0, 2),
        "sectors": [dict(r) for r in sectors_raw],
        "latest_nav": dict(perf) if perf else {},
    }
    query_cache.set(key, facts)
    return facts


def _sector_facts(sector_name: str) -> dict:
    """Fetch key facts about a sector for context injection."""
    from src.cache import _NOT_FOUND
    key = f"rag:sector:{sector_name}"
    cached = query_cache.get(key)
    if cached is not _NOT_FOUND:
        return cached

    with managed_connection() as conn:
        securities = conn.execute(
            "SELECT s.symbol, s.company_name, s.current_price, s.market_cap "
            "FROM securities s "
            "JOIN sectors sec ON s.sector_id = sec.sector_id "
            "WHERE sec.sector_name = ? AND s.asset_type = 'Stock' "
            "ORDER BY s.market_cap DESC",
            (sector_name,)
        ).fetchall()

    facts = {
        "sector": sector_name,
        "securities": [dict(r) for r in securities],
        "count": len(securities),
    }
    query_cache.set(key, facts)
    return facts


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(question: str, db_path: Optional[Path] = None) -> str:
    """
    Build a grounded context string for the given question.

    Detects entities (portfolios, sectors, securities) in the question,
    fetches verified facts from the database, and returns a compact
    markdown block to prepend to the Gemini prompt.

    Returns an empty string if no relevant entities are detected.
    """
    try:
        all_portfolios = _all_portfolio_names()
        all_sectors = _all_sectors()
        all_symbols = _all_symbols()
    except Exception:
        return ""  # DB not available — degrade gracefully

    mentioned_portfolios = _detect_portfolio_names(question, all_portfolios)
    mentioned_sectors = _detect_sectors(question, all_sectors)
    mentioned_symbols = _detect_securities(question, all_symbols)

    # Nothing detected — return empty (saves a Gemini token)
    if not mentioned_portfolios and not mentioned_sectors and not mentioned_symbols:
        return ""

    lines = ["## Verified Context from Database\n"]

    # Portfolio facts
    for name in mentioned_portfolios[:3]:  # cap at 3 to limit context size
        facts = _portfolio_facts(name)
        if not facts:
            continue
        lines.append(f"### Portfolio: {facts['name']}")
        lines.append(f"- AUM: ${facts['aum']:,.0f}")
        lines.append(f"- Risk level: {facts['risk_level']}")
        lines.append(f"- Strategy: {facts['strategy']}")
        lines.append(f"- Status: {facts['status']}")
        lines.append(f"- Holdings: {facts['holdings_count']}")
        lines.append(f"- Total equity market value: ${facts['market_value']:,.2f}")
        if facts["sectors"]:
            sec_list = ", ".join(s["sector_name"] for s in facts["sectors"][:5])
            lines.append(f"- Equity sectors: {sec_list}")
        if facts["latest_nav"]:
            nav = facts["latest_nav"]
            lines.append(
                f"- Latest NAV: {nav.get('nav', 'N/A')} "
                f"(as of {nav.get('performance_date', 'N/A')})"
            )
        lines.append("")

    # Sector facts
    for name in mentioned_sectors[:2]:
        facts = _sector_facts(name)
        if not facts:
            continue
        lines.append(f"### Sector: {facts['sector']}")
        lines.append(f"- {facts['count']} equity securities")
        syms = [s["symbol"] for s in facts["securities"][:8]]
        lines.append(f"- Symbols: {', '.join(syms)}")
        lines.append("")

    # Symbol mentions (just confirm they exist and their sector)
    if mentioned_symbols:
        lines.append("### Securities mentioned")
        try:
            with managed_connection() as conn:
                for sym in mentioned_symbols[:5]:
                    row = conn.execute(
                        "SELECT s.symbol, s.company_name, sec.sector_name, s.current_price "
                        "FROM securities s "
                        "LEFT JOIN sectors sec ON s.sector_id = sec.sector_id "
                        "WHERE s.symbol = ?",
                        (sym,)
                    ).fetchone()
                    if row:
                        lines.append(
                            f"- {row['symbol']}: {row['company_name']} "
                            f"({row['sector_name'] or 'No sector'}), "
                            f"price ${row['current_price']}"
                        )
        except Exception:
            pass
        lines.append("")

    if len(lines) <= 1:
        return ""  # nothing useful was added

    return "\n".join(lines)
