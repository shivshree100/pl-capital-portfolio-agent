"""
PL Capital — Portfolio Analytics Dashboard

A Streamlit dashboard for portfolio analytics.  Reuses the existing
backend (database, exposure calculator, AI agent) without duplicating
business logic.

Run:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import altair as alt
import streamlit as st

from src.database.connection import managed_connection, get_db_path
from src.tools.exposure_tool import calculate_sector_exposure
from src.tools.exposure_tool import format_exposure_result

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PL Capital — Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

NAVY = "#16213E"
OFF_WHITE = "#FAFAFA"
LIGHT_GREY = "#E0E0E0"
MUTED_GREEN = "#2D6A4F"
MUTED_RED = "#C0392B"
TEXT_DARK = "#1A1A2E"
TEXT_MUTED = "#666666"

# ---------------------------------------------------------------------------
# Helpers — direct database queries (no business logic duplication)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _query_db(sql: str) -> pd.DataFrame:
    """Execute a SELECT query and return a DataFrame. Cached for 60s."""
    with managed_connection() as conn:
        df = pd.read_sql_query(sql, conn)
    return df


@st.cache_data(ttl=60)
def _get_portfolio_names() -> list[str]:
    """Return all portfolio names from the database."""
    df = _query_db("SELECT portfolio_name FROM portfolios ORDER BY portfolio_name")
    return df["portfolio_name"].tolist()


def _format_currency(val) -> str:
    """Format a number as USD currency with commas."""
    if val is None or pd.isna(val):
        return "—"
    return f"${val:,.0f}"


def _format_pct(val) -> str:
    """Format a number as a percentage."""
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.2f}%"


def _kpi_card(label: str, value: str, delta: str = ""):
    """Render a KPI card using Streamlit metrics."""
    st.metric(label=label, value=value, delta=delta if delta else None)


# ---------------------------------------------------------------------------
# Section 1: Overview
# ---------------------------------------------------------------------------

def render_overview():
    st.markdown("## Portfolio Overview")
    st.markdown("---")

    # KPI row
    portfolios_df = _query_db("SELECT * FROM portfolios")
    securities_df = _query_db("SELECT * FROM securities")
    holdings_df = _query_db("SELECT * FROM holdings")

    total_aum = portfolios_df["total_aum"].sum()
    active_count = len(portfolios_df[portfolios_df["status"] == "Active"])
    passive_count = len(portfolios_df[portfolios_df["status"] == "Passive"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _kpi_card("Total Portfolios", str(len(portfolios_df)))
    with col2:
        _kpi_card("Total AUM", _format_currency(total_aum))
    with col3:
        _kpi_card("Active Funds", str(active_count))
    with col4:
        _kpi_card("Securities", str(len(securities_df)))

    st.markdown("")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### AUM by Portfolio")
        aum_df = portfolios_df[["portfolio_name", "total_aum"]].sort_values("total_aum", ascending=True)
        chart = alt.Chart(aum_df).mark_bar(color=NAVY, cornerRadius=2).encode(
            x=alt.X("total_aum:Q", title="AUM (USD)", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("portfolio_name:N", title="", sort="-x"),
            tooltip=[
                alt.Tooltip("portfolio_name:N", title="Portfolio"),
                alt.Tooltip("total_aum:Q", title="AUM", format="$,.0f"),
            ],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.markdown("#### Risk Distribution")
        risk_df = portfolios_df.groupby("target_risk_level").size().reset_index(name="count")
        risk_df.columns = ["Risk Level", "Count"]
        color_map = {"High": MUTED_RED, "Medium": "#E67E22", "Low": MUTED_GREEN}
        chart2 = alt.Chart(risk_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Risk Level:N", scale=alt.Scale(
                domain=["High", "Medium", "Low"],
                range=[MUTED_RED, "#E67E22", MUTED_GREEN],
            )),
            tooltip=["Risk Level:N", "Count:Q"],
        ).properties(height=300)
        st.altair_chart(chart2, use_container_width=True)

    # Status summary
    st.markdown("#### Fund Status")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Active:** {active_count} funds")
    with col_b:
        st.markdown(f"**Passive:** {passive_count} funds")

    # Portfolio summary table
    st.markdown("")
    st.markdown("#### Portfolio Summary")
    summary = portfolios_df[[
        "portfolio_name", "total_aum", "target_risk_level",
        "strategy_type", "status", "benchmark_index"
    ]].copy()
    summary.columns = ["Portfolio", "AUM", "Risk Level", "Strategy", "Status", "Benchmark"]
    summary["AUM"] = summary["AUM"].apply(_format_currency)
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 2: AI Assistant
# ---------------------------------------------------------------------------

def render_assistant():
    st.markdown("## AI Portfolio Assistant")
    st.markdown("---")
    st.markdown("Ask questions about portfolios, holdings, sector exposure, performance, or risk data.")

    # Initialise chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Initialise agent (lazy — only when needed)
    if "portfolio_agent" not in st.session_state:
        try:
            from src.agent.portfolio_agent import PortfolioAgent
            st.session_state.portfolio_agent = PortfolioAgent()
        except RuntimeError as e:
            st.session_state.portfolio_agent = None
            st.warning(f"Agent not available: {e}")

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about portfolio data..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            if st.session_state.portfolio_agent is None:
                st.error("Agent is not available. Check GEMINI_API_KEY in .env file.")
                response_text = "Agent not available."
            else:
                with st.spinner("Thinking..."):
                    try:
                        response_text = st.session_state.portfolio_agent.answer(prompt)
                    except RuntimeError as e:
                        err_msg = str(e)
                        if "429" in err_msg:
                            response_text = "API quota exceeded. Please try again later."
                        elif "503" in err_msg:
                            response_text = "The model is temporarily overloaded. Please try again in a moment."
                        else:
                            response_text = f"Error: {err_msg}"
                        st.error(response_text)
                    except Exception as e:
                        response_text = f"Unexpected error: {e}"
                        st.error(response_text)

                st.markdown(response_text)

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})


# ---------------------------------------------------------------------------
# Section 3: Sector Exposure
# ---------------------------------------------------------------------------

def render_exposure():
    st.markdown("## Sector Exposure Analysis")
    st.markdown("---")

    portfolio_names = _get_portfolio_names()
    selected = st.selectbox("Select Portfolio", portfolio_names)

    if not selected:
        st.info("Select a portfolio to view sector exposure.")
        return

    # Use the existing deterministic calculator
    result = calculate_sector_exposure(selected)

    if not result.portfolio_found:
        st.warning(f"Portfolio '{selected}' not found.")
        return

    if not result.is_valid():
        st.warning("No equity holdings found for this portfolio.")
        if result.warnings:
            for w in result.warnings:
                st.text(f"⚠ {w}")
        return

    # KPI row
    col1, col2, col3 = st.columns(3)
    with col1:
        _kpi_card("Total Equity Value", _format_currency(result.total_equity_market_value))
    with col2:
        _kpi_card("Equity Holdings", str(result.equity_holding_count))
    with col3:
        _kpi_card("Bond Holdings Excluded", str(result.excluded_bond_count))

    st.markdown("")

    # Exposure chart
    st.markdown("#### Sector Exposure Breakdown")
    chart_data = pd.DataFrame([
        {"Sector": k, "Exposure %": v, "Market Value": mv}
        for k, v in result.sector_exposures.items()
        for k2, mv in [(k, result.sector_market_values.get(k, 0))]
    ])
    chart_data = chart_data.sort_values("Exposure %", ascending=True)

    bar = alt.Chart(chart_data).mark_bar(color=NAVY, cornerRadius=2).encode(
        x=alt.X("Exposure %:Q", title="Exposure (%)", axis=alt.Axis(format=".1f")),
        y=alt.Y("Sector:N", title="", sort="-x"),
        tooltip=[
            alt.Tooltip("Sector:N", title="Sector"),
            alt.Tooltip("Exposure %:Q", title="Exposure", format=".2f"),
            alt.Tooltip("Market Value:Q", title="Market Value", format="$,.2f"),
        ],
    ).properties(height=350)
    st.altair_chart(bar, use_container_width=True)

    # Detail table
    st.markdown("#### Sector Details")
    detail_df = pd.DataFrame([
        {
            "Sector": sector,
            "Market Value": _format_currency(result.sector_market_values.get(sector, 0)),
            "Exposure %": _format_pct(result.sector_exposures.get(sector, 0)),
        }
        for sector in result.sector_exposures
    ])
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Warnings
    if result.warnings:
        st.markdown("#### Data Quality Warnings")
        for w in result.warnings:
            st.text(f"⚠ {w}")


# ---------------------------------------------------------------------------
# Section 4: Performance & Risk
# ---------------------------------------------------------------------------

def render_performance_risk():
    st.markdown("## Performance & Risk Metrics")
    st.markdown("---")

    portfolio_names = _get_portfolio_names()
    selected = st.selectbox("Select Portfolio", portfolio_names, key="perf_portfolio")

    if not selected:
        return

    # Get portfolio_id
    portfolio_df = _query_db(
        f"SELECT portfolio_id FROM portfolios WHERE portfolio_name = '{selected}'"
    )
    if portfolio_df.empty:
        st.warning("Portfolio not found.")
        return

    pid = int(portfolio_df.iloc[0]["portfolio_id"])

    # Performance data
    perf_df = _query_db(
        f"SELECT * FROM portfolio_performance WHERE portfolio_id = {pid} ORDER BY performance_date"
    )
    risk_df = _query_db(
        f"SELECT * FROM risk_metrics WHERE portfolio_id = {pid} ORDER BY calculation_date"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### NAV Trend")
        if perf_df.empty:
            st.info("No performance data available.")
        else:
            nav_chart = alt.Chart(perf_df).mark_line(color=NAVY, strokeWidth=2).encode(
                x=alt.X("performance_date:T", title="Date"),
                y=alt.Y("nav:Q", title="NAV", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("performance_date:T", title="Date"),
                    alt.Tooltip("nav:Q", title="NAV", format=".2f"),
                ],
            ).properties(height=300)
            st.altair_chart(nav_chart, use_container_width=True)

        # Return metrics
        if not perf_df.empty:
            latest = perf_df.iloc[-1]
            st.markdown("##### Latest Returns")
            ret_cols = st.columns(4)
            metrics = [
                ("1M", latest.get("total_return_1m")),
                ("3M", latest.get("total_return_3m")),
                ("6M", latest.get("total_return_6m")),
                ("1Y", latest.get("total_return_1y")),
            ]
            for col, (label, val) in zip(ret_cols, metrics):
                with col:
                    _kpi_card(f"Return {label}", _format_pct(val))

    with col_right:
        st.markdown("#### Risk Metrics")
        if risk_df.empty:
            st.info("No risk data available.")
        else:
            latest_risk = risk_df.iloc[-1]
            risk_items = [
                ("VaR 95%", latest_risk.get("var_95"), ".2f"),
                ("VaR 99%", latest_risk.get("var_99"), ".2f"),
                ("CVaR 95%", latest_risk.get("cvar_95"), ".2f"),
                ("Beta", latest_risk.get("beta"), ".2f"),
                ("Sharpe Ratio", latest_risk.get("sharpe_ratio"), ".2f"),
                ("Sortino Ratio", latest_risk.get("sortino_ratio"), ".2f"),
                ("Max Drawdown", latest_risk.get("max_drawdown"), ".2f"),
                ("Tracking Error", latest_risk.get("tracking_error"), ".2f"),
                ("Info Ratio", latest_risk.get("information_ratio"), ".2f"),
            ]
            risk_table = pd.DataFrame(risk_items, columns=["Metric", "Value"])
            risk_table["Value"] = risk_table["Value"].apply(
                lambda v: f"{v:.2f}" if v is not None and not pd.isna(v) else "—"
            )
            st.dataframe(risk_table, use_container_width=True, hide_index=True)

    # Full performance table
    if not perf_df.empty:
        st.markdown("")
        st.markdown("#### Performance History")
        display_perf = perf_df.drop(columns=["performance_id", "portfolio_id"])
        display_perf.columns = [
            "Date", "NAV", "Return 1M", "Return 3M", "Return 6M",
            "Return 1Y", "Volatility", "Sharpe", "Max Drawdown"
        ]
        st.dataframe(display_perf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 5: Data Explorer
# ---------------------------------------------------------------------------

def render_explorer():
    st.markdown("## Data Explorer")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Portfolios", "Holdings", "Securities"])

    with tab1:
        st.markdown("#### All Portfolios")
        df = _query_db("SELECT * FROM portfolios ORDER BY portfolio_name")
        df.columns = [
            "ID", "Portfolio Name", "Created", "Risk Level", "AUM",
            "Strategy", "Benchmark", "Status"
        ]
        df["AUM"] = df["AUM"].apply(_format_currency)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Holdings")
        portfolio_names = _get_portfolio_names()
        selected = st.selectbox(
            "Filter by Portfolio", ["All"] + portfolio_names, key="holdings_filter"
        )

        if selected == "All":
            sql = """
                SELECT h.holding_id, p.portfolio_name, s.symbol, s.company_name,
                       h.quantity, h.purchase_price, h.cost_basis, s.current_price
                FROM holdings h
                JOIN portfolios p ON h.portfolio_id = p.portfolio_id
                JOIN securities s ON h.security_id = s.security_id
                ORDER BY p.portfolio_name, s.symbol
            """
        else:
            sql = f"""
                SELECT h.holding_id, p.portfolio_name, s.symbol, s.company_name,
                       h.quantity, h.purchase_price, h.cost_basis, s.current_price
                FROM holdings h
                JOIN portfolios p ON h.portfolio_id = p.portfolio_id
                JOIN securities s ON h.security_id = s.security_id
                WHERE p.portfolio_name = '{selected}'
                ORDER BY s.symbol
            """
        df = _query_db(sql)
        df.columns = [
            "ID", "Portfolio", "Symbol", "Company",
            "Quantity", "Purchase Price", "Cost Basis", "Current Price"
        ]
        for col in ["Purchase Price", "Cost Basis", "Current Price"]:
            df[col] = df[col].apply(lambda v: _format_currency(v) if v is not None and not pd.isna(v) else "—")
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Securities")
        df = _query_db("""
            SELECT s.security_id, s.symbol, s.company_name, s.asset_type,
                   sec.sector_name, s.current_price, s.market_cap, s.exchange
            FROM securities s
            LEFT JOIN sectors sec ON s.sector_id = sec.sector_id
            ORDER BY s.symbol
        """)
        df.columns = ["ID", "Symbol", "Company", "Type", "Sector", "Price", "Market Cap", "Exchange"]
        for col in ["Price", "Market Cap"]:
            df[col] = df[col].apply(lambda v: _format_currency(v) if v is not None and not pd.isna(v) else "—")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main app — sidebar navigation
# ---------------------------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.markdown("## PL Capital")
    st.sidebar.markdown("Portfolio Analytics")
    st.sidebar.markdown("---")

    section = st.sidebar.radio(
        "Navigation",
        ["Overview", "AI Assistant", "Sector Exposure", "Performance & Risk", "Data Explorer"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<span style='color:{TEXT_MUTED}; font-size:0.8rem'>"
        f"Database: {get_db_path().name}<br>"
        f"Model: Gemini</span>",
        unsafe_allow_html=True,
    )

    # Route to section
    if section == "Overview":
        render_overview()
    elif section == "AI Assistant":
        render_assistant()
    elif section == "Sector Exposure":
        render_exposure()
    elif section == "Performance & Risk":
        render_performance_risk()
    elif section == "Data Explorer":
        render_explorer()


if __name__ == "__main__":
    main()
