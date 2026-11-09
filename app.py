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
# Colour palette — PL Capital inspired (deep navy, clean white, professional)
# ---------------------------------------------------------------------------

NAVY = "#0A2540"          # Primary brand colour — deep navy
BLUE = "#1B4F8C"          # Secondary blue for charts/accents
LIGHT_BG = "#F4F6F9"      # Light background for sections
BORDER = "#E2E8F0"        # Subtle borders/dividers
MUTED_GREEN = "#1A7F5C"   # Positive values
MUTED_RED = "#C0392B"     # Negative values
TEXT_DARK = "#1A1A2E"     # Primary text
TEXT_MUTED = "#64748B"    # Secondary text
WHITE = "#FFFFFF"

# Inject custom CSS for a branded fintech look.
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #F7F9FF 0%, #EFF4FF 52%, #F0FAF6 100%);
    }

    /* Clean header typography */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #122A8A;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .stMarkdown h2 {
        font-size: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
    .stMarkdown h4 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0A2540;
    }

    .welcome-hero {
        margin: 0.5rem 0 2rem;
        padding: 2rem 2.25rem;
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 22px;
        background: linear-gradient(115deg, #102B91 0%, #1767B1 58%, #0A9871 100%);
        box-shadow: 0 16px 38px rgba(18, 42, 138, 0.16);
        color: #FFFFFF;
    }
    .welcome-hero h1 {
        margin: 0 0 0.45rem;
        color: #FFFFFF !important;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        font-weight: 750;
        letter-spacing: -0.035em;
    }
    .welcome-hero h2 {
        margin: 0 0 1rem;
        padding: 0;
        border: 0;
        color: #E8F0FF !important;
        font-size: clamp(1.15rem, 2vw, 1.45rem);
        font-weight: 600;
    }
    .welcome-hero p {
        max-width: 900px;
        margin: 0.55rem 0 0;
        color: rgba(255, 255, 255, 0.92);
        font-size: 1rem;
        line-height: 1.7;
    }

    /* Metric cards — flat, bordered, no shadow */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #FFFFFF 0%, #F2F6FF 100%);
        border: 1px solid #DCE5F5;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 6px 18px rgba(18, 42, 138, 0.06);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #122A8A;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* PL Capital gradient sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(165deg, #07165E 0%, #1236A0 58%, #087A67 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] {
        background: #FFFFFF;
        padding: 8px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(3, 12, 55, 0.22);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #FFFFFF;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #CBD5E1;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
        color: #DCE8FF;
    }
    [data-testid="stSidebar"] hr {
        border-top-color: rgba(255, 255, 255, 0.24);
    }

    /* Data tables — clean borders, readable */
    .stDataFrame {
        border: 1px solid #DCE5F5;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(18, 42, 138, 0.05);
    }

    /* Chat input — professional */
    .stChatInput textarea {
        border: 1px solid #E2E8F0;
        border-radius: 4px;
    }

    /* Selectbox — clean */
    .stSelectbox > div > div {
        border: 1px solid #D4DEF2;
        border-radius: 10px;
    }

    /* Floating Lila button */
    .st-key-lila_assistant_toggle {
        position: fixed !important;
        right: 24px !important;
        bottom: 24px !important;
        z-index: 9999 !important;
    }

    .st-key-lila_assistant_toggle button {
        border-radius: 999px !important;
        background: linear-gradient(110deg, #122A8A 0%, #1767B1 58%, #0A9871 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.8rem 1.25rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 10px 24px rgba(10, 37, 64, 0.22) !important;
        min-width: 230px !important;
    }

    .st-key-lila_chat_panel {
        position: fixed !important;
        bottom: 88px !important;
        right: 24px !important;
        width: min(380px, calc(100vw - 32px)) !important;
        z-index: 9998 !important;
        background: #FFFFFF !important;
        border: 1px solid rgba(10, 37, 64, 0.12) !important;
        border-radius: 16px !important;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18) !important;
        padding: 16px !important;
    }

    .st-key-lila_chat_panel [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    .st-key-lila_chat_panel [data-testid="stChatMessage"] {
        padding: 0.6rem 0.75rem;
        border-radius: 12px;
        background: #F4F6F9;
    }

    .st-key-lila_chat_panel .stTextInput input {
        border-radius: 10px;
    }

    /* Divider lines */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown(
        """
        <section class="welcome-hero">
            <h1>Welcome to PL Capital</h1>
            <h2>Your portfolios. Your insights. One place.</h2>
            <p>Explore your investment portfolios, monitor assets under management,
            analyse sector exposures, and track portfolio performance and risk —
            all through one unified analytics platform.</p>
            <p>Explore your portfolio dashboard or ask Lila, your AI assistant,
            to help you uncover meaningful insights.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
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


def render_lila_chat_panel():
    """Render Lila's compact, in-page chat popup."""
    with st.container(key="lila_chat_panel"):
        title_col, close_col = st.columns([5, 1])
        with title_col:
            st.markdown("### Lila AI Assistant")
        with close_col:
            if st.button("×", key="lila_assistant_close", help="Close Lila"):
                st.session_state.show_lila_assistant = False
                st.rerun()

        st.caption("Ask about portfolios, holdings, exposure, performance, or risk.")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form("lila_chat_form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                placeholder="Ask Lila about your portfolio...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and question.strip():
            st.session_state.chat_messages.append(
                {"role": "user", "content": question.strip()}
            )

            if "portfolio_agent" not in st.session_state:
                try:
                    from src.agent.portfolio_agent import PortfolioAgent
                    st.session_state.portfolio_agent = PortfolioAgent()
                except RuntimeError as exc:
                    st.session_state.portfolio_agent = None
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": f"Agent is not available: {exc}"}
                    )

            agent = st.session_state.portfolio_agent
            if agent is not None:
                try:
                    with st.spinner("Lila is thinking..."):
                        answer = agent.answer(question.strip())
                except RuntimeError as exc:
                    answer = f"Error: {exc}"
                except Exception as exc:
                    answer = f"Unexpected error: {exc}"

                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer}
                )
            st.rerun()


# ---------------------------------------------------------------------------
# Main app — sidebar navigation
# ---------------------------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.image(
        str(PROJECT_ROOT / "assets" / "pl-capital-logo.png"),
        width="stretch",
    )
    st.sidebar.markdown(
        "<p style='color:#DCE8FF; font-size:0.85rem; margin-top:0.6rem;'>Portfolio Analytics</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)

    section = st.sidebar.radio(
        "Navigation",
        ["Overview", "Sector Exposure", "Performance & Risk", "Data Explorer"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='color:#DCE8FF; font-size:0.75rem;'>"
        f"Database: {get_db_path().name}<br>"
        f"Powered by Gemini AI</p>",
        unsafe_allow_html=True,
    )

    if "show_lila_assistant" not in st.session_state:
        st.session_state.show_lila_assistant = False

    # Route to section
    if section == "Overview":
        render_overview()
    elif section == "Sector Exposure":
        render_exposure()
    elif section == "Performance & Risk":
        render_performance_risk()
    elif section == "Data Explorer":
        render_explorer()

    if st.button(
        "Hi, I am Lila, your AI Assistant",
        key="lila_assistant_toggle",
        help="Open AI Assistant",
    ):
        st.session_state.show_lila_assistant = not st.session_state.show_lila_assistant

    if st.session_state.show_lila_assistant:
        render_lila_chat_panel()


if __name__ == "__main__":
    main()
