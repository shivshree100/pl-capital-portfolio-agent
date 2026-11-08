"""
Tests for src/agent/portfolio_agent.py

Test categories
---------------
1. Live integration tests — require GEMINI_API_KEY and portfolio_database.db.
   These call the real Gemini API and verify end-to-end behaviour.
   Skipped automatically when the API key or database is missing.

2. Unit tests — mock the Gemini client to test internal routing logic
   without API calls. These always run.

Running
-------
    # All tests (unit + integration):
    .venv\\Scripts\\python.exe -m pytest tests/test_agent.py -v

    # Unit tests only (no API key needed):
    .venv\\Scripts\\python.exe -m pytest tests/test_agent.py -v -k "Unit"

    # Integration tests only:
    .venv\\Scripts\\python.exe -m pytest tests/test_agent.py -v -k "Integration"
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.portfolio_agent import PortfolioAgent
from src.tools.sql_tool import SqlResult
from src.tools.exposure_tool import ExposureResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_agent():
    """
    Return a real PortfolioAgent instance using the production API key and DB.
    Skips the whole module if the key or database is missing.
    """
    db = PROJECT_ROOT / "portfolio_database.db"
    if not db.exists():
        pytest.skip("portfolio_database.db not found. Run 'python init_db.py'.")

    from dotenv import load_dotenv
    import os
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        pytest.skip("GEMINI_API_KEY not set. Add it to .env.")

    try:
        return PortfolioAgent(db_path=db)
    except RuntimeError as exc:
        pytest.skip(f"Could not initialise agent: {exc}")


# ---------------------------------------------------------------------------
# Helper: build a fake dispatch response with a function_call part
# ---------------------------------------------------------------------------

def _make_fc_response(tool_name: str, args: dict):
    """Build a minimal mock GenerateContentResponse with a function_call."""
    fc = MagicMock()
    fc.name = tool_name
    fc.args = args

    part = MagicMock()
    part.function_call = fc
    part.text = None

    content = MagicMock()
    content.parts = [part]
    content.role = "model"

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    # response.text raises AttributeError (like the real SDK does on fc-only response)
    type(response).text = property(lambda self: None)
    return response


def _make_text_response(text: str):
    """Build a minimal mock GenerateContentResponse with a text part."""
    part = MagicMock()
    part.function_call = None
    part.text = text

    content = MagicMock()
    content.parts = [part]
    content.role = "model"

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    response.text = text
    return response


# ---------------------------------------------------------------------------
# Unit tests — mock _call_with_fallback, no real API calls
# ---------------------------------------------------------------------------

class TestUnitRouting:
    """
    Unit tests that verify internal routing logic by mocking the Gemini client.
    These do not call the real API and do not require GEMINI_API_KEY.
    """

    def _make_agent(self):
        """Create an agent with a fake API key (no real calls made)."""
        with patch("src.agent.portfolio_agent.genai.Client"):
            return PortfolioAgent(api_key="fake-key-for-unit-tests")

    def test_sql_tool_call_is_executed(self):
        """When Gemini returns run_sql_query, the SQL tool is dispatched."""
        agent = self._make_agent()

        dispatch_resp = _make_fc_response(
            "run_sql_query",
            {"sql": "SELECT COUNT(*) FROM portfolios"}
        )
        synth_resp = _make_text_response("There are 13 portfolios.")

        # Patch _call_gemini_dispatch (returns dispatch response) and
        # _call_gemini_synthesise (returns synth response) separately,
        # so the real synthesis method (which builds Content objects from
        # mock data) is never called.
        with patch.object(agent, "_call_gemini_dispatch", return_value=dispatch_resp), \
             patch.object(agent, "_call_gemini_synthesise", return_value=synth_resp), \
             patch.object(agent, "_dispatch_tool", return_value="SQL result: 13") as mock_dispatch:
            result = agent.answer("How many portfolios?")

        mock_dispatch.assert_called_once_with("run_sql_query", {"sql": "SELECT COUNT(*) FROM portfolios"})
        assert "13" in result or "portfolios" in result.lower()

    def test_exposure_tool_call_is_executed(self):
        """When Gemini returns calculate_exposure, the exposure tool is dispatched."""
        agent = self._make_agent()

        dispatch_resp = _make_fc_response(
            "calculate_exposure",
            {"portfolio_name": "Tech Innovation Fund"}
        )
        synth_resp = _make_text_response("Tech Innovation Fund is 100% Technology.")

        with patch.object(agent, "_call_gemini_dispatch", return_value=dispatch_resp), \
             patch.object(agent, "_call_gemini_synthesise", return_value=synth_resp), \
             patch.object(agent, "_dispatch_tool", return_value="Exposure: 100% Technology") as mock_dispatch:
            result = agent.answer("What are the sector exposures for Tech Innovation Fund?")

        mock_dispatch.assert_called_once_with(
            "calculate_exposure", {"portfolio_name": "Tech Innovation Fund"}
        )
        assert "Technology" in result or "100" in result

    def test_no_tool_call_returns_text_without_dispatch(self):
        """
        When Gemini returns plain text (no tool call), answer() returns
        the text directly without executing any tool or calling synthesis.
        """
        agent = self._make_agent()

        text_resp = _make_text_response(
            "I can only help with portfolio analytics questions."
        )

        with patch.object(agent, "_call_gemini_dispatch", return_value=text_resp), \
             patch.object(agent, "_dispatch_tool") as mock_dispatch, \
             patch.object(agent, "_call_gemini_synthesise") as mock_synth:
            result = agent.answer("Tell me about the F1 race winner")

        # No tool must be dispatched
        mock_dispatch.assert_not_called()
        # No synthesis call must be made
        mock_synth.assert_not_called()
        # Result is the plain text from Gemini
        assert "portfolio" in result.lower()

    def test_off_topic_does_not_execute_either_tool(self):
        """
        Off-topic questions that return plain text must not execute
        run_sql_query or calculate_exposure.
        """
        agent = self._make_agent()

        text_resp = _make_text_response("I can only help with portfolio data.")

        with patch.object(agent, "_call_gemini_dispatch", return_value=text_resp), \
             patch.object(agent, "_dispatch_tool") as mock_dispatch:
            agent.answer("What is the weather today?")

        mock_dispatch.assert_not_called()

    def test_malformed_response_produces_clear_error(self):
        """
        Empty or malformed Gemini responses produce a clear error message,
        not a misleading successful response.
        """
        agent = self._make_agent()

        # Response with no candidates
        malformed = MagicMock()
        malformed.candidates = []
        malformed.text = None

        # Mock both dispatch methods to return the malformed response
        # (the retry with mode=ANY also gets the same malformed response)
        with patch.object(agent, "_call_gemini_dispatch", return_value=malformed), \
             patch.object(agent, "_call_gemini_dispatch_forced", return_value=malformed), \
             patch.object(agent, "_dispatch_tool") as mock_dispatch:
            result = agent.answer("How many portfolios?")

        # No tool dispatched for malformed response
        mock_dispatch.assert_not_called()
        # Clear error message, not a misleading success
        assert "could not process" in result.lower() or "unexpected" in result.lower()

    def test_retry_forces_tool_call_on_portfolio_question(self):
        """
        When mode=AUTO returns no tool call for a portfolio question,
        answer() retries with mode=ANY to force a tool call.
        """
        agent = self._make_agent()

        # First dispatch (AUTO) returns text (no tool call)
        text_resp = _make_text_response("I cannot help with that.")
        # Retry dispatch (ANY) returns a function call
        forced_resp = _make_fc_response(
            "run_sql_query",
            {"sql": "SELECT COUNT(*) FROM portfolios"}
        )
        synth_resp = _make_text_response("There are 13 portfolios.")

        with patch.object(agent, "_call_gemini_dispatch", return_value=text_resp), \
             patch.object(agent, "_call_gemini_dispatch_forced", return_value=forced_resp), \
             patch.object(agent, "_call_gemini_synthesise", return_value=synth_resp), \
             patch.object(agent, "_dispatch_tool", return_value="SQL result: 13") as mock_dispatch:
            result = agent.answer("Show me the top 5 holdings by cost basis")

        # The tool WAS dispatched (because the retry forced it)
        mock_dispatch.assert_called_once_with("run_sql_query", {"sql": "SELECT COUNT(*) FROM portfolios"})

    def test_no_retry_for_off_topic_question(self):
        """
        Off-topic questions (no portfolio keywords) do NOT trigger a retry.
        The text response is returned directly.
        """
        agent = self._make_agent()

        text_resp = _make_text_response("I can only help with portfolio data.")

        with patch.object(agent, "_call_gemini_dispatch", return_value=text_resp), \
             patch.object(agent, "_call_gemini_dispatch_forced") as mock_forced, \
             patch.object(agent, "_dispatch_tool") as mock_dispatch:
            result = agent.answer("What is the weather today?")

        # No retry (off-topic, no keywords)
        mock_forced.assert_not_called()
        # No tool dispatched
        mock_dispatch.assert_not_called()
        assert "portfolio" in result.lower()

    def test_dispatch_retries_for_portfolio_question(self):
        """
        dispatch() retries with mode=ANY when no tool call is returned
        for a question with portfolio keywords.
        """
        agent = self._make_agent()

        text_resp = _make_text_response("I cannot help.")
        forced_resp = _make_fc_response(
            "run_sql_query",
            {"sql": "SELECT COUNT(*) FROM portfolios"}
        )

        with patch.object(agent, "_call_gemini_dispatch", return_value=text_resp), \
             patch.object(agent, "_call_gemini_dispatch_forced", return_value=forced_resp):
            tool_name, tool_args, result = agent.dispatch("How many portfolios?")

        assert tool_name == "run_sql_query"
        assert isinstance(result, SqlResult)
        assert result.success

    def test_extract_tool_call_returns_none_on_no_fc(self):
        """
        _extract_tool_call returns None when no function_call part is
        present (model returned plain text for an off-topic question).
        """
        agent = self._make_agent()
        text_only_response = _make_text_response("Hello!")
        # Remove function_call from part
        text_only_response.candidates[0].content.parts[0].function_call = None

        result = agent._extract_tool_call(text_only_response)
        assert result is None

    def test_empty_question_returns_prompt(self):
        """Empty or whitespace input returns a prompt without calling Gemini."""
        agent = self._make_agent()
        with patch.object(agent, "_call_with_fallback") as mock_call:
            result = agent.answer("   ")
        mock_call.assert_not_called()
        assert "question" in result.lower() or "enter" in result.lower()

    def test_sql_validation_blocks_mutating_sql(self):
        """
        Even if Gemini somehow generates a DROP/DELETE statement, the SQL
        validation layer in _dispatch_tool blocks it before any DB access.
        We call _dispatch_tool directly to isolate the validation test.
        """
        agent = self._make_agent()

        tool_result = agent._dispatch_tool(
            "run_sql_query", {"sql": "DROP TABLE portfolios"}
        )

        assert "validation failed" in tool_result.lower() or "forbidden" in tool_result.lower(), (
            f"Expected validation error in tool result. Got:\n{tool_result}"
        )


# ---------------------------------------------------------------------------
# Unit tests for dispatch() — mocked, no API calls
# ---------------------------------------------------------------------------

class TestUnitDispatch:
    """
    Unit tests for PortfolioAgent.dispatch() — the evaluation entry point
    that returns structured tool results without calling the synthesis API.
    All tests mock _call_gemini_dispatch so no real API calls are made.
    """

    def _make_agent(self):
        """Create an agent with a fake API key (no real calls made)."""
        with patch("src.agent.portfolio_agent.genai.Client"):
            return PortfolioAgent(api_key="fake-key-for-unit-tests")

    def test_dispatch_sql_returns_sql_result(self):
        """dispatch() returns a SqlResult when Gemini calls run_sql_query."""
        agent = self._make_agent()
        dispatch_resp = _make_fc_response(
            "run_sql_query",
            {"sql": "SELECT COUNT(*) FROM portfolios"}
        )
        with patch.object(agent, "_call_gemini_dispatch", return_value=dispatch_resp), \
             patch.object(agent, "_call_gemini_synthesise") as mock_synth:
            tool_name, tool_args, result = agent.dispatch("How many portfolios?")

        assert tool_name == "run_sql_query"
        assert "sql" in tool_args
        assert isinstance(result, SqlResult)
        assert result.success
        assert result.row_count == 1
        # Synthesis must NOT be called by dispatch()
        mock_synth.assert_not_called()

    def test_dispatch_exposure_returns_exposure_result(self):
        """dispatch() returns an ExposureResult when Gemini calls calculate_exposure."""
        agent = self._make_agent()
        dispatch_resp = _make_fc_response(
            "calculate_exposure",
            {"portfolio_name": "Tech Innovation Fund"}
        )
        with patch.object(agent, "_call_gemini_dispatch", return_value=dispatch_resp), \
             patch.object(agent, "_call_gemini_synthesise") as mock_synth:
            tool_name, tool_args, result = agent.dispatch(
                "What are the sector exposures for the Tech Innovation Fund?"
            )

        assert tool_name == "calculate_exposure"
        assert tool_args["portfolio_name"] == "Tech Innovation Fund"
        assert isinstance(result, ExposureResult)
        assert result.portfolio_found
        # Synthesis must NOT be called by dispatch()
        mock_synth.assert_not_called()

    def test_dispatch_no_tool_call_returns_decline_sentinel(self):
        """
        dispatch() returns ('decline_question', {}, None) when Gemini
        returns no function call (off-topic question or plain-text response).
        """
        agent = self._make_agent()
        text_resp = _make_text_response("I can only help with portfolio questions.")

        with patch.object(agent, "_call_gemini_dispatch", return_value=text_resp), \
             patch.object(agent, "_call_gemini_synthesise") as mock_synth:
            tool_name, tool_args, result = agent.dispatch("Tell me about F1 racing")

        assert tool_name == "decline_question"
        assert tool_args == {}
        assert result is None
        # Synthesis must NOT be called by dispatch()
        mock_synth.assert_not_called()

    def test_dispatch_sql_validation_failure_returns_failed_result(self):
        """dispatch() returns a failed SqlResult when SQL is invalid, not a string."""
        agent = self._make_agent()
        dispatch_resp = _make_fc_response(
            "run_sql_query",
            {"sql": "DROP TABLE portfolios"}
        )
        with patch.object(agent, "_call_gemini_dispatch", return_value=dispatch_resp):
            tool_name, tool_args, result = agent.dispatch("Drop all portfolios")

        assert tool_name == "run_sql_query"
        assert isinstance(result, SqlResult)
        assert not result.success
        assert result.error  # error message is populated


# ---------------------------------------------------------------------------
# Integration tests — real API + real database
# ---------------------------------------------------------------------------

class TestIntegrationLive:
    """
    End-to-end tests using the real Gemini API and database.
    Each test sends a question to the agent and checks the response.
    Automatically skipped when API key or database is unavailable.
    """

    def test_portfolio_count_question(self, live_agent):
        """
        Q1 ground truth: 'How many portfolios do we have in total?'
        Expected: answer contains '13'.
        """
        result = live_agent.answer("How many portfolios do we have in total?")
        assert "13" in result, (
            f"Expected '13' in answer. Got:\n{result}"
        )

    def test_sector_exposure_tech_fund(self, live_agent):
        """
        Q9 ground truth: sector exposure for Tech Innovation Fund.
        Expected: answer mentions Technology and ~100%.
        """
        result = live_agent.answer(
            "What are the sector exposures for the Tech Innovation Fund?"
        )
        assert "Technology" in result, (
            f"Expected 'Technology' in answer. Got:\n{result}"
        )
        assert "100" in result, (
            f"Expected '100' (percent) in answer. Got:\n{result}"
        )

    def test_f1_question_is_declined(self, live_agent):
        """
        The F1 regression test.
        An unrelated question must NOT trigger SQL execution.
        The response must contain the out-of-scope message.
        """
        result = live_agent.answer("Tell me more about the F1 car race and who won it?")

        # Must NOT look like a SQL result (no table headers, row counts, etc.)
        sql_indicators = ["rows returned", "SELECT", "portfolio_name", "COUNT(*)"]
        for indicator in sql_indicators:
            assert indicator.lower() not in result.lower(), (
                f"Response looks like SQL output (found '{indicator}'). Got:\n{result}"
            )

        # Must contain our scope refusal
        assert "portfolio" in result.lower(), (
            f"Expected scope-refusal message. Got:\n{result}"
        )

    def test_unknown_portfolio_name(self, live_agent):
        """
        Asking for sector exposure of a portfolio that doesn't exist
        must return a clear not-found message, not an empty result.
        """
        result = live_agent.answer(
            "What are the sector exposures for the Nonexistent Fake Fund?"
        )
        # The exposure tool returns a warning for unknown portfolios
        not_found_indicators = [
            "not found", "does not exist", "couldn't find",
            "no portfolio", "unable to find", "Nonexistent"
        ]
        found = any(ind.lower() in result.lower() for ind in not_found_indicators)
        assert found, (
            f"Expected a 'not found' message. Got:\n{result}"
        )

    def test_active_portfolios_question(self, live_agent):
        """
        Q2 ground truth: 'What are the names of all active portfolios?'
        Expected: answer mentions at least one known active portfolio.
        """
        result = live_agent.answer("What are the names of all active portfolios?")
        known_active = ["Growth Equity Fund", "Tech Innovation Fund", "Balanced Portfolio"]
        found = any(name in result for name in known_active)
        assert found, (
            f"Expected at least one active portfolio name. Got:\n{result}"
        )
