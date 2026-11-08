"""
Tests for src/evaluation/evaluator.py

All tests are fully mocked — no live Gemini API calls, no database access
required (comparison tests build SqlResult / ExposureResult objects directly).

Running
-------
    .venv\\Scripts\\python.exe -m pytest tests/test_evaluator.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.sql_tool import SqlResult
from src.tools.exposure_tool import ExposureResult
from src.evaluation.evaluator import (
    compare_sql_results,
    compare_exposure_results,
    evaluate_question,
    run_evaluation,
    load_ground_truth,
    print_report,
    _compare_values,
    _compare_lists,
    _has_order_by,
    _expected_tool_for,
    QuestionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sql_result(rows: list[dict], columns: list[str], success: bool = True,
                error: str = "") -> SqlResult:
    """Build a SqlResult for testing."""
    return SqlResult(
        sql="test",
        success=success,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        error=error,
    )


def _exposure_result(
    portfolio_name: str = "Test Fund",
    portfolio_found: bool = True,
    sector_exposures: dict | None = None,
    sector_market_values: dict | None = None,
) -> ExposureResult:
    """Build an ExposureResult for testing."""
    return ExposureResult(
        portfolio_name=portfolio_name,
        portfolio_found=portfolio_found,
        sector_exposures=sector_exposures or {},
        sector_market_values=sector_market_values or {},
    )


def _mock_agent(tool_name: str, tool_args: dict, tool_result):
    """Build a mock agent whose dispatch() returns the given values."""
    agent = MagicMock()
    agent.dispatch.return_value = (tool_name, tool_args, tool_result)
    return agent


# ---------------------------------------------------------------------------
# Tests: _compare_values
# ---------------------------------------------------------------------------

class TestCompareValues:

    def test_integers_match(self):
        ok, _ = _compare_values(13, 13)
        assert ok

    def test_integers_mismatch(self):
        ok, msg = _compare_values(13, 14)
        assert not ok
        assert "13" in msg and "14" in msg

    def test_floats_within_tolerance(self):
        ok, _ = _compare_values(100.0, 100.005)
        assert ok  # within 0.01 tolerance

    def test_floats_outside_tolerance(self):
        ok, _ = _compare_values(100.0, 100.02)
        assert not ok

    def test_int_and_float_match(self):
        ok, _ = _compare_values(100, 100.0)
        assert ok

    def test_strings_match(self):
        ok, _ = _compare_values("Technology", "Technology")
        assert ok

    def test_strings_mismatch(self):
        ok, _ = _compare_values("Technology", "Healthcare")
        assert not ok

    def test_string_case_sensitive(self):
        ok, _ = _compare_values("Active", "active")
        assert not ok  # SQLite is case-sensitive


# ---------------------------------------------------------------------------
# Tests: _compare_lists
# ---------------------------------------------------------------------------

class TestCompareLists:

    def test_numeric_lists_match_unordered(self):
        ok, _ = _compare_lists([3, 1, 2], [2, 3, 1])
        assert ok

    def test_numeric_lists_mismatch(self):
        ok, _ = _compare_lists([1, 2, 3], [1, 2, 4])
        assert not ok

    def test_different_lengths(self):
        ok, msg = _compare_lists([1, 2], [1, 2, 3])
        assert not ok
        assert "length" in msg

    def test_string_lists_match_unordered(self):
        ok, _ = _compare_lists(["AAPL", "MSFT"], ["MSFT", "AAPL"])
        assert ok


# ---------------------------------------------------------------------------
# Tests: compare_sql_results
# ---------------------------------------------------------------------------

class TestCompareSqlResults:

    def test_single_value_match(self):
        exp = _sql_result([{"COUNT(*)": 13}], ["COUNT(*)"])
        act = _sql_result([{"total": 13}], ["total"])
        ok, _ = compare_sql_results(exp, act, "single_value")
        assert ok

    def test_single_value_mismatch(self):
        exp = _sql_result([{"COUNT(*)": 13}], ["COUNT(*)"])
        act = _sql_result([{"total": 14}], ["total"])
        ok, msg = compare_sql_results(exp, act, "single_value")
        assert not ok

    def test_list_match_unordered(self):
        exp = _sql_result(
            [{"portfolio_name": "Growth Equity Fund"},
             {"portfolio_name": "Tech Innovation Fund"}],
            ["portfolio_name"],
        )
        act = _sql_result(
            [{"portfolio_name": "Tech Innovation Fund"},
             {"portfolio_name": "Growth Equity Fund"}],
            ["portfolio_name"],
        )
        ok, _ = compare_sql_results(exp, act, "list")
        assert ok

    def test_list_mismatch(self):
        exp = _sql_result(
            [{"portfolio_name": "A"}, {"portfolio_name": "B"}],
            ["portfolio_name"],
        )
        act = _sql_result(
            [{"portfolio_name": "A"}, {"portfolio_name": "C"}],
            ["portfolio_name"],
        )
        ok, _ = compare_sql_results(exp, act, "list")
        assert not ok

    def test_table_ordered_match(self):
        exp = _sql_result(
            [{"symbol": "AAPL", "cost": 100}, {"symbol": "MSFT", "cost": 90}],
            ["symbol", "cost"],
        )
        act = _sql_result(
            [{"symbol": "AAPL", "cost": 100}, {"symbol": "MSFT", "cost": 90}],
            ["symbol", "cost"],
        )
        ok, _ = compare_sql_results(exp, act, "table", ordered=True)
        assert ok

    def test_table_ordered_mismatch(self):
        exp = _sql_result(
            [{"symbol": "AAPL", "cost": 100}, {"symbol": "MSFT", "cost": 90}],
            ["symbol", "cost"],
        )
        act = _sql_result(
            [{"symbol": "MSFT", "cost": 90}, {"symbol": "AAPL", "cost": 100}],
            ["symbol", "cost"],
        )
        ok, _ = compare_sql_results(exp, act, "table", ordered=True)
        assert not ok  # order differs

    def test_table_unordered_match(self):
        exp = _sql_result(
            [{"symbol": "AAPL", "sector": "Tech"},
             {"symbol": "MSFT", "sector": "Tech"}],
            ["symbol", "sector"],
        )
        act = _sql_result(
            [{"symbol": "MSFT", "sector": "Tech"},
             {"symbol": "AAPL", "sector": "Tech"}],
            ["symbol", "sector"],
        )
        ok, _ = compare_sql_results(exp, act, "table", ordered=False)
        assert ok

    def test_table_row_count_mismatch(self):
        exp = _sql_result([{"a": 1}, {"a": 2}], ["a"])
        act = _sql_result([{"a": 1}], ["a"])
        ok, msg = compare_sql_results(exp, act, "table")
        assert not ok
        assert "rows" in msg

    def test_table_extra_columns_ignored(self):
        """Actual result with extra columns should still match if expected
        columns are present with correct values."""
        exp = _sql_result(
            [{"symbol": "AAPL", "company_name": "Apple Inc."},
             {"symbol": "MSFT", "company_name": "Microsoft Corporation"}],
            ["symbol", "company_name"],
        )
        act = _sql_result(
            [{"symbol": "AAPL", "company_name": "Apple Inc.",
              "asset_type": "Stock", "sector_id": 1, "current_price": 185.5},
             {"symbol": "MSFT", "company_name": "Microsoft Corporation",
              "asset_type": "Stock", "sector_id": 1, "current_price": 415.26}],
            ["symbol", "company_name", "asset_type", "sector_id", "current_price"],
        )
        ok, _ = compare_sql_results(exp, act, "table", ordered=False)
        assert ok

    def test_table_column_reorder_matches(self):
        """Actual result with columns in a different order should match
        because projection is by column name, not position."""
        exp = _sql_result(
            [{"symbol": "AAPL", "cost": 100}, {"symbol": "MSFT", "cost": 90}],
            ["symbol", "cost"],
        )
        act = _sql_result(
            [{"cost": 100, "symbol": "AAPL"}, {"cost": 90, "symbol": "MSFT"}],
            ["cost", "symbol"],
        )
        ok, _ = compare_sql_results(exp, act, "table", ordered=True)
        assert ok  # names are used, not positions

    def test_table_missing_expected_column_fails(self):
        """If the actual result has fewer columns than expected, it fails."""
        exp = _sql_result(
            [{"symbol": "AAPL", "company_name": "Apple"}],
            ["symbol", "company_name"],
        )
        act = _sql_result(
            [{"symbol": "AAPL"}],
            ["symbol"],
        )
        ok, msg = compare_sql_results(exp, act, "table")
        assert not ok
        assert "columns" in msg.lower()

    def test_table_different_column_aliases_match_positionally(self):
        """When the agent uses different column aliases than the ground
        truth, the evaluator falls back to positional comparison."""
        exp = _sql_result(
            [{"sector_name": "Technology", "avg_price": 719.25}],
            ["sector_name", "avg_price"],
        )
        act = _sql_result(
            [{"sector": "Technology", "average_price": 719.25}],
            ["sector", "average_price"],
        )
        ok, _ = compare_sql_results(exp, act, "table", ordered=False)
        assert ok

    def test_failed_expected_sql(self):
        exp = SqlResult(sql="test", success=False, error="syntax error")
        act = _sql_result([{"x": 1}], ["x"])
        ok, msg = compare_sql_results(exp, act, "single_value")
        assert not ok
        assert "expected SQL failed" in msg

    def test_failed_actual_sql(self):
        exp = _sql_result([{"x": 1}], ["x"])
        act = SqlResult(sql="test", success=False, error="syntax error")
        ok, msg = compare_sql_results(exp, act, "single_value")
        assert not ok
        assert "actual SQL failed" in msg


# ---------------------------------------------------------------------------
# Tests: compare_exposure_results
# ---------------------------------------------------------------------------

class TestCompareExposureResults:

    def test_match(self):
        exp = _exposure_result(
            sector_exposures={"Technology": 100.0},
            sector_market_values={"Technology": 1565292.50},
        )
        act = _exposure_result(
            sector_exposures={"Technology": 100.0},
            sector_market_values={"Technology": 1565292.50},
        )
        ok, _ = compare_exposure_results(exp, act)
        assert ok

    def test_sector_mismatch(self):
        exp = _exposure_result(
            sector_exposures={"Technology": 100.0},
            sector_market_values={"Technology": 1000.0},
        )
        act = _exposure_result(
            sector_exposures={"Energy": 100.0},
            sector_market_values={"Energy": 1000.0},
        )
        ok, msg = compare_exposure_results(exp, act)
        assert not ok
        assert "sectors" in msg.lower()

    def test_percentage_mismatch(self):
        exp = _exposure_result(
            sector_exposures={"Technology": 47.20, "Energy": 30.68},
            sector_market_values={"Technology": 346888, "Energy": 225470},
        )
        act = _exposure_result(
            sector_exposures={"Technology": 50.0, "Energy": 30.68},
            sector_market_values={"Technology": 346888, "Energy": 225470},
        )
        ok, msg = compare_exposure_results(exp, act)
        assert not ok
        assert "Technology" in msg

    def test_portfolio_not_found_actual(self):
        exp = _exposure_result(portfolio_found=True,
                                sector_exposures={"Tech": 100.0},
                                sector_market_values={"Tech": 1000.0})
        act = _exposure_result(portfolio_found=False)
        ok, msg = compare_exposure_results(exp, act)
        assert not ok
        assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# Tests: evaluate_question (mocked agent)
# ---------------------------------------------------------------------------

class TestEvaluateQuestion:
    """
    Tests that mock the agent's dispatch() method to test the full
    evaluate_question flow without any API calls.
    """

    # --- Text2SQL question ---

    def test_text2sql_correct_routing_and_data(self):
        """Agent picks run_sql_query and returns matching data → PASS."""
        question = {
            "id": 1,
            "type": "text2sql",
            "difficulty": "easy",
            "question": "How many portfolios do we have?",
            "ground_truth": {
                "sql_query": "SELECT COUNT(*) FROM portfolios;",
                "expected_result_type": "single_value",
                "explanation": "Simple COUNT",
            },
        }
        # Mock agent returns correct tool + matching result
        agent = _mock_agent(
            "run_sql_query",
            {"sql": "SELECT COUNT(*) FROM portfolios"},
            _sql_result([{"COUNT(*)": 13}], ["COUNT(*)"]),
        )
        result = evaluate_question(question, agent)

        assert result.routing_correct
        assert result.data_correct
        assert result.passed

    def test_text2sql_wrong_tool(self):
        """Agent picks calculate_exposure for a SQL question → routing FAIL."""
        question = {
            "id": 1,
            "type": "text2sql",
            "difficulty": "easy",
            "question": "How many portfolios?",
            "ground_truth": {
                "sql_query": "SELECT COUNT(*) FROM portfolios;",
                "expected_result_type": "single_value",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "calculate_exposure",
            {"portfolio_name": "Tech Innovation Fund"},
            _exposure_result(),
        )
        result = evaluate_question(question, agent)

        assert not result.routing_correct
        assert not result.passed
        assert "wrong tool" in result.comparison_details

    def test_text2sql_correct_routing_wrong_data(self):
        """Agent picks run_sql_query but returns wrong count → data FAIL."""
        question = {
            "id": 1,
            "type": "text2sql",
            "difficulty": "easy",
            "question": "How many portfolios?",
            "ground_truth": {
                "sql_query": "SELECT COUNT(*) FROM portfolios;",
                "expected_result_type": "single_value",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "run_sql_query",
            {"sql": "SELECT COUNT(*) FROM portfolios"},
            _sql_result([{"total": 99}], ["total"]),  # wrong value
        )
        result = evaluate_question(question, agent)

        assert result.routing_correct
        assert not result.data_correct
        assert not result.passed

    def test_text2sql_decline_question(self):
        """Agent declines a portfolio question → routing FAIL."""
        question = {
            "id": 1,
            "type": "text2sql",
            "difficulty": "easy",
            "question": "How many portfolios?",
            "ground_truth": {
                "sql_query": "SELECT COUNT(*) FROM portfolios;",
                "expected_result_type": "single_value",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "decline_question",
            {"reason": "I cannot help with that."},
            None,
        )
        result = evaluate_question(question, agent)

        assert not result.routing_correct
        assert not result.passed

    def test_text2sql_api_error(self):
        """Agent dispatch raises RuntimeError → error recorded, FAIL."""
        question = {
            "id": 1,
            "type": "text2sql",
            "difficulty": "easy",
            "question": "How many portfolios?",
            "ground_truth": {
                "sql_query": "SELECT COUNT(*) FROM portfolios;",
                "expected_result_type": "single_value",
                "explanation": "",
            },
        }
        agent = MagicMock()
        agent.dispatch.side_effect = RuntimeError("API 503")

        result = evaluate_question(question, agent)

        assert not result.passed
        assert "503" in result.error

    # --- Exposure calculator question ---

    def test_exposure_correct_routing_and_data(self):
        """Agent picks calculate_exposure and returns matching result → PASS."""
        question = {
            "id": 9,
            "type": "exposure_calculator",
            "difficulty": "medium",
            "question": "What are the sector exposures for the Tech Innovation Fund?",
            "ground_truth": {
                "tool_name": "exposure_calculator",
                "parameters": {"portfolio_name": "Tech Innovation Fund"},
                "expected_result_type": "sector_exposure_breakdown",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "calculate_exposure",
            {"portfolio_name": "Tech Innovation Fund"},
            _exposure_result(
                portfolio_found=True,
                sector_exposures={"Technology": 100.0},
                sector_market_values={"Technology": 1565292.50},
            ),
        )
        result = evaluate_question(question, agent)

        assert result.routing_correct
        assert result.data_correct
        assert result.passed

    def test_exposure_wrong_tool(self):
        """Agent picks run_sql_query for an exposure question → routing FAIL."""
        question = {
            "id": 9,
            "type": "exposure_calculator",
            "difficulty": "medium",
            "question": "Sector exposures for Tech Innovation Fund?",
            "ground_truth": {
                "tool_name": "exposure_calculator",
                "parameters": {"portfolio_name": "Tech Innovation Fund"},
                "expected_result_type": "sector_exposure_breakdown",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "run_sql_query",
            {"sql": "SELECT * FROM sectors"},
            _sql_result([{"x": 1}], ["x"]),
        )
        result = evaluate_question(question, agent)

        assert not result.routing_correct
        assert not result.passed

    def test_exposure_wrong_portfolio_data(self):
        """Agent picks calculate_exposure but returns different sectors → data FAIL."""
        question = {
            "id": 10,
            "type": "exposure_calculator",
            "difficulty": "medium",
            "question": "Sector exposures for International Equity Fund?",
            "ground_truth": {
                "tool_name": "exposure_calculator",
                "parameters": {"portfolio_name": "International Equity Fund"},
                "expected_result_type": "sector_exposure_breakdown",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "calculate_exposure",
            {"portfolio_name": "International Equity Fund"},
            _exposure_result(
                portfolio_found=True,
                sector_exposures={"Technology": 50.0, "Energy": 50.0},
                sector_market_values={"Technology": 1000, "Energy": 1000},
            ),
        )
        result = evaluate_question(question, agent)

        assert result.routing_correct
        assert not result.data_correct
        assert not result.passed

    def test_exposure_returns_none(self):
        """Agent returns None for an exposure question → type mismatch, FAIL."""
        question = {
            "id": 9,
            "type": "exposure_calculator",
            "difficulty": "medium",
            "question": "Sector exposures?",
            "ground_truth": {
                "tool_name": "exposure_calculator",
                "parameters": {"portfolio_name": "Tech Innovation Fund"},
                "expected_result_type": "sector_exposure_breakdown",
                "explanation": "",
            },
        }
        agent = _mock_agent(
            "calculate_exposure",
            {"portfolio_name": "Tech Innovation Fund"},
            None,  # wrong type — should be ExposureResult
        )
        result = evaluate_question(question, agent)

        assert result.routing_correct
        assert not result.passed
        assert "ExposureResult" in result.comparison_details


# ---------------------------------------------------------------------------
# Tests: run_evaluation + print_report
# ---------------------------------------------------------------------------

class TestRunEvaluation:

    def test_run_evaluation_returns_results_for_all(self):
        questions = [
            {
                "id": 1, "type": "text2sql", "difficulty": "easy",
                "question": "Q1?",
                "ground_truth": {
                    "sql_query": "SELECT COUNT(*) FROM portfolios;",
                    "expected_result_type": "single_value",
                    "explanation": "",
                },
            },
            {
                "id": 9, "type": "exposure_calculator", "difficulty": "medium",
                "question": "Q9?",
                "ground_truth": {
                    "tool_name": "exposure_calculator",
                    "parameters": {"portfolio_name": "Tech Innovation Fund"},
                    "expected_result_type": "sector_exposure_breakdown",
                    "explanation": "",
                },
            },
        ]
        agent = MagicMock()
        agent.dispatch.side_effect = [
            ("run_sql_query", {"sql": "SELECT COUNT(*) FROM portfolios"},
             _sql_result([{"COUNT(*)": 13}], ["COUNT(*)"])),
            ("calculate_exposure", {"portfolio_name": "Tech Innovation Fund"},
             _exposure_result(portfolio_found=True,
                              sector_exposures={"Technology": 100.0},
                              sector_market_values={"Technology": 1565292.50})),
        ]

        results = run_evaluation(agent, questions)

        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_print_report_does_not_crash(self):
        """print_report should not raise for any combination of results."""
        results = [
            QuestionResult(
                question_id=1, question="Q1?", question_type="text2sql",
                difficulty="easy", expected_tool="run_sql_query",
                actual_tool="run_sql_query", routing_correct=True,
                data_correct=True, passed=True,
            ),
            QuestionResult(
                question_id=9, question="Q9?", question_type="exposure_calculator",
                difficulty="medium", expected_tool="calculate_exposure",
                actual_tool="decline_question", routing_correct=False,
                data_correct=False, passed=False, error="API 503",
            ),
        ]
        print_report(results)  # should not raise


# ---------------------------------------------------------------------------
# Tests: utility functions
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_has_order_by_true(self):
        assert _has_order_by("SELECT * FROM x ORDER BY y DESC")

    def test_has_order_by_false(self):
        assert not _has_order_by("SELECT * FROM x")

    def test_has_order_by_case_insensitive(self):
        assert _has_order_by("select * from x order by y")

    def test_expected_tool_text2sql(self):
        assert _expected_tool_for("text2sql") == "run_sql_query"

    def test_expected_tool_exposure(self):
        assert _expected_tool_for("exposure_calculator") == "calculate_exposure"

    def test_expected_tool_unknown(self):
        assert _expected_tool_for("unknown") == "unknown"


# ---------------------------------------------------------------------------
# Tests: ground truth loading
# ---------------------------------------------------------------------------

class TestGroundTruthLoading:

    def test_load_returns_list_of_questions(self):
        path = PROJECT_ROOT / "ground_truth_dataset.json"
        if not path.exists():
            pytest.skip("ground_truth_dataset.json not found")
        questions = load_ground_truth(path)
        assert isinstance(questions, list)
        assert len(questions) == 10
        assert all("id" in q for q in questions)
        assert all("ground_truth" in q for q in questions)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ground_truth(Path("/nonexistent/path.json"))
