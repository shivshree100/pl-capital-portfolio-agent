"""
Ground-truth evaluator for the Portfolio Analytics Agent.

Loads the 10 ground-truth questions, computes expected results at runtime
(by executing the reference SQL or calling the exposure calculator), then
calls PortfolioAgent.dispatch() to get the agent's tool selection and
structured result.  Compares expected vs actual with type-aware logic.

The evaluator never calls the NL synthesis step — it works entirely with
structured data (SqlResult, ExposureResult), so evaluation is independent
of natural-language phrasing.

Public API
----------
evaluate_question(question_dict, agent, db_path) -> QuestionResult
run_evaluation(agent, questions, db_path) -> list[QuestionResult]
print_report(results) -> None

compare_sql_results(expected, actual, result_type, ordered) -> (bool, str)
compare_exposure_results(expected, actual) -> (bool, str)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.tools.sql_tool import execute_sql_query, SqlResult
from src.tools.exposure_tool import calculate_sector_exposure, ExposureResult
from src.database.connection import get_project_root


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """Result of evaluating one ground-truth question."""

    question_id: int
    question: str
    question_type: str          # "text2sql" or "exposure_calculator"
    difficulty: str

    expected_tool: str           # "run_sql_query" or "calculate_exposure"
    actual_tool: str = ""        # what the agent actually called
    routing_correct: bool = False
    data_correct: bool = False
    passed: bool = False
    error: str = ""             # non-empty only on infrastructure failure
    comparison_details: str = "" # human-readable comparison outcome


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: Optional[Path] = None) -> list[dict]:
    """
    Load and return the questions array from ground_truth_dataset.json.

    Returns a list of question dicts in their original order (Q1–Q10).
    """
    if path is None:
        path = get_project_root() / "ground_truth_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ---------------------------------------------------------------------------
# Comparison helpers (pure functions — fully unit-testable)
# ---------------------------------------------------------------------------

NUMERIC_TOLERANCE = 0.01  # absolute tolerance for float comparisons


def _compare_values(expected, actual) -> tuple[bool, str]:
    """
    Compare two scalar values with numeric tolerance where appropriate.

    Returns (is_match, detail_string).
    """
    # Try numeric comparison
    try:
        exp_num = float(expected)
        act_num = float(actual)
        if abs(exp_num - act_num) <= NUMERIC_TOLERANCE:
            return True, f"{expected} == {actual}"
        return False, f"expected {expected}, got {actual}"
    except (ValueError, TypeError):
        pass

    # String comparison (case-sensitive — SQLite is case-sensitive)
    exp_str = str(expected).strip()
    act_str = str(actual).strip()
    if exp_str == act_str:
        return True, f"'{exp_str}'"
    return False, f"expected '{exp_str}', got '{act_str}'"


def _compare_lists(expected: list, actual: list) -> tuple[bool, str]:
    """Compare two lists of values, order-independent."""
    if len(expected) != len(actual):
        return False, f"length: expected {len(expected)}, got {len(actual)}"

    # Try numeric comparison (sorted, element-by-element)
    try:
        exp_sorted = sorted(float(v) for v in expected)
        act_sorted = sorted(float(v) for v in actual)
        for e, a in zip(exp_sorted, act_sorted):
            if abs(e - a) > NUMERIC_TOLERANCE:
                return False, f"value {e} != {a}"
        return True, f"all {len(expected)} values match"
    except (ValueError, TypeError):
        pass

    # String set comparison
    exp_set = {str(v).strip() for v in expected}
    act_set = {str(v).strip() for v in actual}
    if exp_set == act_set:
        return True, f"all {len(expected)} values match"
    missing = exp_set - act_set
    return False, f"{len(missing)} missing values: {list(missing)[:5]}"


def _row_to_tuple(row: dict, columns: list) -> tuple:
    """Extract a row dict into a positional tuple."""
    return tuple(row.get(col) for col in columns)


def compare_sql_results(
    expected: SqlResult,
    actual: SqlResult,
    result_type: str,
    ordered: bool = False,
) -> tuple[bool, str]:
    """
    Compare two SqlResult objects based on the expected result type.

    Parameters
    ----------
    expected : SqlResult
    actual : SqlResult
    result_type : str — "single_value", "list", or "table"
    ordered : bool — if True, rows must appear in the same order

    Returns
    -------
    (is_match, detail_string)
    """
    if not expected.success:
        return False, f"expected SQL failed: {expected.error}"
    if not actual.success:
        return False, f"actual SQL failed: {actual.error}"

    # --- single_value ---
    if result_type == "single_value":
        if expected.row_count != 1 or len(expected.columns) != 1:
            return False, f"expected has {expected.row_count} rows × {len(expected.columns)} cols"
        if actual.row_count != 1 or len(actual.columns) != 1:
            return False, f"actual has {actual.row_count} rows × {len(actual.columns)} cols"

        exp_val = expected.rows[0][expected.columns[0]]
        act_val = actual.rows[0][actual.columns[0]]
        return _compare_values(exp_val, act_val)

    # --- list (single column, multiple rows) ---
    if result_type == "list":
        if len(expected.columns) != 1:
            return False, f"expected has {len(expected.columns)} columns (need 1)"
        if len(actual.columns) != 1:
            return False, f"actual has {len(actual.columns)} columns (need 1)"

        exp_list = [row[expected.columns[0]] for row in expected.rows]
        act_list = [row[actual.columns[0]] for row in actual.rows]
        return _compare_lists(exp_list, act_list)

    # --- table (multiple columns, multiple rows) ---
    if result_type == "table":
        if expected.row_count != actual.row_count:
            return False, f"rows: expected {expected.row_count}, got {actual.row_count}"

        n_cols = len(expected.columns)

        # Try name-based projection first: check that all expected column
        # names exist in the actual result.  If they do, project actual rows
        # onto expected columns by name (handles extra columns + reordering).
        actual_col_set = set(actual.columns)
        missing_cols = [c for c in expected.columns if c not in actual_col_set]

        if not missing_cols:
            # Name-based: project actual rows onto expected columns by name
            exp_rows = [_row_to_tuple(r, expected.columns) for r in expected.rows]
            act_rows = [_row_to_tuple(r, expected.columns) for r in actual.rows]
        elif n_cols <= len(actual.columns):
            # Name-based failed (different aliases) — fall back to positional
            # comparison: compare the first N columns of actual against the
            # N expected columns.  This handles different column aliases
            # (e.g. agent used 'average_price' vs ground truth 'avg_price').
            exp_rows = [_row_to_tuple(r, expected.columns) for r in expected.rows]
            act_rows = [
                tuple(r[actual.columns[j]] for j in range(n_cols))
                for r in actual.rows
            ]
        else:
            return False, (
                f"columns: expected {n_cols} columns "
                f"({expected.columns}), but actual has only "
                f"{len(actual.columns)} ({actual.columns})"
            )

        if not ordered:
            # Sort by string representation for order-independent comparison
            exp_rows = sorted(exp_rows, key=lambda t: str(t))
            act_rows = sorted(act_rows, key=lambda t: str(t))

        for i, (exp_tuple, act_tuple) in enumerate(zip(exp_rows, act_rows)):
            for j, (e, a) in enumerate(zip(exp_tuple, act_tuple)):
                match, msg = _compare_values(e, a)
                if not match:
                    return False, f"row {i}, col {j}: {msg}"

        return True, f"all {expected.row_count} rows match ({'ordered' if ordered else 'unordered'})"

    return False, f"unknown result_type: {result_type}"


def compare_exposure_results(
    expected: ExposureResult,
    actual: ExposureResult,
) -> tuple[bool, str]:
    """
    Compare two ExposureResult objects.

    Checks portfolio_found, sector names, market values, and percentages
    with numeric tolerance.
    """
    if not expected.portfolio_found:
        return False, "expected result: portfolio not found"
    if not actual.portfolio_found:
        return False, "actual result: portfolio not found"

    # Sector names must match (as sets)
    exp_sectors = set(expected.sector_exposures.keys())
    act_sectors = set(actual.sector_exposures.keys())
    if exp_sectors != act_sectors:
        missing = exp_sectors - act_sectors
        extra = act_sectors - exp_sectors
        parts = []
        if missing:
            parts.append(f"missing: {missing}")
        if extra:
            parts.append(f"extra: {extra}")
        return False, f"sectors: {', '.join(parts)}"

    # Compare market values
    for sector in expected.sector_market_values:
        exp_mv = expected.sector_market_values[sector]
        act_mv = actual.sector_market_values.get(sector, 0)
        if abs(float(exp_mv) - float(act_mv)) > NUMERIC_TOLERANCE:
            return False, f"{sector} market value: expected {exp_mv}, got {act_mv}"

    # Compare percentages
    for sector in expected.sector_exposures:
        exp_pct = expected.sector_exposures[sector]
        act_pct = actual.sector_exposures.get(sector, 0)
        if abs(float(exp_pct) - float(act_pct)) > NUMERIC_TOLERANCE:
            return False, f"{sector} exposure: expected {exp_pct}%, got {act_pct}%"

    return True, f"all {len(exp_sectors)} sectors match"


# ---------------------------------------------------------------------------
# Tool-routing logic
# ---------------------------------------------------------------------------

def _expected_tool_for(question_type: str) -> str:
    """Map a ground-truth question type to the expected tool name."""
    if question_type == "text2sql":
        return "run_sql_query"
    if question_type == "exposure_calculator":
        return "calculate_exposure"
    return "unknown"


def _has_order_by(sql: str) -> bool:
    """Check whether a SQL string contains an ORDER BY clause."""
    return "ORDER BY" in sql.upper()


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    question_dict: dict,
    agent,
    db_path: Optional[Path] = None,
) -> QuestionResult:
    """
    Evaluate a single ground-truth question against the agent.

    Parameters
    ----------
    question_dict : dict
        One entry from ground_truth_dataset.json.
    agent : PortfolioAgent
        Must have a dispatch(question) method.
    db_path : Path, optional
        Database path for computing expected results.

    Returns
    -------
    QuestionResult
    """
    q_id = question_dict["id"]
    q_text = question_dict["question"]
    q_type = question_dict["type"]
    difficulty = question_dict["difficulty"]
    gt = question_dict["ground_truth"]
    expected_tool = _expected_tool_for(q_type)

    result = QuestionResult(
        question_id=q_id,
        question=q_text,
        question_type=q_type,
        difficulty=difficulty,
        expected_tool=expected_tool,
    )

    try:
        # --- Step 1: compute expected result at runtime ---
        expected_sql_result: Optional[SqlResult] = None
        expected_exposure_result: Optional[ExposureResult] = None
        result_type = ""
        ordered = False

        if q_type == "text2sql":
            sql = gt["sql_query"]
            expected_sql_result = execute_sql_query(sql, db_path=db_path)
            result_type = gt.get("expected_result_type", "table")
            ordered = _has_order_by(sql)

        elif q_type == "exposure_calculator":
            portfolio_name = gt["parameters"]["portfolio_name"]
            expected_exposure_result = calculate_sector_exposure(
                portfolio_name, db_path=db_path
            )
            result_type = gt.get("expected_result_type", "sector_exposure_breakdown")

        # --- Step 2: call agent dispatch (no synthesis) ---
        tool_name, tool_args, tool_result = agent.dispatch(q_text)

        result.actual_tool = tool_name
        result.routing_correct = (tool_name == expected_tool)

        # --- Step 3: compare ---
        if not result.routing_correct:
            result.comparison_details = (
                f"wrong tool: expected '{expected_tool}', got '{tool_name}'"
            )
            result.passed = False
            return result

        if q_type == "text2sql":
            if not isinstance(tool_result, SqlResult):
                result.comparison_details = (
                    f"expected SqlResult, got {type(tool_result).__name__}"
                )
                result.passed = False
                return result

            match, details = compare_sql_results(
                expected_sql_result, tool_result, result_type, ordered
            )
            result.data_correct = match
            result.comparison_details = details

        elif q_type == "exposure_calculator":
            if not isinstance(tool_result, ExposureResult):
                result.comparison_details = (
                    f"expected ExposureResult, got {type(tool_result).__name__}"
                )
                result.passed = False
                return result

            match, details = compare_exposure_results(
                expected_exposure_result, tool_result
            )
            result.data_correct = match
            result.comparison_details = details

        result.passed = result.routing_correct and result.data_correct

    except RuntimeError as exc:
        result.error = str(exc)
        result.passed = False
    except Exception as exc:
        result.error = f"unexpected error: {exc}"
        result.passed = False

    return result


def run_evaluation(
    agent,
    questions: list[dict],
    db_path: Optional[Path] = None,
) -> list[QuestionResult]:
    """
    Run the full evaluation across all provided questions.

    Returns a list of QuestionResult objects.
    """
    results = []
    for q in questions:
        r = evaluate_question(q, agent, db_path)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[QuestionResult]) -> None:
    """Print a formatted evaluation report to stdout."""
    total = len(results)

    print(f"\n{'='*72}")
    print("  Ground-Truth Evaluation Report")
    print(f"{'='*72}")
    print()

    # Per-question table
    header = (
        f"  {'Q':>3}  {'Type':<20} {'Diff':<8} "
        f"{'Route':<8} {'Data':<8} {'Result':<6}  Details"
    )
    print(header)
    print(f"  {'-'*3}  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6}  {'-'*40}")

    for r in results:
        route = "OK" if r.routing_correct else "FAIL"
        data = "OK" if r.data_correct else "FAIL"
        verdict = "PASS" if r.passed else "FAIL"
        if r.error:
            detail = f"[ERROR] {r.error[:50]}"
        else:
            detail = r.comparison_details[:50] if r.comparison_details else ""
        print(
            f"  Q{r.question_id:<2} {r.question_type:<20} {r.difficulty:<8} "
            f"{route:<8} {data:<8} {verdict:<6}  {detail}"
        )

    # Summary
    passed = sum(1 for r in results if r.passed)
    routing_ok = sum(1 for r in results if r.routing_correct)
    # Data accuracy is measured only against questions that were correctly
    # routed — if routing failed, the data comparison never ran and should
    # not count against the data accuracy metric.
    routed_results = [r for r in results if r.routing_correct]
    data_ok = sum(1 for r in routed_results if r.data_correct)
    not_routed = total - len(routed_results)

    print()
    if total > 0:
        print(f"  Routing accuracy : {routing_ok}/{total}  ({routing_ok/total*100:.0f}%)")
        if routed_results:
            print(f"  Data accuracy    : {data_ok}/{len(routed_results)}  "
                  f"({data_ok/len(routed_results)*100:.0f}%)  "
                  f"[of {len(routed_results)} correctly routed]")
        else:
            print(f"  Data accuracy    : N/A  (no questions were correctly routed)")
        print(f"  Overall pass rate: {passed}/{total}  ({passed/total*100:.0f}%)")
        if not_routed > 0:
            print(f"  Not evaluated    : {not_routed} question(s) skipped due to "
                  f"routing failure")
    else:
        print("  No questions evaluated.")
    print(f"{'='*72}\n")
