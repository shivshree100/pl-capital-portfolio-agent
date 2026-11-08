"""
evaluate.py — CLI for the ground-truth evaluator.

Usage
-----
    # Evaluate all 10 questions:
    python evaluate.py

    # Evaluate specific questions:
    python evaluate.py --question 1
    python evaluate.py --question 1,3,5
    python evaluate.py --question Q1,Q9

Requirements
------------
    - portfolio_database.db must exist  (run: python init_db.py)
    - GEMINI_API_KEY must be set in .env

Exit codes
----------
    0 — all evaluated questions passed
    1 — one or more questions failed (or an error occurred)
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import (
    load_ground_truth,
    run_evaluation,
    print_report,
)
from src.database.connection import get_db_path
from src.agent.portfolio_agent import PortfolioAgent


def _parse_question_ids(raw: str) -> list[int]:
    """Parse a comma-separated question ID string like '1,3,Q5' → [1, 3, 5]."""
    ids = []
    for part in raw.split(","):
        part = part.strip().upper()
        if part.startswith("Q"):
            part = part[1:]
        ids.append(int(part))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Portfolio Analytics Agent against ground-truth questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py              # all 10 questions\n"
            "  python evaluate.py --question 1 # only Q1\n"
            "  python evaluate.py --question 1,9,10\n"
        ),
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Comma-separated question IDs to evaluate (e.g. '1,3,5' or 'Q1,Q9').",
    )
    args = parser.parse_args()

    # --- Check database ---
    db = get_db_path()
    if not db.exists():
        print(f"\n[ERROR] Database not found: {db}")
        print("Run 'python init_db.py' first.")
        sys.exit(1)

    # --- Check API key ---
    from dotenv import load_dotenv
    import os
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("\n[ERROR] GEMINI_API_KEY is not set. Create a .env file.")
        sys.exit(1)

    # --- Load ground truth ---
    try:
        all_questions = load_ground_truth()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # --- Filter questions ---
    if args.question:
        ids = _parse_question_ids(args.question)
        questions = [q for q in all_questions if q["id"] in ids]
        if not questions:
            print(f"\n[ERROR] No questions found with IDs: {ids}")
            sys.exit(1)
        print(f"\nEvaluating {len(questions)} question(s): Q{', Q'.join(str(q['id']) for q in questions)}")
    else:
        questions = all_questions
        print(f"\nEvaluating all {len(questions)} ground-truth questions.")

    # --- Create agent ---
    try:
        agent = PortfolioAgent(db_path=db)
    except RuntimeError as exc:
        print(f"\n[ERROR] Failed to initialise agent: {exc}")
        sys.exit(1)

    # --- Run evaluation ---
    results = run_evaluation(agent, questions, db_path=db)

    # --- Print report ---
    print_report(results)

    # --- Exit code ---
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
