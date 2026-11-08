"""
main.py — Interactive CLI for the Portfolio Analytics Agent.

Usage
-----
    python main.py

Requirements
------------
    - portfolio_database.db must exist  (run: python init_db.py)
    - GEMINI_API_KEY must be set in .env or the environment

Commands recognised by the REPL
---------------------------------
    exit / quit / q   — exit the session
    help              — show usage hints
    clear             — clear the screen (Windows: cls, others: clear)

All other input is treated as a natural-language question sent to the agent.
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_db_path
from src.database.loader import verify_database
from src.agent.portfolio_agent import PortfolioAgent


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def _check_database() -> bool:
    """Return True if the database exists and passes a quick sanity check."""
    db = get_db_path()
    if not db.exists():
        print(
            f"\n[ERROR] Database not found at:\n  {db}\n"
            "Run the initializer first:\n"
            "  python init_db.py\n"
        )
        return False
    return True


def _check_api_key() -> bool:
    """Return True if GEMINI_API_KEY is discoverable."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print(
            "\n[ERROR] GEMINI_API_KEY is not set.\n"
            "Create a .env file in the project root:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://ai.google.dev/gemini-api/docs\n"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# REPL helpers
# ---------------------------------------------------------------------------

_BANNER = """
╔══════════════════════════════════════════════════════╗
║       PL Capital — Portfolio Analytics Agent        ║
╚══════════════════════════════════════════════════════╝
  Ask me anything about your portfolios.
  Type 'help' for examples, 'exit' to quit.
"""

_HELP = """
Example questions
─────────────────
  How many portfolios do we have?
  What are the names of all active portfolios?
  Which securities are in the Technology sector?
  What is the total AUM for high risk portfolios?
  Show the top 5 holdings by cost basis in the Growth Equity Fund.
  What is the average current price of securities in each sector?
  What are the sector exposures for the Tech Innovation Fund?
  Calculate the sector exposure breakdown for the International Equity Fund.

Commands
────────
  help    Show this message
  exit    Exit the agent  (also: quit, q)
  clear   Clear the screen
"""


def _clear_screen() -> None:
    """Clear the terminal screen cross-platform."""
    os.system("cls" if os.name == "nt" else "clear")


def _print_separator() -> None:
    print("\n" + "─" * 56 + "\n")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def run_repl(agent: PortfolioAgent) -> None:
    print(_BANNER)

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in ("exit", "quit", "q"):
            print("\nGoodbye.")
            break

        if lower == "help":
            print(_HELP)
            continue

        if lower == "clear":
            _clear_screen()
            continue

        # Send to agent
        print()
        try:
            answer = agent.answer(raw)
            print(f"Agent: {answer}")
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
        except Exception as exc:
            print(f"[UNEXPECTED ERROR] {exc}")

        _print_separator()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not _check_database():
        sys.exit(1)

    if not _check_api_key():
        sys.exit(1)

    try:
        agent = PortfolioAgent()
    except RuntimeError as exc:
        print(f"\n[ERROR] Failed to initialise agent: {exc}")
        sys.exit(1)

    run_repl(agent)


if __name__ == "__main__":
    main()
