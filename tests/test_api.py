"""
tests/test_api.py — Gemini API connectivity test.

This file tests that the Gemini API key is valid and the model responds.
It is a live integration test — it requires GEMINI_API_KEY and makes a
real API call.  When the key is unavailable, the test is skipped cleanly
so that pytest collection never fails.

Running
-------
    # All tests including this one:
    .venv\\Scripts\\python.exe -m pytest tests/ -v

    # This file only:
    .venv\\Scripts\\python.exe -m pytest tests/test_api.py -v

    # Everything except live API tests:
    .venv\\Scripts\\python.exe -m pytest tests/ -v -k "not Integration and not test_api"
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def gemini_api_key():
    """
    Load GEMINI_API_KEY from the project root .env file.
    Skips the test if the key is not found.
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        pytest.skip("GEMINI_API_KEY not set. Add it to the project root .env file.")
    return key


def test_gemini_api_responds(gemini_api_key):
    """
    Verify that the Gemini API key is valid and the model responds with text.
    This is a live test — it makes one real API call.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(timeout=120000),
    )

    # Use the same model the agent uses
    from src.agent.portfolio_agent import PortfolioAgent
    model = PortfolioAgent.DEFAULT_MODEL

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini API is working!",
    )

    assert response.text is not None, "Gemini returned no text"
    assert len(response.text.strip()) > 0, "Gemini returned empty text"
