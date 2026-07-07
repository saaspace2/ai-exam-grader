"""Configuration for the AI Exam-Grading System.

Loads settings from a local .env file (never committed to git) and decides
WHICH AI grader to use:
  - if OPENROUTER_API_KEY is set  -> the real OpenRouterAIGrader
  - otherwise                     -> the free, offline MockAIGrader

This keeps tests free and deterministic (no key = mock), while real grading
turns on automatically the moment a key is present. The API key is read from
the environment - it is NEVER written into the code.
"""

import os

from dotenv import load_dotenv

# Read a .env file in the project root (if present) into environment variables.
load_dotenv()


class Settings:
    """All tunable settings, read from the environment with safe defaults."""

    # The API key. None if not set -> the system falls back to the mock grader.
    OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")

    # Which model to use. 'openrouter/free' auto-picks a working FREE model,
    # so it costs nothing and keeps working even as individual free models change.
    OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "openrouter/free")

    # The OpenRouter API endpoint (OpenAI-compatible).
    OPENROUTER_BASE_URL: str = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
    )

    # Where the SQLite database lives (the API uses this persistent file).
    GRADER_DB_PATH: str = os.environ.get("GRADER_DB_PATH", "grader.sqlite")

    # Where uploaded answer-script images are saved (maps to a Databricks Volume later).
    UPLOADS_DIR: str = os.environ.get("UPLOADS_DIR", "uploads")

    # Where the auto-saved CSV of every prediction (grade + answer + photo link)
    # is appended to, one row per graded answer.
    CSV_EXPORT_PATH: str = os.environ.get("CSV_EXPORT_PATH", "grader_dataset.csv")

    # Obvious placeholder values that do NOT count as a real key.
    _PLACEHOLDERS = {"", "paste-your-openrouter-key-here", "your-key-here", "sk-..."}

    @classmethod
    def has_api_key(cls) -> bool:
        """True only if a real (non-placeholder) API key is configured."""
        key = (cls.OPENROUTER_API_KEY or "").strip()
        return key not in cls._PLACEHOLDERS


settings = Settings()