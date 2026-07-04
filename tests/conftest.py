"""Shared test setup: force everything offline/mock.

Grading, vision, and clarify all decide real-vs-mock via Settings.has_api_key(),
which reads the Settings CLASS attribute OPENROUTER_API_KEY. If a developer has a
real key in their .env, tests could accidentally hit the network and become slow,
costly, and flaky. This fixture blanks the key for the whole test session so every
factory picks the free mock. Tests that specifically want the real path can
override it themselves.
"""

import pytest

from grader.config import Settings


@pytest.fixture(autouse=True)
def _force_mock(monkeypatch):
    # blank the key on the CLASS (has_api_key is a classmethod reading the class attr)
    monkeypatch.setattr(Settings, "OPENROUTER_API_KEY", "")
    # also stop .env from reloading a real key during any importlib.reload
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)