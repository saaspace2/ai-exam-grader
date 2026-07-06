"""Shared test setup for the whole suite.

Two things live here, and because this is the TOP-LEVEL conftest, pytest makes
everything in it available to EVERY test folder automatically - no imports needed.

1. _force_mock (autouse): forces grading/vision/clarify to the free MOCK path so
   tests never hit the network (fast, deterministic, no cost).
2. Live-endpoint fixtures (endpoint_url, dbr_token, call_endpoint, sample): used
   by the endpoint tests (serving/fuzz/contract/performance). They SKIP unless
   ENDPOINT_URL and DBR_TOKEN are set, so offline CI still passes.
"""

import json
import os

import pytest

from grader.config import Settings


# ---------------------------------------------------------------------------
# 1. Force mock mode for every test (no real API calls).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _force_mock(monkeypatch):
    monkeypatch.setattr(Settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# 2. Live-endpoint fixtures (shared by all endpoint tests).
# ---------------------------------------------------------------------------
# One valid grading request: a question + a student answer, as the endpoint wants.
_SAMPLE = [{
    "question_json": json.dumps({
        "id": "Q1", "type": "mcq", "text": "Capital of France?",
        "correct_answer": "Paris", "max_marks": 2,
    }),
    "answer_json": json.dumps({
        "id": "A1", "question_id": "Q1",
        "student_id": "test", "answer_text": "Paris",
    }),
}]


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


@pytest.fixture(scope="session")
def endpoint_url() -> str:
    url = _env("ENDPOINT_URL")
    if not url:
        pytest.skip("ENDPOINT_URL not set - skipping live-endpoint test.")
    return url


@pytest.fixture(scope="session")
def dbr_token() -> str:
    tok = _env("DBR_TOKEN")
    if not tok:
        pytest.skip("DBR_TOKEN not set - skipping live-endpoint test.")
    return tok


@pytest.fixture(scope="session")
def call_endpoint(endpoint_url, dbr_token):
    """Return a function that POSTs records to the endpoint -> (status, text).

    The endpoint expects the MLflow 'dataframe_split' format:
        {"dataframe_split": {"columns": [...], "data": [[...], ...]}}
    We accept a list of {column: value} dicts and convert to that shape.
    """
    import requests

    def _to_split(records):
        if not records:
            return {"columns": [], "data": []}
        columns = list(records[0].keys())
        data = [[rec.get(c) for c in columns] for rec in records]
        return {"columns": columns, "data": data}

    def _call(records, timeout: int = 60):
        payload = {"dataframe_split": _to_split(records)}
        resp = requests.post(
            endpoint_url,
            headers={
                "Authorization": f"Bearer {dbr_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        return resp.status_code, resp.text
    return _call


@pytest.fixture(scope="session")
def sample():
    return list(_SAMPLE)