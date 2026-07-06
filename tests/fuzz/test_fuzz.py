"""Fuzz Tests - throw random/invalid input at the endpoint; it must NEVER 500.

Analogy: a vending machine - fuzzing inserts a banana instead of a coin and
presses 999 buttons. Golden rule: 200 = handled, 400 = rejected cleanly,
500 = CRASH = bug. Needs ENDPOINT_URL + DBR_TOKEN (else SKIP).
"""

import json

import pytest

try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    # Stand-ins so the decorators below do not crash at import time when
    # hypothesis is missing. The whole class is skipped anyway (see skipif).
    def given(*a, **k):
        def _wrap(f):
            return f
        return _wrap

    def settings(*a, **k):
        def _wrap(f):
            return f
        return _wrap

    class _St:
        def __getattr__(self, _):
            return lambda *a, **k: None
    st = _St()


def _q(text="Q?", correct="Paris", mx=2):
    return json.dumps({"id": "Q", "type": "mcq", "text": text,
                       "correct_answer": correct, "max_marks": mx})


def _a(ans="Paris"):
    return json.dumps({"id": "A", "question_id": "Q",
                       "student_id": "fuzz", "answer_text": ans})


class TestSchemaFuzzing:
    """Send wrong/missing structure - the endpoint must not crash (500)."""

    def test_empty_records_no_500(self, call_endpoint):
        status, _ = call_endpoint([])
        assert status != 500, "Endpoint crashed on empty records."

    def test_missing_fields_no_500(self, call_endpoint):
        status, _ = call_endpoint([{"question_json": _q()}])  # no answer_json
        assert status != 500, "Endpoint crashed on a missing field."

    def test_garbage_json_no_500(self, call_endpoint):
        status, _ = call_endpoint([{"question_json": "not json{{",
                                    "answer_json": "also not json"}])
        assert status != 500, "Endpoint crashed on garbage JSON."


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestRandomFuzzing:
    """Property-based random text - the endpoint must never 500."""

    @settings(max_examples=20, deadline=None)
    @given(answer=st.text(max_size=500))
    def test_random_answer_text_no_500(self, call_endpoint, answer):
        status, _ = call_endpoint([{"question_json": _q(), "answer_json": _a(answer)}])
        assert status != 500, f"Crashed on answer_text={answer!r}"