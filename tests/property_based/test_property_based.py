"""Property-Based Tests - check PROPERTIES that must hold for ANY input.

Instead of fixed examples, we assert rules that must always be true - e.g.
"marks are never negative and never exceed max" - and let Hypothesis generate
hundreds of random inputs to try to break them. Offline (mock), runs anywhere.
"""

import pytest

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer

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


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestGradingProperties:
    @settings(max_examples=50, deadline=None)
    @given(
        answer_text=st.text(max_size=200),
        max_marks=st.floats(min_value=0.5, max_value=100),
    )
    def test_marks_never_out_of_bounds(self, answer_text, max_marks):
        # PROPERTY: for ANY answer + max, 0 <= marks <= max_marks.
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris",
                     max_marks=max_marks)
        a = StudentAnswer(id="A", question_id="Q", student_id="S",
                          answer_text=answer_text or "x")
        rec = grade_answer(q, a)
        assert 0 <= rec.marks_awarded <= max_marks

    @settings(max_examples=30, deadline=None)
    @given(answer_text=st.text(max_size=200))
    def test_confidence_always_fraction(self, answer_text):
        # PROPERTY: confidence is always a 0-1 fraction, whatever the answer.
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S",
                          answer_text=answer_text or "x")
        rec = grade_answer(q, a)
        assert 0.0 <= rec.confidence <= 1.0