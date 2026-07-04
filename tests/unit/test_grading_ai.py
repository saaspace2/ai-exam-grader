"""Tests for Component 3.5: numeric step-marking routing + grader selection.

These use the MockAIGrader (passed explicitly) so they stay free and
deterministic - no network, no API key needed.
"""

import importlib

from grader.grading import MockAIGrader, grade_answer
from grader.models import Question, StudentAnswer


def _a(text, qid="Q"):
    return StudentAnswer(id="A", question_id=qid, student_id="Riya", answer_text=text)


class TestNumericRouting:
    """Numeric WITH a rubric uses AI (step marking); WITHOUT uses tolerance."""

    def test_plain_numeric_uses_tolerance(self):
        """No rubric -> exact tolerance check (method 'numeric-tolerance')."""
        q = Question(id="Q", type="numeric", text="7x8?", correct_answer="56",
                     tolerance=0, max_marks=5)
        g = grade_answer(q, _a("56"), ai=MockAIGrader())
        assert g.grading_method == "numeric-tolerance"
        assert g.marks_awarded == 5

    def test_numeric_with_rubric_uses_ai(self):
        """A rubric present -> AI step marking (method starts with 'ai-')."""
        q = Question(id="Q", type="numeric", text="Solve 2x+3=7 showing steps.",
                     rubric="subtract three divide two answer", max_marks=6)
        g = grade_answer(q, _a("subtract three then divide two, answer is two"),
                         ai=MockAIGrader())
        assert g.grading_method.startswith("ai-")
        assert 0 < g.marks_awarded <= 6


class TestGraderSelection:
    """get_ai_grader picks mock or real based on whether a key is set.

    We patch load_dotenv to do nothing during these tests, so the .env file
    on disk does not override the environment we set here.
    """

    def _reload_with_env(self, monkeypatch, key_value):
        """Reload config + grading with a controlled env and no .env reload."""
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

        if key_value is None:
            monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        else:
            monkeypatch.setenv("OPENROUTER_API_KEY", key_value)

        import grader.config as cfg
        importlib.reload(cfg)
        import grader.grading as grading
        importlib.reload(grading)
        return grading

    def test_returns_mock_when_no_key(self, monkeypatch):
        """With no API key, we get the free MockAIGrader."""
        grading = self._reload_with_env(monkeypatch, None)
        assert isinstance(grading.get_ai_grader(), grading.MockAIGrader)

    def test_returns_openrouter_when_key_set(self, monkeypatch):
        """With an API key set, we get the real OpenRouterAIGrader."""
        grading = self._reload_with_env(monkeypatch, "test-key-123")
        assert isinstance(grading.get_ai_grader(), grading.OpenRouterAIGrader)