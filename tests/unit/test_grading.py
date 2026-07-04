"""Unit tests for grading (Component 3).

We test each grading strategy: MCQ exact match, numeric tolerance, and the
AI-judged path (using the deterministic MockAIGrader so results are stable).
Every test also confirms the result is a valid GradeRecord.
"""

from grader.grading import MockAIGrader, grade_answer
from grader.models import GradeRecord, Question, StudentAnswer


def _q(**kw):
    """Small helper to build a Question with sensible defaults."""
    base = dict(id="Q", type="mcq", text="t", max_marks=5)
    base.update(kw)
    return Question(**base)


def _a(text, qid="Q"):
    """Small helper to build a StudentAnswer."""
    return StudentAnswer(id="A", question_id=qid, student_id="Riya", answer_text=text)


class TestMCQ:
    """Exact-match grading."""

    def test_correct_mcq_gets_full_marks(self):
        q = _q(type="mcq", correct_answer="4", max_marks=5)
        g = grade_answer(q, _a("4"))
        assert g.marks_awarded == 5
        assert g.confidence == 1.0

    def test_wrong_mcq_gets_zero(self):
        q = _q(type="mcq", correct_answer="4", max_marks=5)
        g = grade_answer(q, _a("5"))
        assert g.marks_awarded == 0

    def test_mcq_is_case_insensitive(self):
        q = _q(type="mcq", correct_answer="Paris", max_marks=5)
        g = grade_answer(q, _a("paris"))
        assert g.marks_awarded == 5   # "paris" matches "Paris"


class TestNumeric:
    """Tolerance-based grading."""

    def test_within_tolerance_gets_full_marks(self):
        q = _q(type="numeric", correct_answer="3.14", tolerance=0.01, max_marks=5)
        g = grade_answer(q, _a("3.14159"))
        assert g.marks_awarded == 5

    def test_outside_tolerance_gets_zero(self):
        q = _q(type="numeric", correct_answer="3.14", tolerance=0.01, max_marks=5)
        g = grade_answer(q, _a("3.20"))
        assert g.marks_awarded == 0

    def test_non_number_answer_gets_zero(self):
        q = _q(type="numeric", correct_answer="3.14", tolerance=0.01, max_marks=5)
        g = grade_answer(q, _a("banana"))
        assert g.marks_awarded == 0


class TestAIJudged:
    """Short/essay grading via the MockAIGrader (deterministic)."""

    def test_essay_partial_credit(self):
        q = _q(type="essay", text="Explain.", rubric="alpha beta gamma delta", max_marks=10)
        # answer contains 2 of the 4 keywords -> ~half marks
        g = grade_answer(q, _a("the alpha and beta parts"), ai=MockAIGrader())
        assert 0 < g.marks_awarded < 10          # partial credit
        assert g.grading_method == "ai-essay"

    def test_essay_full_match(self):
        q = _q(type="essay", text="Explain.", rubric="alpha beta gamma", max_marks=9)
        g = grade_answer(q, _a("alpha beta gamma all present"), ai=MockAIGrader())
        assert g.marks_awarded == 9              # all keywords matched

    def test_result_is_a_valid_grade_record(self):
        q = _q(type="short", text="Define X.", rubric="clear correct definition", max_marks=4)
        g = grade_answer(q, _a("a clear and correct definition"), ai=MockAIGrader())
        assert isinstance(g, GradeRecord)        # bouncer validated the AI output
        assert g.marks_awarded <= g.max_marks    # safety rule held


class TestRouting:
    """The main grade_answer routes by question type."""

    def test_grade_record_links_are_correct(self):
        q = _q(id="Q9", type="mcq", correct_answer="yes", max_marks=1)
        a = StudentAnswer(id="A9", question_id="Q9", student_id="Sam", answer_text="yes")
        g = grade_answer(q, a)
        assert g.question_id == "Q9"
        assert g.student_id == "Sam"
        assert g.answer_id == "A9"
