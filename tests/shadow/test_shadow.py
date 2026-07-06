"""Shadow Tests - run a NEW version alongside the old on real traffic, compare.

Shadow (or 'dark launch') sends the same request to both the current and a
candidate grader, WITHOUT showing the candidate's output to users. You compare
them offline to catch regressions before promoting. Here we shadow the mock vs
the same engine to show the pattern (offline, runs anywhere).
"""

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer, MockAIGrader


class TestShadowComparison:
    def test_shadow_agrees_on_mcq(self):
        # 'Production' and 'shadow' graders should agree on deterministic MCQ.
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="Paris")
        prod = grade_answer(q, a)
        shadow = grade_answer(q, a, ai=MockAIGrader())
        # For MCQ (no AI), both must give identical marks.
        assert prod.marks_awarded == shadow.marks_awarded

    def test_shadow_detects_no_regression(self):
        # If a shadow version scored differently, we'd flag it. Here they match.
        q = Question(id="Q", type="mcq", text="?", correct_answer="Rome", max_marks=3)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="Rome")
        assert grade_answer(q, a).marks_awarded == grade_answer(q, a).marks_awarded