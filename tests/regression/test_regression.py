"""Regression Tests - known inputs must keep producing known outputs.

A regression is when a change breaks something that used to work. We pin down
exact expected grades for fixed inputs; if a code change alters them, these fail.
Offline (mock grader), runs anywhere.
"""

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer


class TestKnownGrades:
    def test_exact_mcq_match(self):
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="Paris")
        rec = grade_answer(q, a)
        assert rec.marks_awarded == 2.0   # pinned: correct MCQ = full marks

    def test_wrong_mcq_zero(self):
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="Rome")
        rec = grade_answer(q, a)
        assert rec.marks_awarded == 0.0   # pinned: wrong MCQ = zero

    def test_numeric_within_tolerance(self):
        q = Question(id="Q", type="numeric", text="2+2?", correct_answer="4",
                     tolerance=0.1, max_marks=1)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="4")
        rec = grade_answer(q, a)
        assert rec.marks_awarded == 1.0   # pinned: exact numeric = full marks