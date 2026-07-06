"""Snapshot Tests - capture a known-good output and detect any drift from it.

We record the exact grade for a fixed input as a 'snapshot'. If future code
changes the output, the test fails - alerting you to review whether the change
was intended. Offline (mock), runs anywhere.
"""

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer

# The saved snapshot: what grading THIS input produced when we approved it.
SNAPSHOT = {
    "marks_awarded": 2.0,
    "max_marks": 2.0,
    "grading_method": "mcq-exact",
}


class TestSnapshot:
    def test_mcq_grade_matches_snapshot(self):
        q = Question(id="Q", type="mcq", text="Capital of France?",
                     correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="Paris")
        rec = grade_answer(q, a)
        # Compare the key fields against the approved snapshot.
        assert rec.marks_awarded == SNAPSHOT["marks_awarded"]
        assert rec.max_marks == SNAPSHOT["max_marks"]
        assert rec.grading_method == SNAPSHOT["grading_method"]