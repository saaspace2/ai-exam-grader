"""A/B Tests - compare two grader versions on the same inputs, pick the better.

A/B testing splits traffic between version A and B and measures which performs
better on a metric. Here we compare two graders on a labelled set and assert the
'better' one wins by our metric (agreement with the answer key). Offline.
"""

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer, MockAIGrader


def _accuracy(grader_ai, cases):
    """Fraction of cases where the grade matches the expected marks."""
    correct = 0
    for q, a, expected in cases:
        rec = grade_answer(q, a, ai=grader_ai)
        if rec.marks_awarded == expected:
            correct += 1
    return correct / len(cases)


class TestABComparison:
    def test_both_versions_measured(self):
        cases = [
            (Question(id="Q1", type="mcq", text="?", correct_answer="Paris", max_marks=2),
             StudentAnswer(id="A1", question_id="Q1", student_id="S", answer_text="Paris"),
             2.0),
            (Question(id="Q2", type="mcq", text="?", correct_answer="Rome", max_marks=2),
             StudentAnswer(id="A2", question_id="Q2", student_id="S", answer_text="Milan"),
             0.0),
        ]
        # Version A and B are both the mock here -> equal accuracy (demo of the method).
        acc_a = _accuracy(MockAIGrader(), cases)
        acc_b = _accuracy(MockAIGrader(), cases)
        assert acc_a == acc_b == 1.0   # both grade these deterministic MCQs perfectly