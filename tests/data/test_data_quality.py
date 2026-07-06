"""Data Quality Tests - is the data going into grading clean and valid?

Garbage In = Garbage Out. These check the models/data invariants using the
offline (mock) path, so they run anywhere.
"""

import pytest

from grader.models import Question, StudentAnswer, GradeRecord


class TestQuestionQuality:
    def test_max_marks_positive(self):
        q = Question(id="Q1", type="mcq", text="?", correct_answer="A", max_marks=2)
        assert q.max_marks > 0, "max_marks must be positive."

    def test_rejects_negative_marks(self):
        # Pydantic should reject an invalid max_marks.
        with pytest.raises(Exception):
            Question(id="Q1", type="mcq", text="?", correct_answer="A", max_marks=-1)


class TestGradeQuality:
    def test_marks_within_bounds(self):
        g = GradeRecord(id="G1", answer_id="A1", question_id="Q1", student_id="S",
                        marks_awarded=2, max_marks=2, justification="ok",
                        confidence=1.0, grading_method="mcq-exact")
        assert 0 <= g.marks_awarded <= g.max_marks, "Marks out of bounds."

    def test_confidence_is_fraction(self):
        g = GradeRecord(id="G1", answer_id="A1", question_id="Q1", student_id="S",
                        marks_awarded=1, max_marks=2, justification="ok",
                        confidence=0.5, grading_method="mcq-exact")
        assert 0.0 <= g.confidence <= 1.0, "Confidence must be a 0-1 fraction."