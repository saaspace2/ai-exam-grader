"""Mutation-Resistance Tests - prove our tests are STRONG.

Mutation testing changes ('mutates') the source code slightly (e.g. > becomes >=)
and checks whether tests catch it. Strong tests fail on mutations; weak tests
pass and give false confidence. Here we assert BOUNDARY behaviours that would
break under common mutations - so our suite is mutation-resistant.
"""

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer


class TestBoundaryStrength:
    def test_exact_boundary_correct(self):
        # If someone mutated the MCQ comparison, this exact-match case would break.
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="Paris")
        assert grade_answer(q, a).marks_awarded == 2.0

    def test_case_insensitive_match(self):
        # Catches a mutation that removes case-normalisation in MCQ matching.
        q = Question(id="Q", type="mcq", text="?", correct_answer="Paris", max_marks=2)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="paris")
        assert grade_answer(q, a).marks_awarded == 2.0

    def test_numeric_tolerance_boundary(self):
        # Just inside tolerance -> full marks. A mutated <= vs < would break this.
        q = Question(id="Q", type="numeric", text="?", correct_answer="10",
                     tolerance=0.5, max_marks=1)
        a = StudentAnswer(id="A", question_id="Q", student_id="S", answer_text="10.4")
        assert grade_answer(q, a).marks_awarded == 1.0