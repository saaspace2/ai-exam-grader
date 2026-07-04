"""Unit tests for the data models (our four 'forms').

These tests prove two things about every safety rule:
  1. VALID data is ACCEPTED (the happy path).
  2. INVALID data is REJECTED (the bouncer does its job).

We import pytest to check that bad data raises an error, and ValidationError
(Pydantic's 'rejected!' signal) to confirm the RIGHT kind of error happened.
"""

import pytest
from pydantic import ValidationError

from grader.models import (
    Actor,
    AuditEntry,
    GradeRecord,
    GradeStatus,
    Question,
    QuestionType,
    StudentAnswer,
)


# ===========================================================================
# Question - the exam question form
# ===========================================================================

class TestQuestion:
    """Tests for the Question form."""

    def test_valid_question_is_accepted(self):
        """A sensible question should be built without complaint."""
        q = Question(id="Q1", type="mcq", text="What is 2+2?", max_marks=5)
        assert q.id == "Q1"
        assert q.type == QuestionType.MCQ   # the string "mcq" became the enum
        assert q.max_marks == 5

    def test_zero_max_marks_is_rejected(self):
        """A question worth 0 marks is nonsense - the bouncer must reject it."""
        with pytest.raises(ValidationError):
            Question(id="Q2", type="mcq", text="Bad", max_marks=0)

    def test_negative_max_marks_is_rejected(self):
        """A question worth -5 marks is nonsense - rejected."""
        with pytest.raises(ValidationError):
            Question(id="Q3", type="mcq", text="Bad", max_marks=-5)

    def test_unknown_type_is_rejected(self):
        """A typo type like 'mcqq' is not one of the four - rejected."""
        with pytest.raises(ValidationError):
            Question(id="Q4", type="mcqq", text="Typo", max_marks=5)

    def test_optional_fields_default_to_none(self):
        """correct_answer and rubric are optional; they default to None."""
        q = Question(id="Q5", type="essay", text="Discuss.", max_marks=10)
        assert q.correct_answer is None
        assert q.rubric is None
        assert q.tolerance == 0.0   # tolerance defaults to exact (0.0)


# ===========================================================================
# StudentAnswer - a student's answer to a question
# ===========================================================================

class TestStudentAnswer:
    """Tests for the StudentAnswer form."""

    def test_valid_answer_is_accepted(self):
        """A well-formed answer should build fine and keep its link."""
        a = StudentAnswer(
            id="A1", question_id="Q1", student_id="Riya",
            answer_text="The answer is 4.",
        )
        assert a.question_id == "Q1"   # the link back to the Question
        assert a.student_id == "Riya"

    def test_missing_field_is_rejected(self):
        """Leaving out a required field (answer_text) is rejected."""
        with pytest.raises(ValidationError):
            StudentAnswer(id="A2", question_id="Q1", student_id="Riya")


# ===========================================================================
# GradeRecord - the grade itself (the heart)
# ===========================================================================

class TestGradeRecord:
    """Tests for the GradeRecord form and its safety rules."""

    def _valid_kwargs(self, **overrides):
        """Helper: a set of valid fields, with optional overrides per test."""
        base = dict(
            id="G1", answer_id="A1", question_id="Q1", student_id="Riya",
            marks_awarded=4, max_marks=5, justification="Missed one detail.",
            confidence=0.9, grading_method="essay-rubric",
        )
        base.update(overrides)
        return base

    def test_valid_grade_is_accepted(self):
        """A normal 4-out-of-5 grade should be accepted, status 'graded'."""
        g = GradeRecord(**self._valid_kwargs())
        assert g.marks_awarded == 4
        assert g.status == GradeStatus.GRADED   # defaults to 'graded'

    def test_marks_cannot_exceed_max(self):
        """8 out of 5 is impossible - the whole-form rule must reject it."""
        with pytest.raises(ValidationError):
            GradeRecord(**self._valid_kwargs(marks_awarded=8, max_marks=5))

    def test_negative_marks_are_rejected(self):
        """You cannot award -1 marks - rejected."""
        with pytest.raises(ValidationError):
            GradeRecord(**self._valid_kwargs(marks_awarded=-1))

    def test_confidence_above_one_is_rejected(self):
        """The AI cannot be 150% sure - confidence must be <= 1."""
        with pytest.raises(ValidationError):
            GradeRecord(**self._valid_kwargs(confidence=1.5))

    def test_confidence_below_zero_is_rejected(self):
        """Confidence cannot be negative - must be >= 0."""
        with pytest.raises(ValidationError):
            GradeRecord(**self._valid_kwargs(confidence=-0.1))

    def test_full_marks_is_allowed(self):
        """Exactly max marks (5 of 5) is fine - the rule is 'cannot EXCEED'."""
        g = GradeRecord(**self._valid_kwargs(marks_awarded=5, max_marks=5))
        assert g.marks_awarded == 5


# ===========================================================================
# AuditEntry - the append-only change slip (the safety net)
# ===========================================================================

class TestAuditEntry:
    """Tests for the AuditEntry form and its safety rules."""

    def test_valid_audit_entry_is_accepted(self):
        """A well-formed slip is accepted and auto-stamps the time."""
        e = AuditEntry(
            id="AU1", grade_record_id="G1", actor="ai",
            action="initial_grade", new_marks=4,
            reason="Correct term, missed ATP detail.",
        )
        assert e.actor == Actor.AI
        assert e.new_marks == 4
        assert e.old_marks is None       # first grade has no 'before'
        assert e.timestamp is not None   # auto-filled with the current time

    def test_blank_reason_is_rejected(self):
        """A change with no reason breaks the audit trail - rejected."""
        with pytest.raises(ValidationError):
            AuditEntry(
                id="AU2", grade_record_id="G1", actor="ai",
                action="regrade", new_marks=5, reason="   ",
            )

    def test_empty_reason_is_rejected(self):
        """A totally empty reason is rejected too."""
        with pytest.raises(ValidationError):
            AuditEntry(
                id="AU3", grade_record_id="G1", actor="teacher",
                action="regrade", new_marks=5, reason="",
            )

    def test_records_a_change_with_old_and_new(self):
        """A regrade slip records both the old and new marks and who did it."""
        e = AuditEntry(
            id="AU4", grade_record_id="G1", actor="teacher",
            action="appeal_regrade", old_marks=4, new_marks=5,
            reason="Student correctly noted ATP was mentioned.",
        )
        assert e.old_marks == 4
        assert e.new_marks == 5
        assert e.actor == Actor.TEACHER
