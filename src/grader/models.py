"""Data models (our 'forms') for the AI Exam-Grading System.

This file defines the SHAPE of every piece of information in the system,
using Pydantic. Pydantic is the 'bouncer': it checks each form is filled in
correctly and REJECTS bad data at the door.

There are FOUR forms, and they link together in a chain:

    Question  <--  StudentAnswer  <--  GradeRecord  <--  AuditEntry

    - A StudentAnswer points back to a Question.
    - A GradeRecord points to a StudentAnswer (and its Question).
    - An AuditEntry points to a GradeRecord (every change leaves a trail).

That chain is what lets us trace ANY grade back to the exact answer, the
exact question, and every change ever made to it.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Small fixed-choice lists (Enums). An Enum means "the value MUST be one of
# these exact options" - anything else is rejected by the bouncer.
# ---------------------------------------------------------------------------

class QuestionType(str, Enum):
    """The four kinds of exam question we can grade."""

    MCQ = "mcq"          # multiple choice: graded by exact match
    NUMERIC = "numeric"  # a number: graded by 'close enough' (tolerance)
    SHORT = "short"      # short answer: graded by AI against a short rubric
    ESSAY = "essay"      # essay: graded by AI against a fuller rubric


class GradeStatus(str, Enum):
    """Where a grade is in its life-cycle."""

    GRADED = "graded"              # freshly graded, nothing disputed
    UNDER_APPEAL = "under_appeal"  # a student has flagged it for review
    REGRADED = "regraded"          # it was re-checked and possibly changed


class Actor(str, Enum):
    """WHO (or what) performed an action that changed a grade."""

    AI = "ai"            # the AI grader
    TEACHER = "teacher"  # a human teacher
    SYSTEM = "system"    # the system itself (e.g. initial grading)


# ---------------------------------------------------------------------------
# Form 1 - Question: one exam question.
# ---------------------------------------------------------------------------

class Question(BaseModel):
    """One exam question."""

    id: str                       # unique label, e.g. "Q3". Must be text.
    type: QuestionType            # must be one of the four kinds.
    text: str                     # the wording the student reads.
    max_marks: float = Field(gt=0)  # worth > 0 marks (0 or negative is nonsense).

    # For MCQ / NUMERIC we store the correct answer. Optional (essays have none).
    correct_answer: str | None = None

    # For SHORT / ESSAY we store the marking guide. Optional.
    rubric: str | None = None

    # For NUMERIC only: how close is 'close enough'. Defaults to 0 (exact).
    tolerance: float = Field(default=0.0, ge=0)


# ---------------------------------------------------------------------------
# Form 2 - StudentAnswer: one student's answer to one question.
# ---------------------------------------------------------------------------

class StudentAnswer(BaseModel):
    """A single student's answer to a single question."""

    id: str              # unique label for this answer, e.g. "A_Riya_Q3".
    question_id: str     # which Question this answers (links to Question.id).
    student_id: str      # which student wrote it, e.g. "Riya".
    answer_text: str     # what the student actually wrote.


# ---------------------------------------------------------------------------
# Form 3 - GradeRecord: the result after grading. The heart of the system.
# ---------------------------------------------------------------------------

class GradeRecord(BaseModel):
    """The outcome of grading one StudentAnswer."""

    id: str              # unique label for this grade, e.g. "G_Riya_Q3".
    answer_id: str       # which StudentAnswer this grades (links to it).
    question_id: str     # which Question (kept here too for easy lookup).
    student_id: str      # which student.

    marks_awarded: float = Field(ge=0)  # marks given. Never negative.
    max_marks: float = Field(gt=0)      # out of how many. Always > 0.

    justification: str                  # WHY these marks were given.
    confidence: float = Field(ge=0, le=1)  # how sure the grader is (0.0 to 1.0).
    grading_method: str                 # how it was graded, e.g. "mcq-exact".
    status: GradeStatus = GradeStatus.GRADED  # life-cycle; starts as 'graded'.

    @model_validator(mode="after")
    def marks_cannot_exceed_max(self) -> "GradeRecord":
        """Safety rule: a student can never score MORE than the max marks."""
        if self.marks_awarded > self.max_marks:
            raise ValueError(
                f"marks_awarded ({self.marks_awarded}) cannot exceed "
                f"max_marks ({self.max_marks})."
            )
        return self


# ---------------------------------------------------------------------------
# Form 4 - AuditEntry: an immutable record of ONE change to a grade.
# This is the safety heart: the audit drawer is APPEND-ONLY.
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """One immutable slip recording a change (or the creation) of a grade."""

    id: str                 # unique label for this audit slip.
    grade_record_id: str    # which GradeRecord this slip is about.

    # When it happened. Defaults to 'right now' (in UTC) if not given.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    actor: Actor            # who/what made the change (ai / teacher / system).
    action: str             # a short label, e.g. "initial_grade" or "appeal_regrade".

    old_marks: float | None = None  # marks before the change (None for the first grade).
    new_marks: float               # marks after the change.

    reason: str = Field(min_length=1)  # WHY. Must not be empty - no silent changes.

    @model_validator(mode="after")
    def reason_must_be_meaningful(self) -> "AuditEntry":
        """Safety rule: every change must carry a non-blank reason."""
        if not self.reason.strip():
            raise ValueError("An audit entry must include a non-empty reason.")
        return self
