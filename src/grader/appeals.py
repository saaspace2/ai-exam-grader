"""Appeals / Re-evaluation - the fairness loop (Component 6).

The flow:
  1. raise_doubt()  - a student flags a graded question (reason optional).
                      The grade's status becomes 'under_appeal' and a
                      'doubt_raised' slip is written to the audit trail.
  2. reevaluate()   - the AI re-grades the answer, considering the student's
                      reason. If the new mark would be LOWER, it is NOT applied
                      yet: a warning is returned with old vs proposed marks, and
                      the student must confirm. If the mark stays or rises, it is
                      applied immediately.
  3. confirm_reevaluation() - applies a held (lower) mark once the student
                      confirms. Either way, an audit slip records the change and
                      a clarification explains the result.

Every change appends to the append-only audit trail from Component 4, so the
full history of any grade is always provable.
"""

from dataclasses import dataclass

from grader.grading import AIGrader, grade_answer
from grader.models import (
    Actor,
    AuditEntry,
    GradeRecord,
    GradeStatus,
    Question,
    StudentAnswer,
)
from grader.retrieval import ClarifyExplainer, retrieve_context
from grader.store import Store


# ---------------------------------------------------------------------------
# The result of a re-evaluation: either applied, or held pending confirmation.
# ---------------------------------------------------------------------------

@dataclass
class ReevalResult:
    """What happened when we re-evaluated a grade."""

    student_id: str
    question_id: str
    old_marks: float
    new_marks: float
    applied: bool          # True if the new mark was applied now
    needs_confirmation: bool  # True if held because the mark would DROP
    message: str           # a plain-language explanation / warning

    @property
    def changed(self) -> bool:
        return self.new_marks != self.old_marks


# ---------------------------------------------------------------------------
# Step 1 - raise a doubt.
# ---------------------------------------------------------------------------

def raise_doubt(store: Store, student_id: str, question_id: str,
                reason: str | None = None) -> AuditEntry:
    """Flag a graded question for re-evaluation.

    Sets the grade status to 'under_appeal' and logs a 'doubt_raised' audit
    slip. Returns that slip. Raises if there is no grade to dispute.
    """
    grade = store.get_grade(student_id, question_id)
    if grade is None:
        raise ValueError(
            f"No grade for student '{student_id}' on question '{question_id}'."
        )

    # move the grade into the 'under_appeal' state
    grade.status = GradeStatus.UNDER_APPEAL
    store.update_grade(grade)

    entry = AuditEntry(
        id=f"AU_{grade.id}_doubt",
        grade_record_id=grade.id,
        actor=Actor.SYSTEM,
        action="doubt_raised",
        old_marks=grade.marks_awarded,
        new_marks=grade.marks_awarded,   # a doubt does not change marks yet
        reason=(reason.strip() if reason and reason.strip()
                else "Student raised a doubt (no reason given)."),
    )
    store.append_audit(entry)
    return entry


# ---------------------------------------------------------------------------
# Step 2 - re-evaluate.
# ---------------------------------------------------------------------------

def reevaluate(store: Store, student_id: str, question_id: str,
               student_reason: str | None = None,
               ai: AIGrader | None = None) -> ReevalResult:
    """Re-grade a disputed answer. Apply immediately unless the mark would drop.

    If the new mark is LOWER than the old one, we do NOT apply it - we return a
    result flagged needs_confirmation so the student can be warned first.
    """
    grade = store.get_grade(student_id, question_id)
    if grade is None:
        raise ValueError(
            f"No grade for student '{student_id}' on question '{question_id}'."
        )
    question = store.get_question(question_id)

    # rebuild the StudentAnswer from storage to re-grade it
    import json
    row = store.conn.execute(
        "SELECT data FROM answers WHERE student_id = ? AND question_id = ?",
        (student_id, question_id),
    ).fetchone()
    answer_data = json.loads(row["data"])
    answer = StudentAnswer(**answer_data)

    # if the student gave a reason, append it to the rubric so the AI weighs it
    q_for_regrade = question
    if student_reason and question is not None:
        q_for_regrade = question.model_copy()
        extra = f"\nStudent's argument to consider: {student_reason}"
        q_for_regrade.rubric = (question.rubric or "") + extra

    fresh = grade_answer(q_for_regrade, answer, ai=ai)
    old_marks = grade.marks_awarded
    new_marks = fresh.marks_awarded

    # DECISION: a drop is held for confirmation; same-or-higher applies now.
    if new_marks < old_marks:
        return ReevalResult(
            student_id=student_id, question_id=question_id,
            old_marks=old_marks, new_marks=new_marks,
            applied=False, needs_confirmation=True,
            message=(
                f"Heads up: re-evaluation would LOWER your mark from "
                f"{old_marks} to {new_marks}. It has NOT been changed. "
                f"Confirm only if you want to proceed. Reason: {fresh.justification}"
            ),
        )

    # same or higher -> apply immediately
    _apply_reeval(store, grade, fresh, actor=Actor.AI,
                  reason=f"Re-evaluation: {fresh.justification}")
    verb = "raised" if new_marks > old_marks else "confirmed (unchanged)"
    return ReevalResult(
        student_id=student_id, question_id=question_id,
        old_marks=old_marks, new_marks=new_marks,
        applied=True, needs_confirmation=False,
        message=(f"Your mark was {verb}: {old_marks} -> {new_marks}. "
                 f"Reason: {fresh.justification}"),
    )


def confirm_reevaluation(store: Store, student_id: str, question_id: str,
                         new_marks: float, justification: str) -> ReevalResult:
    """Apply a previously-held (lower) mark after the student confirms."""
    grade = store.get_grade(student_id, question_id)
    if grade is None:
        raise ValueError("No grade to confirm.")
    old_marks = grade.marks_awarded

    updated = grade.model_copy()
    updated.marks_awarded = new_marks
    _apply_reeval(store, grade, updated, actor=Actor.AI,
                  reason=f"Student confirmed lower mark. {justification}")
    return ReevalResult(
        student_id=student_id, question_id=question_id,
        old_marks=old_marks, new_marks=new_marks,
        applied=True, needs_confirmation=False,
        message=f"Confirmed. Your mark is now {new_marks}.",
    )


# ---------------------------------------------------------------------------
# Step 3 - the clarification of the result.
# ---------------------------------------------------------------------------

def clarify_result(store: Store, student_id: str, question_id: str,
                   explainer: ClarifyExplainer | None = None) -> str:
    """Explain the CURRENT (possibly re-evaluated) mark, grounded in the facts."""
    from grader.retrieval import clarify
    return clarify(store, student_id, question_id,
                   student_question="Please explain my current mark after re-evaluation.",
                   explainer=explainer)


# ---------------------------------------------------------------------------
# Internal helper: apply a new grade + log the audit slip.
# ---------------------------------------------------------------------------

def _apply_reeval(store: Store, old_grade: GradeRecord, new_grade: GradeRecord,
                  actor: Actor, reason: str) -> None:
    """Write the new marks + status to storage and append an audit slip.

    We update the ALREADY-STORED grade (old_grade.id), copying the new marks and
    justification onto it, so the UPDATE targets the real row. The fresh grade
    from re-grading may have a different id, so we do not store it directly.
    """
    to_store = old_grade.model_copy()
    to_store.marks_awarded = new_grade.marks_awarded
    to_store.justification = new_grade.justification
    to_store.confidence = new_grade.confidence
    to_store.status = GradeStatus.REGRADED
    store.update_grade(to_store)

    store.append_audit(AuditEntry(
        id=f"AU_{old_grade.id}_reeval_{int(new_grade.marks_awarded*10)}",
        grade_record_id=old_grade.id,
        actor=actor,
        action="reevaluated",
        old_marks=old_grade.marks_awarded,
        new_marks=new_grade.marks_awarded,
        reason=reason,
    ))