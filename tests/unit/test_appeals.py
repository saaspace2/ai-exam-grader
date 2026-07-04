"""Unit tests for appeals / re-evaluation (Component 6).

We use a controllable stub grader so 'mark goes up' and 'mark goes down' are
deterministic, and an in-memory store so everything is free and offline.
"""

import pytest

from grader.appeals import (
    ReevalResult,
    confirm_reevaluation,
    raise_doubt,
    reevaluate,
)
from grader.grading import AIGrader
from grader.models import (
    GradeRecord,
    GradeStatus,
    Question,
    StudentAnswer,
)
from grader.store import Store


class StubGrader(AIGrader):
    """A grader that always returns a fixed mark - so tests are deterministic."""

    def __init__(self, marks):
        self.marks = marks

    def grade_text(self, question_text, rubric, answer_text, max_marks):
        return (self.marks, f"Stub gave {self.marks}.", 0.9)


@pytest.fixture
def store():
    s = Store(":memory:")
    q = Question(id="Q4", type="essay", text="Explain photosynthesis.",
                 rubric="sunlight water glucose oxygen", max_marks=10)
    a = StudentAnswer(id="A4", question_id="Q4", student_id="Riya",
                      answer_text="Plants use sunlight and water to make glucose.")
    s.save_question(q)
    s.save_answer(a)
    # seed an initial grade of 6.0
    s.save_grade(GradeRecord(id="G4", answer_id="A4", question_id="Q4",
                             student_id="Riya", marks_awarded=6.0, max_marks=10,
                             justification="Initial.", confidence=0.8,
                             grading_method="ai-essay"))
    yield s
    s.close()


class TestRaiseDoubt:
    def test_raise_doubt_sets_status_and_logs(self, store):
        entry = raise_doubt(store, "Riya", "Q4", reason="I covered more")
        assert entry.action == "doubt_raised"
        # status moved to under_appeal
        assert store.get_grade("Riya", "Q4").status == GradeStatus.UNDER_APPEAL
        # audit slip present
        hist = store.get_audit_history("G4")
        assert any(h.action == "doubt_raised" for h in hist)

    def test_raise_doubt_without_grade_raises(self, store):
        with pytest.raises(ValueError):
            raise_doubt(store, "Nobody", "Q4")


class TestReevaluate:
    def test_higher_mark_is_applied_immediately(self, store):
        raise_doubt(store, "Riya", "Q4")
        res = reevaluate(store, "Riya", "Q4", ai=StubGrader(9.0))
        assert res.applied is True
        assert res.needs_confirmation is False
        assert res.new_marks == 9.0
        assert store.get_grade("Riya", "Q4").marks_awarded == 9.0

    def test_same_mark_is_applied_and_confirmed(self, store):
        raise_doubt(store, "Riya", "Q4")
        res = reevaluate(store, "Riya", "Q4", ai=StubGrader(6.0))
        assert res.applied is True
        assert res.changed is False

    def test_lower_mark_is_held_for_confirmation(self, store):
        raise_doubt(store, "Riya", "Q4")
        res = reevaluate(store, "Riya", "Q4", ai=StubGrader(3.0))
        assert res.applied is False
        assert res.needs_confirmation is True
        assert res.new_marks == 3.0
        # the stored mark is UNCHANGED until confirmation
        assert store.get_grade("Riya", "Q4").marks_awarded == 6.0
        assert "LOWER" in res.message


class TestConfirmLower:
    def test_confirming_applies_the_lower_mark(self, store):
        raise_doubt(store, "Riya", "Q4")
        res = reevaluate(store, "Riya", "Q4", ai=StubGrader(3.0))
        assert res.needs_confirmation is True
        confirm = confirm_reevaluation(store, "Riya", "Q4",
                                       new_marks=res.new_marks,
                                       justification="Accepted.")
        assert confirm.applied is True
        assert store.get_grade("Riya", "Q4").marks_awarded == 3.0


class TestAuditTrailComplete:
    def test_full_history_is_recorded(self, store):
        raise_doubt(store, "Riya", "Q4", reason="please check")
        reevaluate(store, "Riya", "Q4", ai=StubGrader(8.0))
        actions = [h.action for h in store.get_audit_history("G4")]
        assert actions == ["initial_grade", "doubt_raised", "reevaluated"]