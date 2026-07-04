"""Unit tests for storage + audit trail (Component 4).

Every test uses an in-memory database (":memory:") so it is fast and never
touches real data. We prove: save/fetch works, both mix-up safeguards fire,
and the audit trail is append-only and ordered.
"""

import pytest

from grader.models import (
    Actor,
    AuditEntry,
    GradeRecord,
    Question,
    StudentAnswer,
)
from grader.store import Store


@pytest.fixture
def store():
    """A fresh in-memory store for each test."""
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    """A store with one question already saved."""
    store.save_question(
        Question(id="Q1", type="mcq", text="Capital of France?",
                 correct_answer="Paris", max_marks=2)
    )
    return store


def _grade(**kw):
    base = dict(id="G1", answer_id="A1", question_id="Q1", student_id="Riya",
                marks_awarded=2, max_marks=2, justification="Correct.",
                confidence=1.0, grading_method="mcq-exact")
    base.update(kw)
    return GradeRecord(**base)


class TestSaveAndFetch:
    def test_save_and_get_question(self, store):
        q = Question(id="Q1", type="mcq", text="?", correct_answer="a", max_marks=1)
        store.save_question(q)
        assert store.get_question("Q1").id == "Q1"

    def test_get_missing_question_returns_none(self, store):
        assert store.get_question("nope") is None

    def test_save_and_get_grade(self, seeded):
        seeded.save_answer(StudentAnswer(id="A1", question_id="Q1",
                                         student_id="Riya", answer_text="Paris"))
        seeded.save_grade(_grade())
        g = seeded.get_grade("Riya", "Q1")
        assert g.marks_awarded == 2


class TestSafeguards:
    def test_answer_for_missing_question_is_blocked(self, store):
        """SAFEGUARD: cannot save an answer whose question is not stored."""
        with pytest.raises(ValueError):
            store.save_answer(StudentAnswer(id="A1", question_id="GHOST",
                                            student_id="Sam", answer_text="x"))

    def test_duplicate_answer_is_blocked(self, seeded):
        """SAFEGUARD: one answer per (student, question)."""
        seeded.save_answer(StudentAnswer(id="A1", question_id="Q1",
                                         student_id="Riya", answer_text="Paris"))
        with pytest.raises(ValueError):
            seeded.save_answer(StudentAnswer(id="A2", question_id="Q1",
                                             student_id="Riya", answer_text="again"))

    def test_duplicate_grade_is_blocked(self, seeded):
        """SAFEGUARD: one grade per (student, question)."""
        seeded.save_answer(StudentAnswer(id="A1", question_id="Q1",
                                         student_id="Riya", answer_text="Paris"))
        seeded.save_grade(_grade())
        with pytest.raises(ValueError):
            seeded.save_grade(_grade(id="G2"))

    def test_different_students_can_answer_same_question(self, seeded):
        """Two DIFFERENT students CAN answer the same question."""
        seeded.save_answer(StudentAnswer(id="A1", question_id="Q1",
                                         student_id="Riya", answer_text="Paris"))
        seeded.save_answer(StudentAnswer(id="A2", question_id="Q1",
                                         student_id="Sam", answer_text="Paris"))
        # no error = success


class TestAuditTrail:
    def test_saving_a_grade_writes_first_audit_slip(self, seeded):
        seeded.save_answer(StudentAnswer(id="A1", question_id="Q1",
                                         student_id="Riya", answer_text="Paris"))
        seeded.save_grade(_grade())
        history = seeded.get_audit_history("G1")
        assert len(history) == 1
        assert history[0].action == "initial_grade"
        assert history[0].old_marks is None
        assert history[0].new_marks == 2

    def test_appended_slips_are_kept_in_order(self, seeded):
        seeded.save_answer(StudentAnswer(id="A1", question_id="Q1",
                                         student_id="Riya", answer_text="Paris"))
        seeded.save_grade(_grade())
        seeded.append_audit(AuditEntry(
            id="AU_G1_2", grade_record_id="G1", actor=Actor.TEACHER,
            action="appeal_regrade", old_marks=2, new_marks=1,
            reason="Deducted on review.",
        ))
        history = seeded.get_audit_history("G1")
        assert len(history) == 2
        assert history[0].action == "initial_grade"     # first
        assert history[1].action == "appeal_regrade"    # second, in order
        assert history[1].new_marks == 1


class TestPersistence:
    def test_data_survives_reopen(self, tmp_path):
        """Closing and reopening the file keeps the data (real persistence)."""
        db = str(tmp_path / "t.sqlite")
        s1 = Store(db)
        s1.save_question(Question(id="Q1", type="mcq", text="?",
                                  correct_answer="a", max_marks=1))
        s1.close()
        s2 = Store(db)
        assert s2.get_question("Q1") is not None
        s2.close()