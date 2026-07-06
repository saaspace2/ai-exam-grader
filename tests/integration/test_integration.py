"""Integration Tests - do multiple components work TOGETHER?

Analogy: you tested the engine, gearbox, wheels alone (unit tests). Now drive
the car. Here: question -> answer -> grade -> store -> read back, end to end.
Uses the local SQLite store (offline), so it runs anywhere.
"""

import os
import tempfile

import pytest

from grader.models import Question, StudentAnswer
from grader.grading import grade_answer
from grader.store import Store


@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    s = Store(os.path.join(d, "i.sqlite"))
    yield s
    s.close()


class TestFullFlow:
    def test_question_to_grade_to_store(self, store):
        # 1. save a question, 2. student answers, 3. grade, 4. store, 5. read back
        q = Question(id="Q1", type="mcq", text="Capital of France?",
                     correct_answer="Paris", max_marks=2)
        store.save_question(q)

        a = StudentAnswer(id="A1", question_id="Q1", student_id="Riya",
                          answer_text="Paris")
        store.save_answer(a)

        record = grade_answer(q, a)
        store.save_grade(record)

        # read it back - the whole pipeline worked if this matches
        got = store.get_grade("Riya", "Q1")
        assert got is not None
        assert got.marks_awarded == 2.0

    def test_audit_trail_created(self, store):
        # grading should auto-create an audit entry (integration of grade + audit)
        q = Question(id="Q2", type="mcq", text="?", correct_answer="A", max_marks=1)
        store.save_question(q)
        a = StudentAnswer(id="A2", question_id="Q2", student_id="S", answer_text="A")
        store.save_answer(a)
        rec = grade_answer(q, a)
        store.save_grade(rec)
        history = store.get_audit_history(rec.id)
        assert len(history) >= 1, "No audit entry created on grade."