"""Tests for the batch grader (using the SQLite store, which shares the interface)."""

import os
import tempfile

import pytest

from grader.batch_grader import BatchGrader
from grader.models import Question, StudentAnswer
from grader.store import Store


@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    s = Store(os.path.join(d, "b.sqlite"))
    yield s
    s.close()


def _seed(store):
    for qid, ans in [("Q1", "Paris"), ("Q2", "Tokyo")]:
        store.save_question(Question(id=qid, type="mcq", text="q",
                                     correct_answer=ans, max_marks=2))
    store.save_answer(StudentAnswer(id="A_R_Q1", question_id="Q1",
                                    student_id="R", answer_text="Paris"))   # right
    store.save_answer(StudentAnswer(id="A_R_Q2", question_id="Q2",
                                    student_id="R", answer_text="Osaka"))   # wrong


class TestBatchGrader:
    def test_grades_all_ungraded(self, store):
        _seed(store)
        summary = BatchGrader(store).grade_all()
        assert summary["graded"] == 2
        assert summary["total_marks"] == 2.0    # Q1 right (2), Q2 wrong (0)
        assert summary["out_of"] == 4.0

    def test_is_idempotent(self, store):
        _seed(store)
        BatchGrader(store).grade_all()
        again = BatchGrader(store).grade_all()
        assert again["graded"] == 0             # nothing new to grade

    def test_writes_predictions(self, store):
        _seed(store)
        BatchGrader(store).grade_all()
        assert len(store.list_predictions()) == 2