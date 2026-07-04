"""Unit tests for ingestion (Component 2).

These prove three things:
  1. Good raw data becomes valid objects (the happy path).
  2. Bad raw data is caught and REPORTED, not crashed on.
  3. A batch with a mix keeps the good records and collects the bad ones.
"""

from grader.ingestion import (
    IngestReport,
    ingest_answer,
    ingest_batch,
    ingest_question,
)
from grader.models import Question, StudentAnswer

import pytest
from pydantic import ValidationError


class TestSingleIngest:
    """Ingesting one record at a time."""

    def test_good_question_becomes_a_Question(self):
        """A valid raw dict turns into a real Question object."""
        raw = {"id": "Q1", "type": "mcq", "text": "2+2?",
               "correct_answer": "4", "max_marks": 5}
        q = ingest_question(raw)
        assert isinstance(q, Question)   # we got a real Question
        assert q.id == "Q1"

    def test_good_answer_becomes_a_StudentAnswer(self):
        """A valid raw dict turns into a real StudentAnswer object."""
        raw = {"id": "A1", "question_id": "Q1",
               "student_id": "Riya", "answer_text": "4"}
        a = ingest_answer(raw)
        assert isinstance(a, StudentAnswer)
        assert a.student_id == "Riya"

    def test_bad_question_raises(self):
        """A single bad record raises (the caller decides what to do)."""
        raw = {"id": "Qx", "type": "mcq", "text": "bad", "max_marks": -5}
        with pytest.raises(ValidationError):
            ingest_question(raw)


class TestBatchIngest:
    """Ingesting a whole pile at once."""

    def test_all_good_batch(self):
        """A clean pile: everything succeeds, nothing fails."""
        report = ingest_batch(
            raw_questions=[{"id": "Q1", "type": "mcq", "text": "2+2?", "max_marks": 5}],
            raw_answers=[{"id": "A1", "question_id": "Q1",
                          "student_id": "Riya", "answer_text": "4"}],
        )
        assert report.ok is True          # nothing failed
        assert len(report.questions) == 1
        assert len(report.answers) == 1
        assert len(report.failures) == 0

    def test_mixed_batch_keeps_good_and_collects_bad(self):
        """One bad record must NOT stop the good ones."""
        report = ingest_batch(
            raw_questions=[
                {"id": "Q1", "type": "mcq", "text": "good", "max_marks": 5},
                {"id": "Q2", "type": "mcq", "text": "bad", "max_marks": -5},  # bad
            ],
        )
        assert len(report.questions) == 1   # the good one got through
        assert len(report.failures) == 1    # the bad one was recorded
        assert report.ok is False
        # the failure report tells us what and why
        assert report.failures[0]["kind"] == "question"
        assert report.failures[0]["raw"]["id"] == "Q2"

    def test_empty_batch_is_fine(self):
        """Ingesting nothing is not an error - you just get an empty report."""
        report = ingest_batch()
        assert report.ok is True
        assert report.summary() == "0 questions, 0 answers, 0 failed."
