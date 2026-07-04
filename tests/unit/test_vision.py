"""Unit tests for the vision/OCR reader (Component 3.7).

All tests use the MockImageReader - free, offline, deterministic. No network
or API key is ever needed. We prove the reader produces an ingestion-ready
dict and that a missing file is caught.
"""

import importlib
from pathlib import Path

import pytest

from grader.vision import (
    MockImageReader,
    ReadResult,
    read_answer_from_image,
)


@pytest.fixture
def fake_image(tmp_path):
    """Create a dummy image file on disk and return its path."""
    p = tmp_path / "answer.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    return str(p)


class TestMockReader:
    """The mock reader returns preset values without touching a model."""

    def test_read_returns_a_ReadResult(self, fake_image):
        reader = MockImageReader(student_id="Sam", answer_text="42", confidence=0.8)
        result = reader.read(fake_image)
        assert isinstance(result, ReadResult)
        assert result.student_id == "Sam"
        assert result.answer_text == "42"
        assert result.confidence == 0.8

    def test_missing_file_raises(self):
        reader = MockImageReader()
        with pytest.raises(FileNotFoundError):
            reader.read("/no/such/image.png")


class TestReadToDict:
    """read_answer_from_image produces a dict shaped for ingestion."""

    def test_produces_ingestion_ready_dict(self, fake_image):
        reader = MockImageReader(student_id="Riya", answer_text="Paris")
        raw = read_answer_from_image(fake_image, question_id="Q1",
                                     answer_id="A1", reader=reader)
        # core fields must match (an internal _question_id_source may also be present)
        assert raw["id"] == "A1"
        assert raw["question_id"] == "Q1"          # mock has no page qid -> fallback
        assert raw["student_id"] == "Riya"
        assert raw["answer_text"] == "Paris"

    def test_blank_student_id_becomes_unknown(self, fake_image):
        """If the reader finds no ID, we fall back to 'UNKNOWN' (not blank)."""
        reader = MockImageReader(student_id="", answer_text="something")
        raw = read_answer_from_image(fake_image, question_id="Q2",
                                     answer_id="A2", reader=reader)
        assert raw["student_id"] == "UNKNOWN"

    def test_read_dict_can_be_ingested_and_graded(self, fake_image):
        """The read dict flows into ingest + grade with no glue code."""
        from grader.ingestion import ingest_answer, ingest_question
        from grader.grading import grade_answer, MockAIGrader

        reader = MockImageReader(student_id="Riya", answer_text="Paris")
        raw = read_answer_from_image(fake_image, question_id="Q1",
                                     answer_id="A1", reader=reader)
        q = ingest_question({"id": "Q1", "type": "mcq", "text": "Capital?",
                             "correct_answer": "Paris", "max_marks": 2})
        a = ingest_answer(raw)
        g = grade_answer(q, a, ai=MockAIGrader())
        assert g.marks_awarded == 2
        assert g.student_id == "Riya"


class TestReaderSelection:
    """get_image_reader picks mock or real based on whether a key is set."""

    def _reload(self, monkeypatch, key_value):
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        if key_value is None:
            monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        else:
            monkeypatch.setenv("OPENROUTER_API_KEY", key_value)
        import grader.config as cfg
        importlib.reload(cfg)
        import grader.vision as vision
        importlib.reload(vision)
        return vision

    def test_returns_mock_when_no_key(self, monkeypatch):
        vision = self._reload(monkeypatch, None)
        assert isinstance(vision.get_image_reader(), vision.MockImageReader)

    def test_returns_openrouter_when_key_set(self, monkeypatch):
        vision = self._reload(monkeypatch, "test-key-123")
        assert isinstance(vision.get_image_reader(), vision.OpenRouterImageReader)