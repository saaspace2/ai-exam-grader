"""Unit tests for retrieval / RAG - the Clarify feature (Component 5).

Uses an in-memory store and the mock explainer/embedder, so everything is
free, offline, and deterministic.
"""

import pytest

from grader.models import GradeRecord, Question, StudentAnswer
from grader.retrieval import (
    MockClarifyExplainer,
    MockEmbedder,
    VectorStore,
    build_answer_index,
    clarify,
    retrieve_context,
)
from grader.store import Store


@pytest.fixture
def graded_store():
    """A store with one graded essay answer for Riya on Q4."""
    s = Store(":memory:")
    s.save_question(Question(id="Q4", type="essay", text="Explain photosynthesis.",
                             rubric="sunlight water glucose oxygen", max_marks=10))
    s.save_answer(StudentAnswer(id="A4", question_id="Q4", student_id="Riya",
                                answer_text="Plants use sunlight and water to make glucose."))
    s.save_grade(GradeRecord(id="G4", answer_id="A4", question_id="Q4",
                             student_id="Riya", marks_awarded=6.7, max_marks=10,
                             justification="Matched 3 of 4 keywords.",
                             confidence=0.7, grading_method="ai-essay"))
    yield s
    s.close()


class TestDirectRetrieval:
    def test_retrieve_context_has_the_facts(self, graded_store):
        ctx = retrieve_context(graded_store, "Riya", "Q4")
        assert ctx is not None
        assert ctx.marks_awarded == 6.7
        assert ctx.max_marks == 10
        assert "photosynthesis" in ctx.question_text.lower()
        assert "sunlight" in ctx.student_answer.lower()

    def test_retrieve_missing_grade_returns_none(self, graded_store):
        assert retrieve_context(graded_store, "Nobody", "Q4") is None


class TestClarify:
    def test_clarify_explains_using_real_facts(self, graded_store):
        """The mock explainer grounds its answer in the retrieved facts."""
        out = clarify(graded_store, "Riya", "Q4", "why this mark?", explainer=MockClarifyExplainer())
        assert "6.7" in out                 # the real marks
        assert "10" in out                  # out of
        assert "keywords" in out.lower()    # the grader's reason

    def test_clarify_when_no_grade(self, graded_store):
        out = clarify(graded_store, "Ghost", "Q4", explainer=MockClarifyExplainer())
        assert "no grade found" in out.lower()


class TestVectorSearch:
    def test_embedder_gives_fixed_length_vector(self):
        v = MockEmbedder(dims=16).embed("hello world")
        assert len(v) == 16

    def test_store_finds_nearest_by_meaning(self):
        vs = VectorStore(MockEmbedder())
        vs.add("photo", "plants use sunlight and water to make glucose")
        vs.add("gravity", "a force of attraction between masses")
        hits = vs.search("sunlight water glucose plants", top_k=2)
        # the photosynthesis text should rank first
        assert hits[0].key == "photo"
        assert hits[0].score >= hits[1].score   # sorted by similarity

    def test_build_index_over_stored_answers(self, graded_store):
        index = build_answer_index(graded_store)
        hits = index.search("photosynthesis sunlight", top_k=1)
        assert len(hits) == 1
        assert hits[0].key == "Riya|Q4"