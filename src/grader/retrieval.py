"""Retrieval / RAG - the Clarify feature (Component 5).

This powers "why did I get this mark?". It RETRIEVES the real facts about a
grade (the student's answer, the question, the rubric, the AI's justification)
and then GENERATES a plain-language explanation grounded ONLY in those facts.
That is RAG: retrieve first, then generate from what was retrieved - so the
explanation cannot hallucinate.

Two retrieval layers:
  1. DIRECT retrieval - for "why this mark on Q4?", we know exactly which grade
     is meant (student_id + question_id), so we fetch the exact record. Precise
     and free, no embeddings needed.
  2. VECTOR SEARCH - for fuzzier questions ("why did I lose marks on the
     photosynthesis questions?"), we find relevant answers by MEANING using
     embeddings + a vector store. Built behind an interface so the local mock
     swaps for Databricks Vector Search later.
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from grader.store import Store


# ===========================================================================
# LAYER 1 - DIRECT RETRIEVAL: fetch the exact facts for one grade.
# ===========================================================================

@dataclass
class ClarifyContext:
    """The bundle of real facts used to explain one grade."""

    student_id: str
    question_id: str
    question_text: str
    rubric: str
    student_answer: str
    marks_awarded: float
    max_marks: float
    justification: str

    def as_facts(self) -> str:
        """Format the facts as a short block for an AI prompt or display."""
        return (
            f"Question: {self.question_text}\n"
            f"Rubric: {self.rubric or '(none)'}\n"
            f"Student answer: {self.student_answer}\n"
            f"Marks: {self.marks_awarded} / {self.max_marks}\n"
            f"Grader's reason: {self.justification}"
        )


def retrieve_context(store: Store, student_id: str, question_id: str) -> ClarifyContext | None:
    """Fetch the exact facts needed to explain one student's grade on one question.

    Returns None if there is no grade for that student+question.
    """
    grade = store.get_grade(student_id, question_id)
    if grade is None:
        return None
    question = store.get_question(question_id)

    # the student's answer text is stored inside the answers table; fetch it
    row = store.conn.execute(
        "SELECT data FROM answers WHERE student_id = ? AND question_id = ?",
        (student_id, question_id),
    ).fetchone()
    answer_text = json.loads(row["data"])["answer_text"] if row else "(answer not found)"

    return ClarifyContext(
        student_id=student_id,
        question_id=question_id,
        question_text=question.text if question else "(question not found)",
        rubric=(question.rubric if question else "") or "",
        student_answer=answer_text,
        marks_awarded=grade.marks_awarded,
        max_marks=grade.max_marks,
        justification=grade.justification,
    )


# ===========================================================================
# LAYER 2 - VECTOR SEARCH: find relevant answers by MEANING.
# An embedder turns text into numbers; the vector store finds nearest ones.
# Both are behind interfaces so the local mock swaps for Databricks later.
# ===========================================================================

class Embedder(ABC):
    """Interface: turn a piece of text into a vector (list of numbers)."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...


class MockEmbedder(Embedder):
    """A tiny, deterministic embedder - free and offline.

    It is NOT real semantic embedding; it turns text into a small fixed-size
    vector from word hashes, so similar wording lands somewhat nearby. Good
    enough to build and test the vector-store machinery without any API.
    """

    def __init__(self, dims: int = 16):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in text.lower().split():
            # hash each word into a bucket and add 1 there
            bucket = hash(word) % self.dims
            vec[bucket] += 1.0
        # normalise to unit length so comparisons are fair
        length = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / length for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: 1.0 = same direction (very similar), 0 = unrelated."""
    return sum(x * y for x, y in zip(a, b))


@dataclass
class ScoredHit:
    """One search result: the stored item and how similar it was."""

    key: str
    text: str
    score: float


class VectorStore:
    """A minimal in-memory vector store: add texts, then find nearest by meaning.

    This mirrors what Databricks Vector Search does; later we swap this class
    for a Databricks-backed one behind the same add()/search() methods.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._items: list[tuple[str, str, list[float]]] = []  # (key, text, vector)

    def add(self, key: str, text: str) -> None:
        """Store a piece of text (embedded into a vector)."""
        self._items.append((key, text, self.embedder.embed(text)))

    def search(self, query: str, top_k: int = 3) -> list[ScoredHit]:
        """Return the top_k stored items most similar in MEANING to the query."""
        q = self.embedder.embed(query)
        scored = [
            ScoredHit(key=key, text=text, score=_cosine(q, vec))
            for (key, text, vec) in self._items
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


def build_answer_index(store: Store, embedder: Embedder | None = None) -> VectorStore:
    """Build a vector index over all stored answers, for fuzzy clarify queries."""
    if embedder is None:
        embedder = MockEmbedder()
    vs = VectorStore(embedder)
    rows = store.conn.execute(
        "SELECT id, student_id, question_id, data FROM answers"
    ).fetchall()
    for r in rows:
        text = json.loads(r["data"])["answer_text"]
        key = f"{r['student_id']}|{r['question_id']}"
        vs.add(key, text)
    return vs


# ===========================================================================
# THE CLARIFY EXPLAINER: turn retrieved facts into a plain-language answer.
# Mock by default (free); real AI when a key is set.
# ===========================================================================

class ClarifyExplainer(ABC):
    """Interface: explain a grade, given the retrieved facts."""

    @abstractmethod
    def explain(self, context: ClarifyContext, student_question: str) -> str:
        ...


class MockClarifyExplainer(ClarifyExplainer):
    """A free, offline explainer that assembles a clear answer from the facts.

    It does no real reasoning - it templates the retrieved facts into a helpful
    explanation, so the whole feature works and tests without any API.
    """

    def explain(self, context: ClarifyContext, student_question: str) -> str:
        got_full = context.marks_awarded >= context.max_marks
        lead = "You received full marks" if got_full else "You lost some marks"
        return (
            f"{lead} on this question ({context.marks_awarded}/{context.max_marks}). "
            f"The grader's reason was: {context.justification} "
            f"Your answer was: \"{context.student_answer}\". "
            f"The marking guide was: {context.rubric or '(none provided)'}."
        )


class OpenRouterClarifyExplainer(ClarifyExplainer):
    """Explains a grade using a real model via OpenRouter, grounded in the facts."""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def explain(self, context: ClarifyContext, student_question: str) -> str:
        import requests

        system = (
            "You explain exam marks to students kindly and clearly. "
            "Use ONLY the facts provided - do not invent anything. "
            "Explain why the marks were given and how the answer could improve."
        )
        user = (
            f"The student asks: {student_question}\n\n"
            f"Here are the facts:\n{context.as_facts()}\n\n"
            f"Explain the mark using only these facts."
        )
        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            }),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def get_clarify_explainer() -> ClarifyExplainer:
    """Real explainer if a key is configured, else the free mock."""
    from grader.config import settings
    if settings.has_api_key():
        return OpenRouterClarifyExplainer(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockClarifyExplainer()


def clarify(
    store: Store,
    student_id: str,
    question_id: str,
    student_question: str = "Why did I get this mark?",
    explainer: ClarifyExplainer | None = None,
) -> str:
    """The main Clarify entry point: retrieve the facts, then explain the mark.

    Returns a plain-language explanation, or a clear message if no grade exists.
    """
    context = retrieve_context(store, student_id, question_id)
    if context is None:
        return (f"No grade found for student '{student_id}' on question "
                f"'{question_id}', so there is nothing to explain yet.")
    if explainer is None:
        explainer = get_clarify_explainer()
    return explainer.explain(context, student_question)