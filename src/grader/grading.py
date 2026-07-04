"""Grading - the 'brain' of the pipeline (Component 3).

This is STEP 2 of the pipeline. It takes a Question and a StudentAnswer and
produces a GradeRecord (marks + justification + confidence).

The core idea: ONE grading strategy per question type.
    MCQ      -> exact match          (simple logic, no AI)
    NUMERIC  -> within tolerance      (simple logic, no AI)
    SHORT    -> judged against rubric (needs AI)
    ESSAY    -> judged against rubric (needs AI)

For the AI-judged types we use a GRADER INTERFACE with two implementations:
  - MockAIGrader  : returns a sensible fake grade instantly, free, no API key.
  - (later) ClaudeAIGrader : calls the real Claude API.
Because both follow the same interface, we can swap one for the other with a
single change - the 'mock now, swap real later' discipline.
"""

import json
from abc import ABC, abstractmethod

from grader.models import GradeRecord, Question, QuestionType, StudentAnswer


# ---------------------------------------------------------------------------
# The AI grader "interface". Any AI grader must provide a grade_text() method.
# This is like a job description: it says WHAT an AI grader must be able to do,
# without saying HOW. The mock and the real Claude grader each fill it in.
# ---------------------------------------------------------------------------

class AIGrader(ABC):
    """Interface for an AI that grades free-text answers against a rubric."""

    @abstractmethod
    def grade_text(
        self, question_text: str, rubric: str, answer_text: str, max_marks: float
    ) -> tuple[float, str, float]:
        """Return (marks, justification, confidence) for one free-text answer."""
        ...


class MockAIGrader(AIGrader):
    """A stand-in AI grader for building and testing - no real API calls.

    It uses a simple, predictable rule so tests are deterministic: it awards
    marks based on how many rubric keywords appear in the answer. This is NOT
    real intelligence - it just behaves consistently so we can build the whole
    pipeline before wiring in the real Claude API.
    """

    def grade_text(self, question_text, rubric, answer_text, max_marks):
        # Split the rubric into keywords (very simple: words over 3 letters).
        keywords = [w.strip(".,").lower() for w in rubric.split() if len(w) > 3]
        if not keywords:
            # No rubric keywords to check -> award half marks, low confidence.
            return (max_marks / 2, "No rubric keywords to match on.", 0.3)

        answer_low = answer_text.lower()
        hits = [k for k in keywords if k in answer_low]
        fraction = len(hits) / len(keywords)
        marks = round(fraction * max_marks, 1)
        justification = (
            f"Matched {len(hits)} of {len(keywords)} rubric keywords "
            f"({', '.join(hits) if hits else 'none'})."
        )
        # Confidence is higher when the match is clearly high or clearly low.
        confidence = 0.6 + 0.3 * abs(fraction - 0.5) * 2
        return (marks, justification, round(min(confidence, 0.95), 2))


# ---------------------------------------------------------------------------
# The per-type grading strategies. Each returns (marks, justification,
# confidence, method_label).
# ---------------------------------------------------------------------------

def _grade_mcq(question: Question, answer: StudentAnswer):
    """MCQ: full marks for an exact (case-insensitive) match, else zero."""
    correct = (question.correct_answer or "").strip().lower()
    given = answer.answer_text.strip().lower()
    if given == correct:
        return (question.max_marks, f"Correct: '{answer.answer_text}'.", 1.0, "mcq-exact")
    return (0.0, f"Incorrect: expected '{question.correct_answer}', got '{answer.answer_text}'.", 1.0, "mcq-exact")


def _grade_numeric(question: Question, answer: StudentAnswer):
    """Numeric: full marks if the number is within tolerance of the correct one."""
    try:
        given = float(answer.answer_text.strip())
    except ValueError:
        return (0.0, f"Not a number: '{answer.answer_text}'.", 1.0, "numeric-tolerance")

    correct = float(question.correct_answer)
    if abs(given - correct) <= question.tolerance:
        return (question.max_marks,
                f"Within tolerance: {given} is within {question.tolerance} of {correct}.",
                1.0, "numeric-tolerance")
    return (0.0,
            f"Outside tolerance: {given} differs from {correct} by more than {question.tolerance}.",
            1.0, "numeric-tolerance")


def _grade_with_ai(question: Question, answer: StudentAnswer, ai: AIGrader):
    """Short/essay: hand off to the AI grader (mock or real)."""
    marks, justification, confidence = ai.grade_text(
        question.text, question.rubric or "", answer.answer_text, question.max_marks
    )
    method = "ai-" + question.type.value   # e.g. "ai-essay"
    return (marks, justification, confidence, method)


# ---------------------------------------------------------------------------
# The main entry point: route an answer to the right strategy and return a
# fully validated GradeRecord.
# ---------------------------------------------------------------------------

def grade_answer(
    question: Question, answer: StudentAnswer, ai: AIGrader | None = None
) -> GradeRecord:
    """Grade one answer and return a GradeRecord.

    Routes by question type. For AI-judged types, an AIGrader must be provided
    (defaults to MockAIGrader so everything works out of the box).
    """
    if ai is None:
        ai = get_ai_grader()   # real grader if a key is set, else mock

    if question.type == QuestionType.MCQ:
        marks, justification, confidence, method = _grade_mcq(question, answer)
    elif question.type == QuestionType.NUMERIC and not question.rubric:
        # A plain numeric question (no rubric) -> fast, free, exact tolerance check.
        marks, justification, confidence, method = _grade_numeric(question, answer)
    else:  # SHORT, ESSAY, or NUMERIC WITH a rubric -> AI does step marking.
        marks, justification, confidence, method = _grade_with_ai(question, answer, ai)

    # Build a validated GradeRecord - the bouncer checks the AI's output too.
    return GradeRecord(
        id=f"G_{answer.student_id}_{question.id}",
        answer_id=answer.id,
        question_id=question.id,
        student_id=answer.student_id,
        marks_awarded=marks,
        max_marks=question.max_marks,
        justification=justification,
        confidence=confidence,
        grading_method=method,
    )


# ---------------------------------------------------------------------------
# The REAL AI grader: calls OpenRouter (OpenAI-compatible) to grade free text.
# It implements the SAME AIGrader interface as the mock, so it is a drop-in
# replacement. The API key is read from settings (from the .env file), never
# hardcoded.
# ---------------------------------------------------------------------------

class OpenRouterAIGrader(AIGrader):
    """An AI grader that calls a model via the OpenRouter API."""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def grade_text(self, question_text, rubric, answer_text, max_marks):
        import requests  # imported here so the mock path needs no network libs

        # The instruction we give the model. We ask for STRICT JSON back so we
        # can parse it reliably.
        system = (
            "You are a fair, strict exam grader. Grade the student's answer "
            "against the rubric. Award partial credit for correct steps. "
            "Respond ONLY with JSON: "
            '{"marks": <number>, "justification": "<short reason>", '
            '"confidence": <0..1>}.'
        )
        user = (
            f"Question: {question_text}\n"
            f"Rubric / marking guide: {rubric}\n"
            f"Maximum marks: {max_marks}\n"
            f"Student answer: {answer_text}\n"
            f"Grade it now."
        )

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,   # deterministic-ish grading
            }),
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # The model should return JSON; parse it, with a safe fallback.
        marks, justification, confidence = self._parse(content, max_marks)
        # Never let the model exceed the maximum.
        marks = max(0.0, min(float(marks), float(max_marks)))
        return (round(marks, 1), justification, round(float(confidence), 2))

    @staticmethod
    def _parse(content: str, max_marks: float):
        """Pull marks/justification/confidence out of the model's reply."""
        try:
            # find the first {...} block in case the model added extra text
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            return (
                data.get("marks", max_marks / 2),
                data.get("justification", "No justification returned."),
                data.get("confidence", 0.5),
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            # If the model did not return clean JSON, fail safe: half marks.
            return (max_marks / 2, f"Could not parse grader reply: {content[:80]}", 0.3)


def get_ai_grader() -> AIGrader:
    """Pick the AI grader: the real one if a key is configured, else the mock."""
    from grader.config import settings
    if settings.has_api_key():
        return OpenRouterAIGrader(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockAIGrader()
