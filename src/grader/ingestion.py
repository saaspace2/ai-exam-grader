"""Ingestion - the 'receiving desk' of the grading pipeline (Component 2).

This is STEP 1 of the pipeline. It takes RAW input (plain dictionaries, such
as you would load from a JSON file or receive from a web form) and turns it
into clean, validated Question and StudentAnswer objects (our forms from
Component 1). Pydantic checks every one at the door.

The golden rule of ingestion: DO NOT crash on one bad record. If a pile has
100 records and 3 are broken, we let the 97 good ones through and report the
3 failures clearly. Real data pipelines behave this way.
"""

from dataclasses import dataclass, field

from pydantic import ValidationError

from grader.models import Question, StudentAnswer


# ---------------------------------------------------------------------------
# A small container to report what happened during a batch ingest:
# which records succeeded, and which failed (with the reason why).
# ---------------------------------------------------------------------------

@dataclass
class IngestReport:
    """The result of ingesting a batch: the good records and the failures."""

    questions: list[Question] = field(default_factory=list)
    answers: list[StudentAnswer] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if nothing failed."""
        return len(self.failures) == 0

    def summary(self) -> str:
        """A short human-readable summary line."""
        return (
            f"{len(self.questions)} questions, {len(self.answers)} answers, "
            f"{len(self.failures)} failed."
        )


# ---------------------------------------------------------------------------
# Single-record ingest functions. Each takes one raw dict and returns one
# validated object - or raises ValidationError if the data is bad.
# ---------------------------------------------------------------------------

def ingest_question(raw: dict) -> Question:
    """Turn one raw dictionary into a validated Question.

    Pydantic does the checking: if a required field is missing or a value
    breaks a rule (e.g. max_marks <= 0), it raises ValidationError.
    """
    return Question(**raw)


def ingest_answer(raw: dict) -> StudentAnswer:
    """Turn one raw dictionary into a validated StudentAnswer."""
    return StudentAnswer(**raw)


# ---------------------------------------------------------------------------
# Batch ingest. Processes a whole pile, collecting successes and failures
# instead of crashing on the first bad record.
# ---------------------------------------------------------------------------

def ingest_batch(
    raw_questions: list[dict] | None = None,
    raw_answers: list[dict] | None = None,
) -> IngestReport:
    """Ingest many raw questions and answers at once.

    Returns an IngestReport listing the validated records and any failures.
    A failure never stops the batch - it is recorded and we move on.
    """
    report = IngestReport()

    for raw in (raw_questions or []):
        try:
            report.questions.append(ingest_question(raw))
        except ValidationError as e:
            report.failures.append({
                "kind": "question",
                "raw": raw,
                "error": str(e),
            })

    for raw in (raw_answers or []):
        try:
            report.answers.append(ingest_answer(raw))
        except ValidationError as e:
            report.failures.append({
                "kind": "answer",
                "raw": raw,
                "error": str(e),
            })

    return report
