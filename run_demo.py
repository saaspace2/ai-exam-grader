"""Run the AI Exam-Grader on a sample exam and SHOW everything.

Usage (from the project root, with your venv active):
    python run_demo.py

It reads sample_exam.json, prints each question and the student's answer,
grades them, and prints the marks with the AI's justification.

It uses the MockAIGrader by default (free, offline). If you set a real
OPENROUTER_API_KEY in your .env, the short/essay/numeric-with-rubric
questions will be graded by the real AI instead.
"""

import json
from pathlib import Path

from grader.ingestion import ingest_batch
from grader.grading import grade_answer, MockAIGrader
from grader.config import settings


def main():
    # ---- load the exam from the JSON file ----
    data = json.loads(Path("sample_exam.json").read_text(encoding="utf-8"))

    print("=" * 72)
    print("  AI EXAM-GRADER  -  demo run")
    print("=" * 72)

    # tell the user which grader is active
    if settings.has_api_key():
        print(f"  Grader: REAL OpenRouter AI  (model: {settings.OPENROUTER_MODEL})")
    else:
        print("  Grader: MockAIGrader  (free/offline - set OPENROUTER_API_KEY in .env for real AI)")
    print("=" * 72)

    # ---- STEP 1: ingest ----
    report = ingest_batch(
        raw_questions=data["questions"],
        raw_answers=data["answers"],
    )
    print(f"\nSTEP 1 - INGEST: {report.summary()}")
    if report.failures:
        print("  Some records failed to ingest:")
        for f in report.failures:
            print(f"    - {f['kind']} {f['raw'].get('id')}: {f['error'].splitlines()[0]}")

    questions = {q.id: q for q in report.questions}

    # pick the grader: real if a key is set, else the free mock
    ai = None if settings.has_api_key() else MockAIGrader()

    # ---- STEP 2: show each Q + A, then grade ----
    print("\nSTEP 2 - GRADE each answer:\n")
    total_awarded = 0.0
    total_possible = 0.0

    for a in report.answers:
        q = questions.get(a.question_id)
        if q is None:
            print(f"  (skipping answer {a.id}: no matching question)")
            continue

        g = grade_answer(q, a, ai=ai)
        total_awarded += g.marks_awarded
        total_possible += g.max_marks

        print(f"  ┌─ {q.id}  [{q.type.value}]  (worth {q.max_marks} marks)")
        print(f"  │  QUESTION: {q.text}")
        if q.correct_answer:
            print(f"  │  CORRECT : {q.correct_answer}")
        if q.rubric:
            print(f"  │  RUBRIC  : {q.rubric}")
        print(f"  │  {a.student_id}'s ANSWER: {a.answer_text}")
        print(f"  │  ── GRADE: {g.marks_awarded}/{g.max_marks}   (method: {g.grading_method}, confidence: {g.confidence})")
        print(f"  └─ WHY: {g.justification}")
        print()

    # ---- summary ----
    pct = (total_awarded / total_possible * 100) if total_possible else 0
    print("=" * 72)
    print(f"  TOTAL: {total_awarded} / {total_possible} marks  ({pct:.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()