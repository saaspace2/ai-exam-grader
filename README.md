# AI Exam-Grading System

An AI that grades exam answers (MCQ, numeric, short, essay), explains its
reasoning, lets students ask for clarification (RAG), and lets them appeal a
mark — with every grade-change recorded in a tamper-proof audit trail.

Built slowly, one component at a time, for learning.

## Where things live
```
src/grader/models.py   <- the four data 'forms' (Component 1, DONE)
tests/unit/            <- tests for each component
```

## Component 1: The Data Model  (DONE)
Four Pydantic "forms" that shape every piece of information:
- Question       — one exam question
- StudentAnswer  — a student's answer to a question
- GradeRecord    — the grade + justification (the heart)
- AuditEntry     — an append-only record of any grade change (the safety net)

Pydantic acts as a "bouncer", rejecting bad data at the door:
- marks can never exceed max marks
- confidence must be between 0 and 1
- every grade-change must carry a non-empty reason

## Run the tests
```
pip install -e ".[test]"
pytest
```
