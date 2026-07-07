"""Storage + Audit Trail - the filing cabinet (Component 4).

This is STEP 3 of the pipeline. It saves records permanently in a SQLite file
so they survive after the program ends. It also keeps an APPEND-ONLY audit
trail: every grade and every change writes an un-editable slip, so grades are
provable and tamper-evident.

Safeguards against mixed-up answers:
  - UNIQUENESS: one answer per (student_id, question_id). No silent overwrite.
  - QUESTION MUST EXIST: you cannot save an answer for a question that is not
    stored, and cannot grade against a missing question.

The audit table is protected in code: this module offers NO update or delete
for audit entries - only append and read.
"""

import csv
import json
import os
import sqlite3
from pathlib import Path

from grader.config import settings
from grader.models import (
    Actor,
    AuditEntry,
    GradeRecord,
    Question,
    StudentAnswer,
)

# Column order for the auto-saved CSV (kept separate from the DB schema so
# the CSV stays a stable, human-friendly export).
_CSV_COLUMNS = [
    "id", "created_at", "student_id", "question_id",
    "answer_read", "marks_awarded", "max_marks",
    "grading_method", "model_used", "image_path", "image_url",
]


class Store:
    """A SQLite-backed store for questions, answers, grades, and audit slips."""

    def __init__(self, db_path: str = "grader.sqlite"):
        """Open (or create) the database file and make sure the tables exist.

        Use ":memory:" as the path for a temporary in-memory DB (great for
        tests - it vanishes when closed and never touches disk).
        """
        self.db_path = db_path
        # check_same_thread=False keeps it simple for our single-process use.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row   # rows behave like dicts
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._create_tables()

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------
    def _create_tables(self):
        """Create the four tables if they do not already exist."""
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id            TEXT PRIMARY KEY,
                data          TEXT NOT NULL          -- the full Question as JSON
            );

            CREATE TABLE IF NOT EXISTS answers (
                id            TEXT PRIMARY KEY,
                question_id   TEXT NOT NULL,
                student_id    TEXT NOT NULL,
                data          TEXT NOT NULL,          -- the full StudentAnswer as JSON
                -- SAFEGUARD: only one answer per student per question
                UNIQUE (student_id, question_id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            );

            CREATE TABLE IF NOT EXISTS grades (
                id            TEXT PRIMARY KEY,
                answer_id     TEXT NOT NULL,
                question_id   TEXT NOT NULL,
                student_id    TEXT NOT NULL,
                data          TEXT NOT NULL,          -- the full GradeRecord as JSON
                UNIQUE (student_id, question_id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            );

            -- Append-only: we INSERT and SELECT here, never UPDATE or DELETE.
            CREATE TABLE IF NOT EXISTS audit (
                id                TEXT PRIMARY KEY,
                grade_record_id   TEXT NOT NULL,
                data              TEXT NOT NULL       -- the full AuditEntry as JSON
            );

            -- The DATASET: one row per uploaded script + what the AI read + graded.
            -- This is the data flywheel - kept for evidence, monitoring, and training.
            CREATE TABLE IF NOT EXISTS predictions (
                id             TEXT PRIMARY KEY,
                created_at     TEXT NOT NULL,
                student_id     TEXT NOT NULL,
                question_id    TEXT NOT NULL,
                image_path     TEXT,               -- where the original scan is saved
                answer_read    TEXT,               -- what the OCR/vision read
                marks_awarded  REAL,
                max_marks      REAL,
                grading_method TEXT,
                model_used     TEXT
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_question(self, question: Question) -> None:
        """Store a question (replace if the same id is saved again)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO questions (id, data) VALUES (?, ?)",
            (question.id, question.model_dump_json()),
        )
        self.conn.commit()

    def save_answer(self, answer: StudentAnswer) -> None:
        """Store an answer. Enforces: the question must exist, and one answer
        per (student, question)."""
        # SAFEGUARD: the question must already be stored.
        if not self._exists("questions", answer.question_id):
            raise ValueError(
                f"Cannot save answer '{answer.id}': question "
                f"'{answer.question_id}' does not exist in the store."
            )
        try:
            self.conn.execute(
                "INSERT INTO answers (id, question_id, student_id, data) "
                "VALUES (?, ?, ?, ?)",
                (answer.id, answer.question_id, answer.student_id,
                 answer.model_dump_json()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"Duplicate answer for student '{answer.student_id}' on "
                f"question '{answer.question_id}' (or duplicate id '{answer.id}')."
            ) from e

    def save_grade(self, grade: GradeRecord, actor: Actor = Actor.AI,
                   reason: str = "Initial grade.") -> AuditEntry:
        """Store a grade AND append the first audit slip for it.

        Returns the AuditEntry that was written.
        """
        if not self._exists("questions", grade.question_id):
            raise ValueError(
                f"Cannot save grade '{grade.id}': question "
                f"'{grade.question_id}' does not exist."
            )
        try:
            self.conn.execute(
                "INSERT INTO grades (id, answer_id, question_id, student_id, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (grade.id, grade.answer_id, grade.question_id, grade.student_id,
                 grade.model_dump_json()),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"A grade already exists for student '{grade.student_id}' on "
                f"question '{grade.question_id}'."
            ) from e

        # append the first audit slip (initial grade: no old marks)
        entry = AuditEntry(
            id=f"AU_{grade.id}_1",
            grade_record_id=grade.id,
            actor=actor,
            action="initial_grade",
            old_marks=None,
            new_marks=grade.marks_awarded,
            reason=reason,
        )
        self._append_audit(entry)
        self.conn.commit()
        return entry

    def update_grade(self, grade: GradeRecord) -> None:
        """Replace a stored grade with an updated one (same id).

        Used by re-evaluation to change the marks/status. The AUDIT trail is
        separate and append-only: the change itself is recorded there, never
        overwritten. This only updates the 'current' grade row.
        """
        self.conn.execute(
            "UPDATE grades SET data = ?, student_id = ?, question_id = ? WHERE id = ?",
            (grade.model_dump_json(), grade.student_id, grade.question_id, grade.id),
        )
        self.conn.commit()

    def get_grade_by_id(self, grade_id: str) -> "GradeRecord | None":
        row = self.conn.execute(
            "SELECT data FROM grades WHERE id = ?", (grade_id,)
        ).fetchone()
        from grader.models import GradeRecord as _GR
        return _GR.model_validate_json(row["data"]) if row else None

    # ------------------------------------------------------------------
    # The append-only audit trail
    # ------------------------------------------------------------------
    def _append_audit(self, entry: AuditEntry) -> None:
        """Add one audit slip. This is the ONLY way to write to the audit table -
        there is deliberately no update or delete."""
        self.conn.execute(
            "INSERT INTO audit (id, grade_record_id, data) VALUES (?, ?, ?)",
            (entry.id, entry.grade_record_id, entry.model_dump_json()),
        )
        self.conn.commit()

    def append_audit(self, entry: AuditEntry) -> None:
        """Public append for later components (e.g. appeals) to log a change."""
        self._append_audit(entry)

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def get_question(self, question_id: str) -> Question | None:
        row = self.conn.execute(
            "SELECT data FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
        return Question.model_validate_json(row["data"]) if row else None

    def get_grade(self, student_id: str, question_id: str) -> GradeRecord | None:
        row = self.conn.execute(
            "SELECT data FROM grades WHERE student_id = ? AND question_id = ?",
            (student_id, question_id),
        ).fetchone()
        return GradeRecord.model_validate_json(row["data"]) if row else None

    def get_audit_history(self, grade_record_id: str) -> list[AuditEntry]:
        """All audit slips for a grade, in the order they were written."""
        rows = self.conn.execute(
            "SELECT data FROM audit WHERE grade_record_id = ? ORDER BY rowid",
            (grade_record_id,),
        ).fetchall()
        return [AuditEntry.model_validate_json(r["data"]) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _exists(self, table: str, id_value: str) -> bool:
        row = self.conn.execute(
            f"SELECT 1 FROM {table} WHERE id = ? LIMIT 1", (id_value,)
        ).fetchone()
        return row is not None

    def save_prediction(self, *, prediction_id: str, created_at: str,
                        student_id: str, question_id: str, image_path: str | None,
                        answer_read: str, marks_awarded: float, max_marks: float,
                        grading_method: str, model_used: str) -> None:
        """Record one prediction in the dataset (image + read + grade + model).

        Also appends the same row to a running CSV file on disk (see
        settings.CSV_EXPORT_PATH), so the dataset survives as a plain file
        you can open in Excel/Sheets even without querying the database.
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO predictions
               (id, created_at, student_id, question_id, image_path, answer_read,
                marks_awarded, max_marks, grading_method, model_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, created_at, student_id, question_id, image_path,
             answer_read, marks_awarded, max_marks, grading_method, model_used),
        )
        self.conn.commit()
        self._append_prediction_csv(
            prediction_id=prediction_id, created_at=created_at,
            student_id=student_id, question_id=question_id,
            image_path=image_path, answer_read=answer_read,
            marks_awarded=marks_awarded, max_marks=max_marks,
            grading_method=grading_method, model_used=model_used,
        )

    def _append_prediction_csv(self, *, prediction_id, created_at, student_id,
                               question_id, image_path, answer_read,
                               marks_awarded, max_marks, grading_method,
                               model_used) -> None:
        """Append one row to the auto-saved CSV, writing the header first if
        the file doesn't exist yet. Never raises - a CSV write failure should
        not break grading."""
        try:
            csv_path = settings.CSV_EXPORT_PATH
            parent = os.path.dirname(csv_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
            # image_url is a clickable path served by the API (see /uploads
            # static mount in api.py), not the raw local disk path.
            image_url = f"/uploads/{os.path.basename(image_path)}" if image_path else ""
            row = {
                "id": prediction_id, "created_at": created_at,
                "student_id": student_id, "question_id": question_id,
                "answer_read": answer_read, "marks_awarded": marks_awarded,
                "max_marks": max_marks, "grading_method": grading_method,
                "model_used": model_used, "image_path": image_path or "",
                "image_url": image_url,
            }
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            pass

    def list_predictions(self) -> list[dict]:
        """Return every prediction row as a dict (the dataset)."""
        rows = self.conn.execute(
            "SELECT * FROM predictions ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()

    # allow use as a context manager: with Store(...) as s:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()