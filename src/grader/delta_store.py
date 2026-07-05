"""delta_store.py - a Store backed by Delta tables in Unity Catalog.

This is the CLOUD twin of store.py (SQLite). It exposes the SAME public methods
(save_question, save_answer, save_grade, get_grade, ...), so the grading engine,
appeals, and retrieval code can use it WITHOUT any changes - just swap
Store(...) for DeltaStore(...). That is the payoff of the clean interface.

It uses Spark SQL to read/write Delta tables at {catalog}.{schema}.{table}.
Meant to run inside Databricks (where `spark` exists). The methods mirror the
SQLite store's behaviour, including the append-only audit trail.
"""

import json

from grader.models import (
    Actor,
    AuditEntry,
    GradeRecord,
    GradeStatus,
    Question,
    StudentAnswer,
)
from grader.project_config import ProjectConfig


class DeltaStore:
    """Same interface as Store, but persists to Delta tables in Unity Catalog."""

    def __init__(self, config: ProjectConfig, spark):
        self.config = config
        self.spark = spark
        self.base = config.base_path  # e.g. "exam_grader_dev.grading"

    # ---- small helpers ------------------------------------------------
    def _full(self, table: str) -> str:
        return f"{self.base}.{table}"

    def _sql_str(self, value) -> str:
        """Safely format a Python value as a SQL literal."""
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return str(value)
        # escape single quotes for SQL
        return "'" + str(value).replace("'", "''") + "'"

    def _insert(self, table: str, row: dict) -> None:
        """Insert one row into a Delta table using the table's own schema."""
        full = self._full(table)
        target_schema = self.spark.table(full).schema
        cols = [f.name for f in target_schema]
        aligned = [{c: row.get(c) for c in cols}]
        sdf = self.spark.createDataFrame(aligned, schema=target_schema)
        sdf.write.format("delta").mode("append").saveAsTable(full)

    def _exists(self, table: str, id_value: str) -> bool:
        full = self._full(table)
        n = self.spark.sql(
            f"SELECT count(*) AS n FROM {full} WHERE id = {self._sql_str(id_value)}"
        ).collect()[0]["n"]
        return n > 0

    # ---- questions ----------------------------------------------------
    def save_question(self, question: Question) -> None:
        q = question.model_dump(mode="json")
        self._insert("questions", {
            "id": q["id"], "type": q["type"], "text": q.get("text"),
            "correct_answer": q.get("correct_answer"), "rubric": q.get("rubric"),
            "tolerance": q.get("tolerance"), "max_marks": q["max_marks"],
        })

    def get_question(self, question_id: str) -> Question | None:
        full = self._full("questions")
        rows = self.spark.sql(
            f"SELECT * FROM {full} WHERE id = {self._sql_str(question_id)} LIMIT 1"
        ).collect()
        if not rows:
            return None
        r = rows[0].asDict()
        return Question(**{k: v for k, v in r.items() if v is not None})

    # ---- answers ------------------------------------------------------
    def save_answer(self, answer: StudentAnswer) -> None:
        # question must exist (same safeguard as SQLite store)
        if self.get_question(answer.question_id) is None:
            raise ValueError(
                f"Cannot save answer for unknown question '{answer.question_id}'."
            )
        # no duplicate (student_id, question_id)
        if self._answer_exists(answer.student_id, answer.question_id):
            raise ValueError(
                f"Answer for {answer.student_id}/{answer.question_id} already exists."
            )
        a = answer.model_dump(mode="json")
        self._insert("answers", {
            "id": a["id"], "question_id": a["question_id"],
            "student_id": a["student_id"], "answer_text": a["answer_text"],
        })

    def _answer_exists(self, student_id: str, question_id: str) -> bool:
        full = self._full("answers")
        n = self.spark.sql(
            f"SELECT count(*) AS n FROM {full} "
            f"WHERE student_id = {self._sql_str(student_id)} "
            f"AND question_id = {self._sql_str(question_id)}"
        ).collect()[0]["n"]
        return n > 0

    # ---- grades -------------------------------------------------------
    def save_grade(self, grade: GradeRecord, actor: Actor = Actor.AI,
                   reason: str = "Initial grade.") -> None:
        # no duplicate (student_id, question_id) grade
        if self._grade_exists(grade.student_id, grade.question_id):
            raise ValueError(
                f"Grade for {grade.student_id}/{grade.question_id} already exists."
            )
        g = grade.model_dump(mode="json")
        self._insert("grades", {
            "id": g["id"], "answer_id": g["answer_id"],
            "question_id": g["question_id"], "student_id": g["student_id"],
            "marks_awarded": g["marks_awarded"], "max_marks": g["max_marks"],
            "justification": g["justification"], "confidence": g["confidence"],
            "grading_method": g["grading_method"],
            "status": g.get("status", GradeStatus.GRADED.value),
        })
        # write the initial audit slip
        self._append_audit(AuditEntry(
            id=f"AUD_{grade.id}_init", grade_record_id=grade.id,
            actor=actor, action="initial_grade",
            old_marks=None, new_marks=grade.marks_awarded, reason=reason,
        ))

    def _grade_exists(self, student_id: str, question_id: str) -> bool:
        full = self._full("grades")
        n = self.spark.sql(
            f"SELECT count(*) AS n FROM {full} "
            f"WHERE student_id = {self._sql_str(student_id)} "
            f"AND question_id = {self._sql_str(question_id)}"
        ).collect()[0]["n"]
        return n > 0

    def get_grade(self, student_id: str, question_id: str) -> GradeRecord | None:
        full = self._full("grades")
        rows = self.spark.sql(
            f"SELECT * FROM {full} "
            f"WHERE student_id = {self._sql_str(student_id)} "
            f"AND question_id = {self._sql_str(question_id)} LIMIT 1"
        ).collect()
        if not rows:
            return None
        return GradeRecord(**rows[0].asDict())

    def get_grade_by_id(self, grade_id: str) -> GradeRecord | None:
        full = self._full("grades")
        rows = self.spark.sql(
            f"SELECT * FROM {full} WHERE id = {self._sql_str(grade_id)} LIMIT 1"
        ).collect()
        return GradeRecord(**rows[0].asDict()) if rows else None

    def update_grade(self, grade: GradeRecord) -> None:
        """Update a grade's mutable fields (Delta supports UPDATE)."""
        full = self._full("grades")
        self.spark.sql(f"""
            UPDATE {full} SET
                marks_awarded = {grade.marks_awarded},
                justification = {self._sql_str(grade.justification)},
                confidence = {grade.confidence},
                grading_method = {self._sql_str(grade.grading_method)},
                status = {self._sql_str(grade.status.value if hasattr(grade.status, 'value') else grade.status)}
            WHERE id = {self._sql_str(grade.id)}
        """)

    # ---- audit (append-only) -----------------------------------------
    def _append_audit(self, entry: AuditEntry) -> None:
        e = entry.model_dump(mode="json")
        self._insert("audit", {
            "id": e["id"], "grade_record_id": e["grade_record_id"],
            "timestamp": str(e.get("timestamp", "")),
            "actor": e["actor"], "action": e["action"],
            "old_marks": e.get("old_marks"), "new_marks": e.get("new_marks"),
            "reason": e["reason"],
        })

    def append_audit(self, entry: AuditEntry) -> None:
        self._append_audit(entry)

    def get_audit_history(self, grade_record_id: str) -> list[AuditEntry]:
        full = self._full("audit")
        rows = self.spark.sql(
            f"SELECT * FROM {full} "
            f"WHERE grade_record_id = {self._sql_str(grade_record_id)} "
            f"ORDER BY id"
        ).collect()
        return [AuditEntry(**{k: v for k, v in r.asDict().items() if v is not None})
                for r in rows]

    # ---- predictions (the dataset) -----------------------------------
    def save_prediction(self, *, prediction_id: str, created_at: str,
                        student_id: str, question_id: str, image_path: str | None,
                        answer_read: str, marks_awarded: float, max_marks: float,
                        grading_method: str, model_used: str) -> None:
        self._insert("predictions", {
            "id": prediction_id, "created_at": created_at,
            "student_id": student_id, "question_id": question_id,
            "image_path": image_path, "answer_read": answer_read,
            "marks_awarded": marks_awarded, "max_marks": max_marks,
            "grading_method": grading_method, "model_used": model_used,
        })

    def list_predictions(self) -> list[dict]:
        full = self._full("predictions")
        rows = self.spark.sql(f"SELECT * FROM {full} ORDER BY created_at").collect()
        return [r.asDict() for r in rows]

    # ---- context-manager parity with Store ---------------------------
    def close(self):
        pass  # Spark session is managed by Databricks, nothing to close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
