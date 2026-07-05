"""batch_grader.py - grade many answers at once and write results to Delta.

The batch pipeline: read questions + ungraded answers from Delta, grade each
with the SAME grading engine, and write grades + predictions back to Delta.
This is the "process a whole exam" flow, and the class holds no hardcoded
names - it takes a DeltaStore, so it targets whatever environment that store
points at.
"""

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("grader")

import uuid
from datetime import datetime, timezone

from grader.grading import grade_answer


class BatchGrader:
    """Grade every ungraded answer in the store, in one pass."""

    def __init__(self, store):
        # store is a DeltaStore (or the SQLite Store - same interface!)
        self.store = store

    def _get_ungraded(self) -> list[dict]:
        """Return answer dicts that do not yet have a grade.

        Works for a DeltaStore (uses Spark SQL) or the SQLite Store (uses its
        connection) - so the batch logic is testable locally too.
        """
        if hasattr(self.store, "spark"):
            base = self.store.base
            rows = self.store.spark.sql(f"""
                SELECT a.* FROM {base}.answers a
                LEFT JOIN {base}.grades g
                  ON a.student_id = g.student_id AND a.question_id = g.question_id
                WHERE g.id IS NULL
            """).collect()
            return [r.asDict() for r in rows]
        # SQLite fallback
        cur = self.store.conn.execute("""
            SELECT a.data FROM answers a
            LEFT JOIN grades g
              ON a.student_id = g.student_id AND a.question_id = g.question_id
            WHERE g.id IS NULL
        """)
        import json
        return [json.loads(r["data"]) for r in cur.fetchall()]

    def grade_all(self) -> dict:
        """Grade every answer that does not yet have a grade.

        Returns a summary dict: how many graded, total marks, etc.
        """
        ungraded = self._get_ungraded()
        logger.info(f"Found {len(ungraded)} ungraded answer(s).")
        if not ungraded:
            return {"graded": 0, "total_marks": 0.0, "out_of": 0.0}

        graded = 0
        total_marks = 0.0
        out_of = 0.0

        for a in ungraded:
            # a is already a dict
            # 2. Load the matching question
            question = self.store.get_question(a["question_id"])
            if question is None:
                logger.warning(f"No question for answer {a['id']}, skipping.")
                continue

            # 3. Grade with the SAME engine used everywhere
            from grader.models import StudentAnswer
            answer = StudentAnswer(
                id=a["id"], question_id=a["question_id"],
                student_id=a["student_id"], answer_text=a["answer_text"],
            )
            record = grade_answer(question, answer)

            # 4. Save the grade (writes an initial audit slip too)
            try:
                self.store.save_grade(record)
            except ValueError as e:
                logger.warning(f"Could not save grade for {a['id']}: {e}")
                continue

            # 5. Record a prediction row (the dataset)
            self.store.save_prediction(
                prediction_id=uuid.uuid4().hex,
                created_at=datetime.now(timezone.utc).isoformat(),
                student_id=a["student_id"], question_id=a["question_id"],
                image_path=None, answer_read=a["answer_text"],
                marks_awarded=record.marks_awarded, max_marks=record.max_marks,
                grading_method=record.grading_method, model_used="batch",
            )

            graded += 1
            total_marks += record.marks_awarded
            out_of += record.max_marks

        summary = {"graded": graded, "total_marks": total_marks, "out_of": out_of}
        logger.info(f"Batch grading complete: {summary}")
        return summary