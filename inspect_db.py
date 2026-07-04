"""Peek inside the grader database - prints every table in a readable way.

Usage:
    python inspect_db.py
"""
import json
import sqlite3
from grader.config import settings

db = settings.GRADER_DB_PATH
print(f"Opening database: {db}\n")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

def show(title, query):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError as e:
        print("  (table not found:", e, ")\n"); return
    if not rows:
        print("  (empty)\n"); return
    for r in rows:
        d = dict(r)
        # if there's a 'data' JSON column, expand the useful bits
        if "data" in d:
            obj = json.loads(d["data"])
            print(" ", {k: obj.get(k) for k in list(obj)[:6]})
        else:
            print(" ", d)
    print()

show("QUESTIONS", "SELECT id, data FROM questions")
show("ANSWERS", "SELECT id, student_id, question_id FROM answers")
show("GRADES", "SELECT student_id, question_id, data FROM grades")
show("AUDIT TRAIL", "SELECT grade_record_id, data FROM audit ORDER BY rowid")
show("PREDICTIONS (the dataset)",
     "SELECT student_id, question_id, answer_read, marks_awarded, max_marks, model_used FROM predictions")

conn.close()