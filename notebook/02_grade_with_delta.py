# Databricks notebook source
# =============================================================================
# 02_grade_with_delta.py  -  Grade using the SAME engine, store to DELTA.
#
# This shows the payoff of the clean interface: we swap the local SQLite Store
# for DeltaStore, and the grading engine works unchanged - now writing to Delta
# tables in Unity Catalog instead of a local file.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

# If the import below ever fails with 'No module named grader', it means
# the %pip cell above did not run first. Run this notebook TOP to BOTTOM:
# run the %pip install cell, let %restart_python finish, THEN the rest.
# As a fallback, add the repo's src/ to the path directly:
import os, sys
_here = os.getcwd()
_root = _here if os.path.exists(os.path.join(_here, 'pyproject.toml')) else os.path.dirname(_here)
_src = os.path.join(_root, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)
print('Using src path:', _src)

# COMMAND ----------

from grader.project_config import ProjectConfig
from grader.delta_store import DeltaStore
from grader.models import Question, StudentAnswer
from grader.grading import grade_answer

config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
store = DeltaStore(config, spark)   # <-- the ONLY change from local: DeltaStore
print("Using Delta store at:", config.base_path)

# COMMAND ----------

# 1. Save a question (to the Delta questions table)
q = Question(id="Q10", type="mcq", text="Capital of Japan?",
             correct_answer="Tokyo", max_marks=2)
try:
    store.save_question(q)
    print("Saved question Q10.")
except Exception as e:
    print("Question may already exist:", e)

# COMMAND ----------

# 2. A student answers, we grade with the SAME engine, save the grade to Delta
answer = StudentAnswer(id="A_Sara_Q10", question_id="Q10",
                       student_id="Sara", answer_text="Tokyo")
try:
    store.save_answer(answer)
except Exception as e:
    print("Answer note:", e)

record = grade_answer(q, answer)     # the same grading engine as everywhere
print(f"Graded: {record.marks_awarded}/{record.max_marks} ({record.grading_method})")

try:
    store.save_grade(record)
    print("Saved grade to Delta.")
except Exception as e:
    print("Grade note:", e)

# COMMAND ----------

# 3. Read it back from Delta
g = store.get_grade("Sara", "Q10")
print("Read back from Delta:", g.marks_awarded, "/", g.max_marks, "-", g.status)

# 4. See the audit trail (initial_grade slip written automatically)
history = store.get_audit_history(record.id)
print("Audit trail entries:", len(history))
for h in history:
    print(f"  {h.action}: {h.old_marks} -> {h.new_marks} ({h.reason})")

# COMMAND ----------

# 5. Confirm in SQL too
display(spark.sql(f"SELECT * FROM {config.base_path}.grades"))
display(spark.sql(f"SELECT * FROM {config.base_path}.audit"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The payoff
# MAGIC The grading engine, models, and audit logic are **identical** to local.
# MAGIC Only the store changed (SQLite -> Delta). Same interface, cloud backend.