# Databricks notebook source
# =============================================================================
# 03_batch_grade.py  -  Grade EVERY ungraded answer in one pass, to Delta.
#
# The batch pipeline job. Reads questions + ungraded answers from Delta, grades
# each with the same engine, writes grades + predictions back. Idempotent - only
# grades answers that don't yet have a grade, so it's safe to run on a schedule.
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
from grader.batch_grader import BatchGrader

config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
store = DeltaStore(config, spark)

# COMMAND ----------

# (Optional) seed a few answers to grade, if the answers table is empty.
from grader.models import Question, StudentAnswer
seed = [
    ("Q20", "mcq", "Capital of Italy?", "Rome", 2, [("Ana", "Rome"), ("Ben", "Milan")]),
]
for qid, qtype, text, key, mx, students in seed:
    try:
        store.save_question(Question(id=qid, type=qtype, text=text,
                                     correct_answer=key, max_marks=mx))
    except Exception as e:
        print("question note:", e)
    for sid, ans in students:
        try:
            store.save_answer(StudentAnswer(id=f"A_{sid}_{qid}", question_id=qid,
                                            student_id=sid, answer_text=ans))
        except Exception as e:
            print("answer note:", e)

# COMMAND ----------

# Run the batch grader - grades all ungraded answers.
summary = BatchGrader(store).grade_all()
print("Batch summary:", summary)

# COMMAND ----------

# See the results in Delta
display(spark.sql(f"SELECT student_id, question_id, marks_awarded, max_marks FROM {config.base_path}.grades ORDER BY student_id"))
display(spark.sql(f"SELECT count(*) AS predictions FROM {config.base_path}.predictions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotent
# MAGIC Run this again - it grades **0** new answers (all are already graded).