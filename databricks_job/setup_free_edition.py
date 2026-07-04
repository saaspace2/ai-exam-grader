# Databricks notebook source
# =============================================================================
# setup_free_edition.py  -  Create catalog + schema + Delta tables + Volume
# to RUN DIRECTLY in a Databricks Free Edition notebook (serverless).
#
# HOW TO USE:
#   1. In your Databricks workspace: New -> Notebook
#   2. Make sure it's attached to "Serverless" compute (top right)
#   3. Paste this whole file in (or import it)
#   4. Run all cells
#   5. Open the Catalog browser - you'll see exam_grader appear!
#
# No bundle, no job clusters - just serverless SQL. Works on Free Edition.
# =============================================================================

# COMMAND ----------

# Free Edition: one workspace, so we use a single catalog "exam_grader".
# (In a paid setup we'd have exam_grader_dev / _acc / _prd.)
CATALOG = "exam_grader"
SCHEMA = "grading"
print(f"Creating catalog='{CATALOG}', schema='{SCHEMA}'")

# COMMAND ----------

# ---- 1. CATALOG (top-level container) ----
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
print(f"Catalog '{CATALOG}' ready.")

# ---- 2. SCHEMA (database inside the catalog) ----
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema '{CATALOG}.{SCHEMA}' ready.")

# COMMAND ----------

# ---- 3. THE FIVE DELTA TABLES (mirror the local SQLite tables) ----
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.questions (
    id STRING, type STRING, text STRING, correct_answer STRING,
    rubric STRING, tolerance DOUBLE, max_marks DOUBLE
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.answers (
    id STRING, question_id STRING, student_id STRING, answer_text STRING
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.grades (
    id STRING, answer_id STRING, question_id STRING, student_id STRING,
    marks_awarded DOUBLE, max_marks DOUBLE, justification STRING,
    confidence DOUBLE, grading_method STRING, status STRING
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.audit (
    id STRING, grade_record_id STRING, timestamp STRING, actor STRING,
    action STRING, old_marks DOUBLE, new_marks DOUBLE, reason STRING
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.predictions (
    id STRING, created_at STRING, student_id STRING, question_id STRING,
    image_path STRING, answer_read STRING, marks_awarded DOUBLE,
    max_marks DOUBLE, grading_method STRING, model_used STRING
) USING DELTA
""")
print("All 5 Delta tables ready.")

# COMMAND ----------

# ---- 4. VOLUME (cloud file storage for uploaded images) ----
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.uploads")
print(f"Volume ready at: /Volumes/{CATALOG}/{SCHEMA}/uploads/")

# COMMAND ----------

# ---- 5. CONFIRM ----
print("Tables created:")
display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# ---- 6. QUICK TEST: insert one question and read it back ----
spark.sql(f"""
INSERT INTO {CATALOG}.{SCHEMA}.questions
VALUES ('Q1', 'mcq', 'Capital of France?', 'Paris', NULL, 0.0, 2.0)
""")
print("Inserted a test question. Reading it back:")
display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.questions"))
print("\nSetup complete! Open the Catalog browser to see exam_grader > grading > tables.")