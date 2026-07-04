# Databricks notebook source
# =============================================================================
# 01_setup_catalog.py  -  Create the infrastructure FROM SCRATCH.
#
# This notebook creates, for one environment (dev/acc/prd):
#   - the CATALOG        (top-level container in Unity Catalog)
#   - the SCHEMA         (a database inside the catalog)
#   - five DELTA TABLES  (questions, answers, grades, audit, predictions)
#   - a VOLUME           (cloud file storage for the uploaded answer images)
#
# It is run by setup_job (resources/setup.job.yml). The catalog and schema
# names are passed in as job parameters, so the SAME notebook sets up dev, acc,
# or prd depending on which target you deployed.
# =============================================================================

# COMMAND ----------

# Read the catalog + schema names passed in by the job (with safe defaults).
dbutils.widgets.text("catalog", "exam_grader_dev")
dbutils.widgets.text("schema", "grading")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
print(f"Setting up catalog='{CATALOG}', schema='{SCHEMA}'")

# COMMAND ----------

# ---- 1. CREATE THE CATALOG ----
# A catalog is the top-level container in Unity Catalog (like a top folder).
# IF NOT EXISTS means running this twice is safe (idempotent).
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
print(f"Catalog '{CATALOG}' ready.")

# ---- 2. CREATE THE SCHEMA ----
# A schema is a database inside the catalog (a sub-folder holding tables).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema '{CATALOG}.{SCHEMA}' ready.")

# COMMAND ----------

# ---- 3. CREATE THE DELTA TABLES ----
# Delta is Databricks' table format (like SQLite tables, but cloud-scale and
# versioned). These mirror the five tables in the local SQLite store, so the
# Delta-backed store (Stage 3) can use the SAME interface.

# questions: one row per exam question + its key/rubric
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.questions (
    id             STRING,
    type           STRING,
    text           STRING,
    correct_answer STRING,
    rubric         STRING,
    tolerance      DOUBLE,
    max_marks      DOUBLE
) USING DELTA
""")

# answers: one row per student answer
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.answers (
    id           STRING,
    question_id  STRING,
    student_id   STRING,
    answer_text  STRING
) USING DELTA
""")

# grades: one row per grade (current value)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.grades (
    id             STRING,
    answer_id      STRING,
    question_id    STRING,
    student_id     STRING,
    marks_awarded  DOUBLE,
    max_marks      DOUBLE,
    justification  STRING,
    confidence     DOUBLE,
    grading_method STRING,
    status         STRING
) USING DELTA
""")

# audit: append-only history of every grade change
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.audit (
    id               STRING,
    grade_record_id  STRING,
    timestamp        STRING,
    actor            STRING,
    action           STRING,
    old_marks        DOUBLE,
    new_marks        DOUBLE,
    reason           STRING
) USING DELTA
""")

# predictions: the DATASET - one row per graded upload (image + read + grade)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.predictions (
    id             STRING,
    created_at     STRING,
    student_id     STRING,
    question_id    STRING,
    image_path     STRING,
    answer_read    STRING,
    marks_awarded  DOUBLE,
    max_marks      DOUBLE,
    grading_method STRING,
    model_used     STRING
) USING DELTA
""")

print("All 5 Delta tables ready.")

# COMMAND ----------

# ---- 4. CREATE THE VOLUME (cloud file storage for uploaded images) ----
# A Volume is Unity Catalog's managed file storage - the cloud equivalent of
# the local uploads/ folder. Scanned answer scripts are stored here.
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.uploads")
print(f"Volume '{CATALOG}.{SCHEMA}.uploads' ready.")
print(f"  (upload path: /Volumes/{CATALOG}/{SCHEMA}/uploads/)")

# COMMAND ----------

# ---- 5. SHOW WHAT WE CREATED ----
print("Tables in the schema:")
display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))
print("Setup complete. The environment is ready for grading jobs.")