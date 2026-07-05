# Databricks notebook source
# =============================================================================
# 05_monitoring.py  -  Monitor grading quality over time.
#
# Builds a monitoring VIEW over the predictions + audit tables to watch for
# quality signals: average score, appeal rate, and OCR-read issues over time.
# The plain SQL/Delta parts run on Free Edition. The Databricks "Lakehouse
# Quality Monitor" at the end runs on serverless (Free Edition quota applies) - marked clearly.
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
config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
base = config.base_path

# COMMAND ----------

# 1. A monitoring table: daily grading stats (runs on Free Edition).
spark.sql(f"""
CREATE OR REPLACE TABLE {base}.model_monitoring AS
SELECT
    to_date(created_at)                       AS day,
    count(*)                                  AS predictions,
    avg(marks_awarded / max_marks)            AS avg_score_fraction,
    sum(CASE WHEN marks_awarded = 0 THEN 1 ELSE 0 END) AS zero_score_count,
    model_used
FROM {base}.predictions
GROUP BY to_date(created_at), model_used
""")
print("Monitoring table created.")
display(spark.sql(f"SELECT * FROM {base}.model_monitoring ORDER BY day"))

# COMMAND ----------

# 2. Appeal-rate signal from the audit trail (a quality indicator).
display(spark.sql(f"""
SELECT
    action,
    count(*) AS times
FROM {base}.audit
GROUP BY action
ORDER BY times DESC
"""))

# COMMAND ----------

# 3. Enable Change Data Feed so a monitor can process only new rows.
spark.sql(f"ALTER TABLE {base}.model_monitoring "
          f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("CDF enabled on monitoring table.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## NOTE (Free Edition): Lakehouse Quality Monitor
# MAGIC The block below creates a Databricks **Quality Monitor** (drift/quality
# MAGIC dashboards). This runs on serverless (Free Edition quota applies) - it runs on Free Edition serverless (within quota),
# MAGIC but is included to match the Marvel monitoring stage.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
# w = WorkspaceClient()
# w.quality_monitors.create(
#     table_name=f"{base}.model_monitoring",
#     assets_dir=f"/Workspace/Shared/monitoring/{base}",
#     output_schema_name=base,
#     inference_log=MonitorInferenceLog(
#         problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
#         prediction_col="avg_score_fraction",
#         timestamp_col="day",
#         granularities=["1 day"],
#         model_id_col="model_used",
#     ),
# )