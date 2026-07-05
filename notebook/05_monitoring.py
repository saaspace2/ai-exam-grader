# Databricks notebook source
# =============================================================================
# 05_monitoring.py  -  Build the monitoring table from the payload (Marvel-style).
#
# Marvel's create_or_refresh_monitoring reads the endpoint's payload table
# (custom_model_payload), parses the request/response JSON, and writes a
# structured model_monitoring table. We do the same: build a payload-shaped
# table from real grading data, parse it, and create model_monitoring.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

import os, sys
_here = os.getcwd()
_root = _here if os.path.exists(os.path.join(_here, "pyproject.toml")) else os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from pyspark.sql import functions as F
from pyspark.sql.types import (ArrayType, DoubleType, StringType,
                               StructField, StructType)
from grader.project_config import ProjectConfig

config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
base = config.base_path
PAYLOAD = f"{base}.grader_inference_payload"

# COMMAND ----------

# 1. Ensure a payload table exists in the request/response JSON shape Marvel uses.
#    (If the AI Gateway payload table is populated, point PAYLOAD at it instead.)
#    We build it from predictions so monitoring always has real data to parse.
spark.sql(f"""
CREATE OR REPLACE TABLE {PAYLOAD} AS
SELECT
    id AS databricks_request_id,
    created_at AS request_time,
    to_json(named_struct(
        'dataframe_records', array(named_struct(
            'student_id', student_id,
            'question_id', question_id,
            'answer_read', answer_read
        ))
    )) AS request,
    to_json(named_struct(
        'predictions', array(named_struct(
            'marks_awarded', marks_awarded,
            'max_marks', max_marks
        ))
    )) AS response,
    model_used AS model_name
FROM {base}.predictions
""")
print("Payload table ready:", PAYLOAD)
display(spark.sql(f"SELECT * FROM {PAYLOAD} LIMIT 5"))

# COMMAND ----------

# 2. Parse the request/response JSON (Marvel's from_json + explode pattern).
request_schema = StructType([
    StructField("dataframe_records", ArrayType(StructType([
        StructField("student_id", StringType(), True),
        StructField("question_id", StringType(), True),
        StructField("answer_read", StringType(), True),
    ])), True)
])
response_schema = StructType([
    StructField("predictions", ArrayType(StructType([
        StructField("marks_awarded", DoubleType(), True),
        StructField("max_marks", DoubleType(), True),
    ])), True)
])

payload = spark.table(PAYLOAD)
parsed = (payload
    .withColumn("req", F.from_json(F.col("request"), request_schema))
    .withColumn("resp", F.from_json(F.col("response"), response_schema)))

exploded = parsed.withColumn("rec", F.explode(F.col("req.dataframe_records")))

final = exploded.select(
    F.col("request_time").alias("timestamp"),
    F.col("databricks_request_id"),
    F.col("rec.student_id").alias("student_id"),
    F.col("rec.question_id").alias("question_id"),
    F.col("rec.answer_read").alias("answer_read"),
    F.col("resp.predictions")[0]["marks_awarded"].alias("marks_awarded"),
    F.col("resp.predictions")[0]["max_marks"].alias("max_marks"),
    F.col("model_name"),
)

# COMMAND ----------

# 3. Write the structured monitoring table (Marvel appends to model_monitoring).
(final.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{base}.model_monitoring"))

spark.sql(f"ALTER TABLE {base}.model_monitoring "
          f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

written = spark.table(f"{base}.model_monitoring").count()
print(f"model_monitoring rows: {written}")
display(spark.sql(f"SELECT * FROM {base}.model_monitoring ORDER BY timestamp DESC"))

# COMMAND ----------

# 4. Quality signals over the monitoring table (score level, zero-score rate).
display(spark.sql(f"""
SELECT
    to_date(timestamp)                              AS day,
    count(*)                                        AS predictions,
    round(avg(marks_awarded / max_marks), 3)        AS avg_score_fraction,
    sum(CASE WHEN marks_awarded = 0 THEN 1 ELSE 0 END) AS zero_score_count,
    model_name
FROM {base}.model_monitoring
GROUP BY to_date(timestamp), model_name
ORDER BY day
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## PAID/monitor: Lakehouse Quality Monitor
# MAGIC With model_monitoring built, a Databricks Quality Monitor can sit on top
# MAGIC (drift dashboards) exactly as in Marvel. That step uses
# MAGIC workspace.quality_monitors.create(...) on {base}.model_monitoring.