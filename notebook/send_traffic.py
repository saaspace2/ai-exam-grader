# Databricks notebook source
# =============================================================================
# 08_send_traffic.py  -  Send requests to the endpoint (Marvel-style).
#
# This generates traffic so the inference/payload table populates. Marvel's
# monitoring notebook did exactly this: build records, loop, POST to the
# endpoint. Run this AFTER the endpoint is READY and inference tables are on.
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

# COMMAND ----------

import os, json, time, requests
from pyspark.dbutils import DBUtils

# Get the token + host the SAME way Marvel does (from the notebook context).
dbutils = DBUtils(spark)
os.environ["DBR_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook()
    .getContext().apiToken().get()
)
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

ENDPOINT_NAME = "exam-grader-serving"
print("Host:", os.environ["DBR_HOST"])

# COMMAND ----------

def call_endpoint(record):
    """POST one record to the serving endpoint (Marvel's send_request_https)."""
    url = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": [record]},
    )
    return response.status_code, response.text

# COMMAND ----------

# Build a batch of varied records (some right, some wrong answers).
samples = [
    ("Tokyo", "Tokyo"),      # correct
    ("Osaka", "Tokyo"),      # wrong
    ("Paris", "Paris"),      # correct
    ("London", "Paris"),     # wrong
    ("Rome", "Rome"),        # correct
]
records = []
for i, (student_ans, correct) in enumerate(samples):
    records.append({
        "question_json": json.dumps({"id": f"Q{i}", "type": "mcq",
            "text": "Capital?", "correct_answer": correct, "max_marks": 2}),
        "answer_json": json.dumps({"id": f"A{i}", "question_id": f"Q{i}",
            "student_id": f"student_{i}", "answer_text": student_ans}),
    })

# COMMAND ----------

# Smoke test: one request first.
status, text = call_endpoint(records[0])
print("Smoke test status:", status)
print("Response:", text)

# COMMAND ----------

# Send the batch in a loop (Marvel loops with a small sleep to pace requests).
for i, rec in enumerate(records):
    status, text = call_endpoint(rec)
    print(f"Request {i}: status={status} -> {text}")
    time.sleep(0.5)

print("\nDone. Requests sent. The inference/payload table will populate in a")
print("few minutes (Databricks batches the writes). Then check:")
print(f"  SHOW TABLES IN exam_grader_dev.grading  (look for a *_payload table)")

# COMMAND ----------

# Run this a bit later to see the payload table once it appears.
# display(spark.sql("SHOW TABLES IN exam_grader_dev.grading"))
# display(spark.sql("SELECT * FROM exam_grader_dev.grading.grader_inference_payload"))