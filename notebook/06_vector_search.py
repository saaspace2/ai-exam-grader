# Databricks notebook source
# =============================================================================
# 06_vector_search.py  -  Vector Search index for Clarify/RAG.
#
# Builds a Databricks Vector Search index over graded answers so Clarify can
# retrieve similar answers/explanations by meaning. Runs on Free Edition
# serverless (subject to the Vector Search quota). Mirrors the Marvel RAG stage.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %pip install databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

import os, sys
_here = os.getcwd()
_root = _here if os.path.exists(os.path.join(_here, "pyproject.toml")) else os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from grader.project_config import ProjectConfig
config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
base = config.base_path

ENDPOINT_NAME = "exam-grader-vs"
INDEX_NAME = f"{base}.answer_index"
SOURCE_TABLE = f"{base}.answer_index_source"

# COMMAND ----------

# 1. Build the source table: one row per graded answer, with its text.
#    Vector Search needs Change Data Feed on the source table.
spark.sql(f"""
CREATE OR REPLACE TABLE {SOURCE_TABLE} AS
SELECT
    a.id            AS answer_id,
    a.question_id,
    a.student_id,
    a.answer_text,
    g.marks_awarded,
    g.justification
FROM {base}.answers a
JOIN {base}.grades g ON a.id = g.answer_id
""")
spark.sql(f"ALTER TABLE {SOURCE_TABLE} "
          f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("Source table ready:", SOURCE_TABLE)
display(spark.sql(f"SELECT * FROM {SOURCE_TABLE} LIMIT 5"))

# COMMAND ----------

# 2. Create a Vector Search endpoint (holds indexes). Idempotent-ish.
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if ENDPOINT_NAME not in existing:
    vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Creating Vector Search endpoint '{ENDPOINT_NAME}' (takes a few minutes).")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists.")

# COMMAND ----------

# 3. Create the index over the source table (embeds answer_text).
#    Uses a Databricks-hosted embedding model - no key needed.
try:
    index = vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="answer_id",
        embedding_source_column="answer_text",
        embedding_model_endpoint_name="databricks-bge-large-en",
    )
    print(f"Creating index '{INDEX_NAME}'.")
except Exception as e:
    print("Index may already exist or is still provisioning:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for the index to be ONLINE, then query
# MAGIC The index takes a few minutes to build. Watch it under **Catalog** ->
# MAGIC your schema -> the index. Once online, run the query below.

# COMMAND ----------

# 4. Query the index: find answers similar in meaning to a phrase.
try:
    idx = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
    results = idx.similarity_search(
        query_text="capital city answer",
        columns=["answer_id", "question_id", "answer_text", "marks_awarded"],
        num_results=3,
    )
    print("Similar answers:")
    print(results)
except Exception as e:
    print("Index not online yet or error (wait, then retry):", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wiring into Clarify
# MAGIC retrieval.py has a VectorStore behind a `search()` interface. A Databricks
# MAGIC adapter would call `idx.similarity_search(...)` inside that same interface,
# MAGIC so Clarify uses this index instead of the local mock - no other changes.