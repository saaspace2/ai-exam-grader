# Databricks notebook source
# =============================================================================
# 01_setup_and_ingest.py  -  Databricks pipeline notebook (Marvel-style).
#
# FLOW (mirrors the Marvel ingestion notebook):
#   1. %pip install -e ..        -> make the `grader` package importable
#   2. Load ProjectConfig for an environment (dev/acc/prd)
#   3. DataProcessor.setup_infrastructure()  -> create catalog/schema/tables/volume
#   4. Enable Change Data Feed
#   5. (Optional) write a sample question and show counts
#
# Runs on Free Edition serverless - it's just notebook code, no job clusters.
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

# Load the config for this environment. Change env to "acc" or "prd" to target
# those environments with the SAME code (build-once-deploy-many).
from grader.project_config import ProjectConfig

config = ProjectConfig.from_yaml(
    config_path="../project_config_grader.yml", env="dev"
)
print("Configuration loaded:")
print("  catalog :", config.catalog_name)
print("  schema  :", config.schema_name)
print("  tables  :", config.tables)

# COMMAND ----------

# Create the schema, 5 Delta tables, and the volume.
# NOTE: the CATALOG (exam_grader_dev) must already exist - you create it in the
# Databricks UI first (Marvel-style). The notebook uses it, it does not create it.
from grader.data_processor import DataProcessor

processor = DataProcessor(config, spark)
processor.setup_infrastructure(create_catalog=False)

# COMMAND ----------

# Enable Delta Change Data Feed so future changes to these tables are tracked
# (Marvel-style auditing).
processor.enable_change_data_feed()

# COMMAND ----------

# Write one sample question so we can confirm the write path works end-to-end.
sample_questions = [{
    "id": "Q1", "type": "mcq", "text": "Capital of France?",
    "correct_answer": "Paris", "rubric": None, "tolerance": 0.0, "max_marks": 2.0,
}]
processor.save_questions(sample_questions)

# COMMAND ----------

# Show row counts for every table, and read the sample back.
processor.show_counts()
display(spark.sql(f"SELECT * FROM {config.base_path}.questions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC Open the **Catalog** browser and you'll see
# MAGIC `exam_grader_dev` > `grading` > your five tables, with one question in `questions`.
