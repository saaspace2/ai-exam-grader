# Databricks notebook source
# =============================================================================
# 04_mlflow_tracking.py  -  Track each grading run as an MLflow experiment.
#
# Our system doesn't TRAIN a model, so instead of training metrics we log
# GRADING-run metrics: how many graded, average score, pass rate, appeal rate,
# which grader/model. Every batch run becomes a tracked, comparable experiment -
# the same MLflow pattern as the Marvel project, applied to grading.
#
# MLflow works on Databricks Free Edition.
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

import mlflow
import mlflow.tracking._model_registry.utils
from grader.project_config import ProjectConfig
from grader.delta_store import DeltaStore
from grader.batch_grader import BatchGrader

config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
store = DeltaStore(config, spark)

<<<<<<< HEAD
# --- Serverless workaround -------------------------------------------------
# On serverless compute, spark.mlflow.modelRegistryUri is not set, so MLflow
# raises CONFIG_NOT_AVAILABLE when it tries to read it. We set the registry URI
# manually by patching the lookup function. (Known Databricks serverless fix.)
mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
    lambda: "databricks"
)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")
# ---------------------------------------------------------------------------

# Use the current user's workspace folder for the experiment (reliable on serverless).
username = spark.sql("SELECT current_user()").collect()[0][0]
=======
# Point MLflow at the Databricks tracking server explicitly (needed on serverless).
mlflow.set_tracking_uri("databricks")

# Use the current user's workspace folder for the experiment - more reliable on
# serverless than /Shared. Build the path from the logged-in user's name.
username = (
    spark.sql("SELECT current_user()").collect()[0][0]
)
>>>>>>> 70be3bf789c22875a5c640e69a5cfc4698ad2630
experiment_path = f"/Users/{username}/exam-grader-grading-runs"
experiment = mlflow.set_experiment(experiment_path)
print("Experiment:", experiment.name)

# COMMAND ----------

# Run one grading batch INSIDE an MLflow run, logging params + metrics.
with mlflow.start_run(run_name="batch-grading-run") as run:
    # log the settings used (like Marvel logging hyperparameters)
    mlflow.log_params({
        "environment": "dev",
        "catalog": config.catalog_name,
        "grader": config.grader,
        "model_name": config.model_name,
    })

    # run the batch grader
    summary = BatchGrader(store).grade_all()

    # compute a few grading metrics from the grades table
    stats = spark.sql(f"""
        SELECT
            count(*)                             AS total_grades,
            avg(marks_awarded / max_marks)       AS avg_score_fraction,
            sum(CASE WHEN marks_awarded > 0 THEN 1 ELSE 0 END) AS non_zero_count
        FROM {config.base_path}.grades
    """).collect()[0].asDict()

    # appeals: how many grades are under appeal / changed (from audit)
    appeals = spark.sql(f"""
        SELECT count(DISTINCT grade_record_id) AS appealed
        FROM {config.base_path}.audit
        WHERE action LIKE '%reevaluat%' OR action LIKE '%doubt%'
    """).collect()[0]["appealed"]

    # log the metrics
    mlflow.log_metrics({
        "graded_this_run": summary["graded"],
        "total_grades": stats["total_grades"] or 0,
        "avg_score_fraction": float(stats["avg_score_fraction"] or 0.0),
        "non_zero_count": stats["non_zero_count"] or 0,
        "appealed_count": appeals or 0,
    })

    print("Logged run:", run.info.run_id)
    print("Summary:", summary)
    print("Avg score fraction:", stats["avg_score_fraction"])

# COMMAND ----------

# Query past runs (like Marvel's search_runs) to compare over time.
runs = mlflow.search_runs(experiment_names=[experiment_path])
display(runs[["run_id", "metrics.graded_this_run", "metrics.avg_score_fraction",
              "metrics.appealed_count"]] if len(runs) else runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we tracked
# MAGIC Each grading run logs its settings (params) and outcomes (metrics:
# MAGIC graded count, average score, appeal count). Over time you can compare
# MAGIC runs in the MLflow UI - spotting if scores drift or appeals rise.
