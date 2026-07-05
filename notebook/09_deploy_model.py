# Databricks notebook source
# =============================================================================
# 09_deploy_model.py  -  Update the serving endpoint to the latest model version.
#
# Runs as the last task in the grading job. Rather than re-logging the model with
# env_pack every run (which is fragile in a job), this reliably rolls the endpoint
# to the LATEST already-registered version of grader_model. To publish NEW grading
# code, run 07_serving.py (which logs a fresh version), then this promotes it.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %pip install --upgrade "mlflow>=3.1"
# MAGIC %restart_python

# COMMAND ----------

import os, sys
_here = os.getcwd()
_root = _here if os.path.exists(os.path.join(_here, "pyproject.toml")) else os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import mlflow
import mlflow.tracking._model_registry.utils
mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
    lambda: "databricks-uc"
)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

from grader.project_config import ProjectConfig
config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
MODEL_NAME = f"{config.base_path}.grader_model"
ENDPOINT_NAME = "exam-grader-serving"

# COMMAND ----------

# 1. Find the latest registered version of the model.
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(
        f"No registered versions of {MODEL_NAME}. Run 07_serving.py first to "
        f"log + register a servable version."
    )
latest = max(int(v.version) for v in versions)
print(f"Latest registered version: {latest}")

# COMMAND ----------

# 2. Update the serving endpoint to that version (rolls the endpoint).
from mlflow.deployments import get_deploy_client
deploy = get_deploy_client("databricks")
endpoint_config = {
    "served_entities": [{
        "name": "grader-entity",
        "entity_name": MODEL_NAME,
        "entity_version": str(latest),
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    }],
}
try:
    deploy.update_endpoint(endpoint=ENDPOINT_NAME, config=endpoint_config)
    print(f"Endpoint '{ENDPOINT_NAME}' updated to version {latest}.")
except Exception as e:
    print("Update failed, trying create:", e)
    deploy.create_endpoint(name=ENDPOINT_NAME, config=endpoint_config)
    print(f"Endpoint '{ENDPOINT_NAME}' created at version {latest}.")