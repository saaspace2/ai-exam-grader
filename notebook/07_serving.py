# Databricks notebook source
# =============================================================================
# 07_serving.py  -  Deploy the grader as a REAL Model Serving endpoint.
#
# Wraps the grading engine as an MLflow pyfunc model, registers it in Unity
# Catalog, and deploys a Model Serving endpoint (a REST API). Runs on Databricks
# Free Edition serverless (subject to the serving quota - keep test traffic small).
# Mirrors the Marvel serving stage.
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

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from grader.models import Question, StudentAnswer
from grader.grading import grade_answer
from grader.project_config import ProjectConfig

config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
<<<<<<< HEAD
# Serverless workaround: set the registry URI manually (spark config not available).
import mlflow.tracking._model_registry.utils
mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
    lambda: "databricks-uc"
)
=======
>>>>>>> 70be3bf789c22875a5c640e69a5cfc4698ad2630
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")   # register models in Unity Catalog
MODEL_NAME = f"{config.base_path}.grader_model"
ENDPOINT_NAME = "exam-grader-serving"
print("Model:", MODEL_NAME)

# COMMAND ----------

# 1. Wrap the grading engine as an MLflow pyfunc model.
class GraderModel(mlflow.pyfunc.PythonModel):
    """Serve the grading engine. Input columns: question_json, answer_json."""

    def predict(self, context, model_input):
        import json
        results = []
        for _, row in model_input.iterrows():
            q = Question(**json.loads(row["question_json"]))
            a = StudentAnswer(**json.loads(row["answer_json"]))
            record = grade_answer(q, a)
            results.append({
                "marks_awarded": record.marks_awarded,
                "max_marks": record.max_marks,
                "justification": record.justification,
            })
        return results

# COMMAND ----------

# 2. Build an example input (for the model signature) and log the model.
import json
example = pd.DataFrame([{
    "question_json": json.dumps({"id": "Q1", "type": "mcq",
        "text": "Capital of France?", "correct_answer": "Paris", "max_marks": 2}),
    "answer_json": json.dumps({"id": "A1", "question_id": "Q1",
        "student_id": "Riya", "answer_text": "Paris"}),
}])
sample_output = GraderModel().predict(None, example)
signature = infer_signature(example, sample_output)

with mlflow.start_run(run_name="register-grader-model"):
    mlflow.pyfunc.log_model(
        artifact_path="grader_model",
        python_model=GraderModel(),
        registered_model_name=MODEL_NAME,
        input_example=example,
        signature=signature,
        pip_requirements=["pydantic", "requests", "python-dotenv"],
    )
print("Model logged + registered in Unity Catalog:", MODEL_NAME)

# COMMAND ----------

# 3. Find the latest version of the registered model.
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print("Latest model version:", latest)

# COMMAND ----------

# 4. Deploy (or update) the serving endpoint via the MLflow Deployments SDK.
from mlflow.deployments import get_deploy_client
deploy = get_deploy_client("databricks")

endpoint_config = {
    "served_entities": [{
        "name": "grader-entity",
        "entity_name": MODEL_NAME,
        "entity_version": str(latest),
        "workload_size": "Small",
        "scale_to_zero_enabled": True,   # good for Free Edition - no idle cost
    }],
}

try:
    deploy.create_endpoint(name=ENDPOINT_NAME, config=endpoint_config)
    print(f"Creating endpoint '{ENDPOINT_NAME}' (takes a few minutes to be READY).")
except Exception as e:
    # if it already exists, update it to the new version
    print("Create failed (may already exist), updating instead:", e)
    deploy.update_endpoint(endpoint=ENDPOINT_NAME, config=endpoint_config)
    print(f"Updated endpoint '{ENDPOINT_NAME}'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for READY, then query
# MAGIC The endpoint takes a few minutes to reach **READY** (watch it under
# MAGIC **Serving** in the sidebar). Once ready, run the cell below to query it.

# COMMAND ----------

# 5. Query the live endpoint (only works once it's READY).
try:
    response = deploy.predict(
        endpoint=ENDPOINT_NAME,
        inputs={"dataframe_records": [{
            "question_json": json.dumps({"id": "Q1", "type": "mcq",
                "text": "Capital of Japan?", "correct_answer": "Tokyo", "max_marks": 2}),
            "answer_json": json.dumps({"id": "A2", "question_id": "Q1",
                "student_id": "Sara", "answer_text": "Tokyo"}),
        }]},
    )
    print("Endpoint response:", response)
except Exception as e:
    print("Endpoint not ready yet or error (wait for READY then retry):", e)
