# Databricks notebook source
# =============================================================================
# 09_deploy_model.py  -  Re-register the model and UPDATE the serving endpoint.
#
# This is the "deploy_model" step (like Marvel's deploy notebook). Running it
# logs a fresh model version and updates the serving endpoint to that version,
# so every job run rolls the endpoint to a new version. Meant to run as the
# LAST task in the grading job, so a job run => endpoint update.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %pip install --upgrade "mlflow>=3.1"
# MAGIC %restart_python

# COMMAND ----------

import os, sys, glob, subprocess, json
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"
_here = os.getcwd()
_root = _here if os.path.exists(os.path.join(_here, "pyproject.toml")) else os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# COMMAND ----------

# Build the wheel (for code_paths).
subprocess.run([sys.executable, "-m", "pip", "install", "build", "--quiet"],
               capture_output=True, text=True)
subprocess.run([sys.executable, "-m", "build", "--wheel", _root],
               capture_output=True, text=True)
wheels = glob.glob(os.path.join(_root, "dist", "*.whl"))
wheel_path = sorted(wheels)[-1] if wheels else None

# COMMAND ----------

import mlflow
import mlflow.tracking._model_registry.utils
import pandas as pd
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
    lambda: "databricks-uc"
)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

from grader.project_config import ProjectConfig
config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
MODEL_NAME = f"{config.base_path}.grader_model"
ENDPOINT_NAME = "exam-grader-serving"

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/exam-grader-serving")

# COMMAND ----------

class GraderModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        from grader.models import Question, StudentAnswer
        from grader.grading import grade_answer
        import json
        results = []
        for _, row in model_input.iterrows():
            q = Question(**json.loads(row["question_json"]))
            a = StudentAnswer(**json.loads(row["answer_json"]))
            record = grade_answer(q, a)
            results.append({"marks_awarded": record.marks_awarded,
                            "max_marks": record.max_marks,
                            "justification": record.justification})
        return results

# COMMAND ----------

# 1. Log + register a NEW model version (this is what bumps the version).
example = pd.DataFrame([{
    "question_json": json.dumps({"id": "Q1", "type": "mcq",
        "text": "Capital of France?", "correct_answer": "Paris", "max_marks": 2}),
    "answer_json": json.dumps({"id": "A1", "question_id": "Q1",
        "student_id": "Riya", "answer_text": "Paris"}),
}])
signature = infer_signature(example,
    [{"marks_awarded": 2.0, "max_marks": 2.0, "justification": "..."}])

additional = ([f"code/{wheel_path.split('/')[-1]}"] if wheel_path else []) \
    + ["pydantic>=2", "requests", "python-dotenv"]
conda_env = _mlflow_conda_env(additional_pip_deps=additional)

import inspect
sig_params = set(inspect.signature(mlflow.pyfunc.log_model).parameters.keys())
kwargs = dict(python_model=GraderModel(), signature=signature, conda_env=conda_env)
kwargs["name" if "name" in sig_params else "artifact_path"] = "grader_model"
if wheel_path:
    kwargs["code_paths" if "code_paths" in sig_params else "code_path"] = [wheel_path]

with mlflow.start_run(run_name="job-redeploy"):
    model_info = mlflow.pyfunc.log_model(**kwargs)

from mlflow.utils.env_pack import EnvPackConfig
registered = mlflow.register_model(
    model_info.model_uri, MODEL_NAME,
    env_pack=EnvPackConfig(name="databricks_model_serving"),
)
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(MODEL_NAME, "latest-model", registered.version)
print(f"New model version: {registered.version}")

# COMMAND ----------

# 2. UPDATE the serving endpoint to the new version (this is the endpoint bump).
from mlflow.deployments import get_deploy_client
deploy = get_deploy_client("databricks")
endpoint_config = {
    "served_entities": [{
        "name": "grader-entity",
        "entity_name": MODEL_NAME,
        "entity_version": str(registered.version),
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    }],
}
try:
    deploy.update_endpoint(endpoint=ENDPOINT_NAME, config=endpoint_config)
    print(f"Endpoint '{ENDPOINT_NAME}' updated to version {registered.version}.")
except Exception as e:
    print("Update failed, trying create:", e)
    deploy.create_endpoint(name=ENDPOINT_NAME, config=endpoint_config)
    print(f"Endpoint '{ENDPOINT_NAME}' created at version {registered.version}.")