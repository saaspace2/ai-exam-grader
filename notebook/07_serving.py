# Databricks notebook source
# =============================================================================
# 07_serving.py  -  Deploy the grader as a REAL serving endpoint (Marvel-style).
#
# Uses the same approach that worked in the Marvel project: build the package
# into a wheel and pass it via code_paths when logging the model. This avoids
# the python_env.yaml artifact-upload error, because the model's code comes
# from the wheel instead of an auto-generated environment file.
# =============================================================================

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

import os, sys
# Route UC model-artifact upload through the Databricks SDK (fixes AccessDenied).
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

_here = os.getcwd()
_root = _here if os.path.exists(os.path.join(_here, "pyproject.toml")) else os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# COMMAND ----------

# 1. Build the project into a wheel (like Marvel's dist/*.whl) so we can pass it
#    as code_paths. This packages our code cleanly for serving.
import subprocess
build_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "build", "--quiet"],
    capture_output=True, text=True
)
subprocess.run([sys.executable, "-m", "build", "--wheel", _root],
               capture_output=True, text=True)

import glob
wheels = glob.glob(os.path.join(_root, "dist", "*.whl"))
wheel_path = sorted(wheels)[-1] if wheels else None
print("Wheel:", wheel_path)

# COMMAND ----------

import mlflow
import mlflow.tracking._model_registry.utils
import pandas as pd
import json
from mlflow.models import infer_signature

# serverless registry-uri workaround
mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
    lambda: "databricks-uc"
)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

from grader.project_config import ProjectConfig
config = ProjectConfig.from_yaml("../project_config_grader.yml", env="dev")
MODEL_NAME = f"{config.base_path}.grader_model"
ENDPOINT_NAME = "exam-grader-serving"

# set the experiment under the user path (reliable on serverless)
username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/exam-grader-serving")
print("Model target:", MODEL_NAME)

# COMMAND ----------

# 2. The pyfunc model wrapping the grading engine.
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
            results.append({
                "marks_awarded": record.marks_awarded,
                "max_marks": record.max_marks,
                "justification": record.justification,
            })
        return results

# COMMAND ----------

# 3. Log + register the model, passing the wheel via code_paths (Marvel-style).
example = pd.DataFrame([{
    "question_json": json.dumps({"id": "Q1", "type": "mcq",
        "text": "Capital of France?", "correct_answer": "Paris", "max_marks": 2}),
    "answer_json": json.dumps({"id": "A1", "question_id": "Q1",
        "student_id": "Riya", "answer_text": "Paris"}),
}])
sample_output = GraderModel().predict(None, example)
signature = infer_signature(example, sample_output)

base_kwargs = dict(
    python_model=GraderModel(),
    registered_model_name=MODEL_NAME,
    input_example=example,
    signature=signature,
)

# The wheel param name changed across MLflow versions:
#   older: code_paths=[...]   newer (3.x): code_path=[...]
# And the model path arg changed: older 'artifact_path' -> newer 'name'.
# We try combinations until one is accepted, so it works on any version.
import inspect
sig_params = set(inspect.signature(mlflow.pyfunc.log_model).parameters.keys())

kwargs = dict(base_kwargs)
# model path argument
if "name" in sig_params:
    kwargs["name"] = "grader_model"
else:
    kwargs["artifact_path"] = "grader_model"
# wheel/code argument
if wheel_path:
    if "code_paths" in sig_params:
        kwargs["code_paths"] = [wheel_path]
    elif "code_path" in sig_params:
        kwargs["code_path"] = [wheel_path]

print("Using log_model kwargs:", list(kwargs.keys()))
with mlflow.start_run(run_name="register-grader-model"):
    mlflow.pyfunc.log_model(**kwargs)
print("Model logged + registered:", MODEL_NAME)

# COMMAND ----------

# 4. Find the latest version.
from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print("Latest version:", latest)

# COMMAND ----------

# 5. Deploy the serving endpoint.
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
    deploy.create_endpoint(name=ENDPOINT_NAME, config=endpoint_config)
    print(f"Creating endpoint '{ENDPOINT_NAME}' (a few minutes to READY).")
except Exception as e:
    print("Create failed (may exist), updating:", e)
    deploy.update_endpoint(endpoint=ENDPOINT_NAME, config=endpoint_config)
    print(f"Updated endpoint '{ENDPOINT_NAME}'.")

# COMMAND ----------

# 6. Query the endpoint once it's READY.
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
    print("Not ready yet or error (wait for READY, then retry):", e)
