# Databricks notebook source
# =============================================================================
# 09_deploy_model.py  -  CD deploy step: re-register + roll the endpoint.
#
# IDENTICAL logic to 07_serving.py (which produced the working version): build a
# wheel, reference it as code/<wheel> in an explicit conda_env, pass code_paths +
# conda_env to log_model, and register with env_pack (falling back to plain
# register if env_pack fails). This is the version proven to be SERVABLE. Runs as
# the last task of the grading job so each CD run rolls the endpoint.
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

# COMMAND ----------

# 1. Build the wheel (Marvel builds it with uv build into dist/; we use python -m build).
import subprocess, glob
subprocess.run([sys.executable, "-m", "pip", "install", "build", "--quiet"],
               capture_output=True, text=True)
subprocess.run([sys.executable, "-m", "build", "--wheel", _root],
               capture_output=True, text=True)
wheels = glob.glob(os.path.join(_root, "dist", "*.whl"))
wheel_path = sorted(wheels)[-1] if wheels else None
print("Wheel:", wheel_path)

# COMMAND ----------

import mlflow
import mlflow.tracking._model_registry.utils
import pandas as pd
import json
from datetime import datetime
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

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

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/exam-grader-serving")
print("Model target:", MODEL_NAME)

# COMMAND ----------

# 2. The pyfunc wrapper around the grading engine.
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

# 3. Log + register - Marvel's exact method: explicit conda_env referencing the
#    wheel as code/<wheel>, passed alongside code_paths. This avoids the
#    auto-generated python_env.yaml that fails to upload on Free Edition.
example = pd.DataFrame([{
    "question_json": json.dumps({"id": "Q1", "type": "mcq",
        "text": "Capital of France?", "correct_answer": "Paris", "max_marks": 2}),
    "answer_json": json.dumps({"id": "A1", "question_id": "Q1",
        "student_id": "Riya", "answer_text": "Paris"}),
}])
signature = infer_signature(
    model_input=example,
    model_output=[{"marks_awarded": 2.0, "max_marks": 2.0, "justification": "..."}],
)

# Build conda_env that references the wheel as code/<wheel> (Marvel's trick).
additional_pip_deps = []
if wheel_path:
    whl_name = wheel_path.split("/")[-1]
    additional_pip_deps.append(f"code/{whl_name}")
# also add runtime deps the model needs
additional_pip_deps += ["pydantic>=2", "requests", "python-dotenv"]
conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

# handle the code_paths vs code_path param name across MLflow versions
import inspect
sig_params = set(inspect.signature(mlflow.pyfunc.log_model).parameters.keys())
kwargs = dict(
    python_model=GraderModel(),
    signature=signature,
    conda_env=conda_env,
)
# model path arg
if "name" in sig_params:
    kwargs["name"] = "grader_model"
else:
    kwargs["artifact_path"] = "grader_model"
# wheel arg
if wheel_path:
    if "code_paths" in sig_params:
        kwargs["code_paths"] = [wheel_path]
    elif "code_path" in sig_params:
        kwargs["code_path"] = [wheel_path]

print("log_model kwargs:", list(kwargs.keys()))
with mlflow.start_run(run_name=f"grader-wrapper-{datetime.now().strftime('%Y-%m-%d')}"):
    model_info = mlflow.pyfunc.log_model(**kwargs)

# register from the logged model uri.
# SERVERLESS-OPTIMIZED: use env_pack so MLflow packages + stages the model
# artifacts and the serverless environment properly for serving. This is the
# official fix for serverless deployments and avoids the managed-storage
# artifact-upload AccessDenied seen with a plain register_model.
try:
    from mlflow.utils.env_pack import EnvPackConfig
    registered = mlflow.register_model(
        model_info.model_uri, MODEL_NAME,
        env_pack=EnvPackConfig(name="databricks_model_serving"),
    )
    print("Registered with env_pack (serverless-optimized).")
except Exception as e:
    # Fallback: plain register (older MLflow without env_pack).
    print("env_pack not available, using plain register_model:", e)
    registered = mlflow.register_model(model_uri=model_info.model_uri, name=MODEL_NAME)

client = MlflowClient()
client.set_registered_model_alias(name=MODEL_NAME, alias="latest-model",
                                  version=registered.version)
print(f"Registered version {registered.version} with alias latest-model.")

# COMMAND ----------

# 4. Deploy the serving endpoint.
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
    deploy.create_endpoint(name=ENDPOINT_NAME, config=endpoint_config)
    print(f"Creating endpoint '{ENDPOINT_NAME}' (a few minutes to READY).")
except Exception as e:
    print("Create failed (may exist), updating:", e)
    deploy.update_endpoint(endpoint=ENDPOINT_NAME, config=endpoint_config)
    print(f"Updated endpoint '{ENDPOINT_NAME}'.")

# COMMAND ----------

# 5. Query once READY.
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

    print("Not ready yet (wait for READY, then retry):", e)

    print("Not ready yet (wait for READY, then retry):", e)



