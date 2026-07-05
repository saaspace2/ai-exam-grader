"""databricks_grader.py - an AIGrader that calls the Databricks serving endpoint.

Same interface as the other graders (grade_text -> marks, justification,
confidence), but instead of calling OpenRouter directly, it POSTs to the
Databricks Model Serving endpoint. This routes grading through the deployed
model, so the hosted website uses your Databricks endpoint.

Needs two env vars (set in Render / .env):
  DATABRICKS_ENDPOINT_URL  - the endpoint's invocations URL
  DATABRICKS_TOKEN         - a token to authenticate (PAT or SP token)
"""

import json
import os

import requests

from grader.grading import AIGrader


class DatabricksGrader(AIGrader):
    """Grade free-text answers by calling the Databricks serving endpoint."""

    def __init__(self, endpoint_url: str | None = None, token: str | None = None):
        self.endpoint_url = endpoint_url or os.environ.get("DATABRICKS_ENDPOINT_URL", "")
        self.token = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not self.endpoint_url or not self.token:
            raise ValueError(
                "DatabricksGrader needs DATABRICKS_ENDPOINT_URL and "
                "DATABRICKS_TOKEN (set them in the environment)."
            )

    def grade_text(self, question_text, rubric, answer_text, max_marks):
        """Send one answer to the endpoint and return (marks, justification, confidence).

        The endpoint's model expects question_json + answer_json columns, and
        returns marks_awarded / max_marks / justification. We map that back to
        this interface's (marks, justification, confidence) tuple.
        """
        question = {
            "id": "Q", "type": "short", "text": question_text,
            "rubric": rubric, "max_marks": max_marks,
        }
        answer = {
            "id": "A", "question_id": "Q",
            "student_id": "web", "answer_text": answer_text,
        }
        payload = {
            "dataframe_records": [{
                "question_json": json.dumps(question),
                "answer_json": json.dumps(answer),
            }]
        }

        try:
            resp = requests.post(
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
            # The serving response is usually {"predictions": [ {...} ]}
            preds = body.get("predictions", body)
            if isinstance(preds, list) and preds:
                pred = preds[0]
            elif isinstance(preds, dict):
                pred = preds
            else:
                pred = {}
            marks = float(pred.get("marks_awarded", 0.0))
            justification = pred.get("justification", "Graded by Databricks endpoint.")
            # the model does not return a confidence, so use a sensible default
            confidence = 0.9
            return marks, justification, confidence
        except Exception as e:
            # Fail safe: return 0 with an explanation rather than crashing the app.
            return 0.0, f"Databricks endpoint error: {e}", 0.0