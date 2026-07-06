"""Smoke Tests - is the grading endpoint ALIVE? (Run these FIRST after deploy.)

Analogy: plugging in a TV - you are not checking 4K quality yet, just that it
turns on. If smoke tests fail, skip all other endpoint tests - the system is down.

Needs ENDPOINT_URL + DBR_TOKEN (else these SKIP). Uses fixtures from
conftest_endpoint.py.
"""

import json


class TestLiveness:
    """Layer 1 - is it alive at all?"""

    def test_endpoint_returns_200(self, call_endpoint, sample):
        # 200 = the endpoint accepted the request and responded. Anything else
        # (404 not deployed, 401 bad token, 500 crash) means the deploy failed.
        status, _ = call_endpoint(sample)
        assert status == 200, f"Endpoint returned {status}, expected 200."

    def test_response_not_empty(self, call_endpoint, sample):
        # A 200 with an empty body is a silent failure - worse than an error.
        _, text = call_endpoint(sample)
        assert len(text) > 0, "Endpoint returned an empty body."

    def test_response_is_valid_json(self, call_endpoint, sample):
        # Databricks serving always returns JSON; HTML back = infra problem.
        _, text = call_endpoint(sample)
        json.loads(text)  # raises if not valid JSON


class TestResponseShape:
    """Layer 2 - does the response look right?"""

    def test_predictions_key_exists(self, call_endpoint, sample):
        # 'predictions' is the contract with every consumer. Missing = the model
        # output schema changed (a breaking change to catch immediately).
        _, text = call_endpoint(sample)
        data = json.loads(text)
        assert "predictions" in data, f"No 'predictions' key. Keys: {list(data.keys())}"

    def test_prediction_has_marks(self, call_endpoint, sample):
        # Our model returns marks_awarded / max_marks / justification per record.
        _, text = call_endpoint(sample)
        preds = json.loads(text)["predictions"]
        first = preds[0] if isinstance(preds, list) else preds
        assert "marks_awarded" in first, f"No marks_awarded in prediction: {first}"