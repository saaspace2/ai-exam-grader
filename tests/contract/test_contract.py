"""Contract Tests - the endpoint's input/output SHAPE must not change.

A contract is the agreed request/response format between the endpoint and its
consumers (your website). If it changes silently, consumers break. Needs
ENDPOINT_URL + DBR_TOKEN (else SKIP).
"""

import json


class TestOutputContract:
    def test_output_has_required_keys(self, call_endpoint, sample):
        # The contract: every prediction has these three keys.
        _, text = call_endpoint(sample)
        pred = json.loads(text)["predictions"][0]
        for key in ("marks_awarded", "max_marks", "justification"):
            assert key in pred, f"Contract broken: missing '{key}' in {pred}"

    def test_marks_are_numbers(self, call_endpoint, sample):
        _, text = call_endpoint(sample)
        pred = json.loads(text)["predictions"][0]
        assert isinstance(pred["marks_awarded"], (int, float))
        assert isinstance(pred["max_marks"], (int, float))

    def test_justification_is_string(self, call_endpoint, sample):
        _, text = call_endpoint(sample)
        pred = json.loads(text)["predictions"][0]
        assert isinstance(pred["justification"], str)