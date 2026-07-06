"""Serving Endpoint Tests - deeper than smoke: value correctness, latency, edges.

p95 latency = out of 100 requests, 95 finished within this time. Needs
ENDPOINT_URL + DBR_TOKEN (else SKIP).
"""

import json
import time


class TestPredictionValues:
    def test_correct_answer_gets_full_marks(self, call_endpoint, sample):
        # sample is a correct MCQ (Paris) worth 2 -> should score 2/2.
        _, text = call_endpoint(sample)
        pred = json.loads(text)["predictions"][0]
        assert pred["marks_awarded"] == pred["max_marks"], (
            f"Correct answer did not get full marks: {pred}"
        )

    def test_marks_never_exceed_max(self, call_endpoint, sample):
        _, text = call_endpoint(sample)
        pred = json.loads(text)["predictions"][0]
        assert pred["marks_awarded"] <= pred["max_marks"], "Marks exceeded the max!"


class TestLatency:
    def test_single_request_under_5s(self, call_endpoint, sample):
        # A single grade should return quickly once the endpoint is warm.
        start = time.time()
        status, _ = call_endpoint(sample)
        elapsed = time.time() - start
        assert status == 200
        assert elapsed < 5.0, f"Took {elapsed:.1f}s (>5s). Cold start? Re-run to confirm."


class TestEdgeCases:
    def test_multiple_records_at_once(self, call_endpoint, sample):
        # Batch of 3 - must all grade in one call.
        status, text = call_endpoint(sample * 3)
        assert status == 200
        assert len(json.loads(text)["predictions"]) == 3