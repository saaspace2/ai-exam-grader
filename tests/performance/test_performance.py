"""Performance / Load Tests - can the endpoint handle repeated requests?

We keep this SMALL on Free Edition (quotas!) - a handful of requests, not
Marvel's 18,000. Measures p95 latency. Needs ENDPOINT_URL + DBR_TOKEN (else SKIP).
"""

import time


class TestLoad:
    def test_ten_requests_all_succeed(self, call_endpoint, sample):
        # Ten back-to-back requests must all return 200 (stable deployment).
        failures = 0
        for _ in range(10):
            status, _ = call_endpoint(sample)
            if status != 200:
                failures += 1
        assert failures == 0, f"{failures}/10 requests failed."

    def test_p95_latency_reasonable(self, call_endpoint, sample):
        # Measure 10 requests; the 95th-percentile time should be sane.
        times = []
        for _ in range(10):
            start = time.time()
            call_endpoint(sample)
            times.append(time.time() - start)
        p95 = sorted(times)[int(len(times) * 0.95) - 1]
        # generous threshold for Free Edition serverless (cold starts happen)
        assert p95 < 10.0, f"p95 latency {p95:.1f}s is too high."