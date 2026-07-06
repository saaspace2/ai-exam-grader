"""Canary Tests - send a TINY slice of traffic to a new version, watch it.

Named after canaries in coal mines. You route (say) 5% of traffic to the new
grader; if error rate stays low, you roll out wider. This test simulates the
canary decision: given sample results, decide whether the canary is healthy.
Offline, runs anywhere.
"""


def _canary_healthy(results, max_error_rate=0.1):
    """A canary is healthy if its error rate is below the threshold."""
    if not results:
        return False
    errors = sum(1 for r in results if r.get("error"))
    return (errors / len(results)) <= max_error_rate


class TestCanaryDecision:
    def test_healthy_canary_promotes(self):
        # 10 requests, 0 errors -> healthy -> promote.
        results = [{"marks": 2} for _ in range(10)]
        assert _canary_healthy(results) is True

    def test_unhealthy_canary_blocks(self):
        # 10 requests, 3 errors (30%) -> unhealthy -> do NOT promote.
        results = [{"error": True}] * 3 + [{"marks": 2}] * 7
        assert _canary_healthy(results) is False