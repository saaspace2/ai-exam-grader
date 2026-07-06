"""Chaos Tests - inject failures and confirm the system degrades gracefully.

Chaos engineering deliberately breaks things (network down, slow response) to
prove the system survives. Here we simulate the grading endpoint being
unreachable and confirm DatabricksGrader fails SAFELY (returns 0 + message,
does not crash). Offline, runs anywhere.
"""

from grader.databricks_grader import DatabricksGrader


class TestGracefulDegradation:
    def test_endpoint_down_does_not_crash(self):
        # Point at an unreachable URL - the grader must NOT raise, just fail safe.
        grader = DatabricksGrader(
            endpoint_url="https://127.0.0.1:1/invocations",  # nothing listening
            token="fake",
        )
        marks, justification, confidence = grader.grade_text(
            "Q?", "rubric", "answer", 2.0
        )
        # graceful: zero marks, an explanation, zero confidence - no exception
        assert marks == 0.0
        assert "error" in justification.lower()
        assert confidence == 0.0