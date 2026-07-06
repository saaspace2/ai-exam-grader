"""Monitoring Tests - verify our monitoring signals compute correctly.

Monitoring watches production over time (score drift, appeal rate). These tests
check the MATH of those signals on known data, so a broken metric is caught
before it misleads us. Offline.
"""


def _avg_score_fraction(grades):
    """The monitoring metric: mean of marks_awarded / max_marks."""
    if not grades:
        return 0.0
    return sum(g["marks_awarded"] / g["max_marks"] for g in grades) / len(grades)


def _zero_score_rate(grades):
    if not grades:
        return 0.0
    return sum(1 for g in grades if g["marks_awarded"] == 0) / len(grades)


class TestMonitoringMetrics:
    def test_avg_score_fraction(self):
        grades = [{"marks_awarded": 2, "max_marks": 2},
                  {"marks_awarded": 1, "max_marks": 2}]
        # (1.0 + 0.5) / 2 = 0.75
        assert _avg_score_fraction(grades) == 0.75

    def test_zero_score_rate(self):
        grades = [{"marks_awarded": 0, "max_marks": 2},
                  {"marks_awarded": 2, "max_marks": 2}]
        assert _zero_score_rate(grades) == 0.5

    def test_empty_is_safe(self):
        # No data must not divide by zero.
        assert _avg_score_fraction([]) == 0.0
        assert _zero_score_rate([]) == 0.0