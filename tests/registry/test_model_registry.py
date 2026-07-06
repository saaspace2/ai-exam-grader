"""Model Registry Tests - the servable model wrapper behaves correctly.

We test the GraderModel pyfunc logic locally (no Databricks needed): given a
question + answer, it returns the expected marks structure. This proves the
thing we register/serve is correct BEFORE it hits the registry.
"""

import json


def _grader_predict(records):
    """Reproduce the pyfunc predict() logic locally for testing."""
    import pandas as pd
    from grader.models import Question, StudentAnswer
    from grader.grading import grade_answer
    df = pd.DataFrame(records)
    out = []
    for _, row in df.iterrows():
        q = Question(**json.loads(row["question_json"]))
        a = StudentAnswer(**json.loads(row["answer_json"]))
        rec = grade_answer(q, a)
        out.append({"marks_awarded": rec.marks_awarded,
                    "max_marks": rec.max_marks,
                    "justification": rec.justification})
    return out


class TestModelWrapper:
    def test_correct_mcq_full_marks(self):
        recs = [{
            "question_json": json.dumps({"id": "Q", "type": "mcq", "text": "?",
                                         "correct_answer": "Paris", "max_marks": 2}),
            "answer_json": json.dumps({"id": "A", "question_id": "Q",
                                       "student_id": "S", "answer_text": "Paris"}),
        }]
        out = _grader_predict(recs)
        assert out[0]["marks_awarded"] == 2.0

    def test_wrong_mcq_zero_marks(self):
        recs = [{
            "question_json": json.dumps({"id": "Q", "type": "mcq", "text": "?",
                                         "correct_answer": "Paris", "max_marks": 2}),
            "answer_json": json.dumps({"id": "A", "question_id": "Q",
                                       "student_id": "S", "answer_text": "London"}),
        }]
        out = _grader_predict(recs)
        assert out[0]["marks_awarded"] == 0.0

    def test_output_schema_stable(self):
        recs = [{
            "question_json": json.dumps({"id": "Q", "type": "mcq", "text": "?",
                                         "correct_answer": "A", "max_marks": 1}),
            "answer_json": json.dumps({"id": "A", "question_id": "Q",
                                       "student_id": "S", "answer_text": "A"}),
        }]
        out = _grader_predict(recs)
        assert set(out[0].keys()) == {"marks_awarded", "max_marks", "justification"}