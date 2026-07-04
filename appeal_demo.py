"""Demo: the full re-evaluation flow - raise doubt, re-evaluate, clarify.

Usage:
    python appeal_demo.py
"""
from grader.store import Store
from grader.models import Question, StudentAnswer
from grader.grading import grade_answer, MockAIGrader, AIGrader
from grader.appeals import raise_doubt, reevaluate, confirm_reevaluation, clarify_result
from grader.retrieval import MockClarifyExplainer


class FixedGrader(AIGrader):
    """A grader returning a chosen mark, to show the flow clearly."""
    def __init__(self, marks): self.marks = marks
    def grade_text(self, q, r, a, m): return (self.marks, f"Re-check gave {self.marks}.", 0.9)


def main():
    s = Store(":memory:")
    q = Question(id="Q4", type="essay", text="Explain photosynthesis.",
                 rubric="sunlight water glucose oxygen", max_marks=10)
    a = StudentAnswer(id="A4", question_id="Q4", student_id="Riya",
                      answer_text="Plants use sunlight and water to make glucose.")
    s.save_question(q); s.save_answer(a)
    g = grade_answer(q, a, ai=MockAIGrader()); s.save_grade(g)
    print(f"Initial grade: {g.marks_awarded}/10\n")

    # 1. student raises a doubt
    print("1. Riya raises a doubt on Q4 (reason: 'I think I said more').")
    raise_doubt(s, "Riya", "Q4", reason="I think I said more than credited")

    # 2. re-evaluate - here we simulate the AI finding a HIGHER mark
    print("2. Re-evaluating...")
    res = reevaluate(s, "Riya", "Q4", ai=FixedGrader(8.0))
    print("   ->", res.message)

    # if it had been lower, we'd confirm:
    if res.needs_confirmation:
        print("   (mark would drop - asking student to confirm)")
        res = confirm_reevaluation(s, "Riya", "Q4", res.new_marks, "Confirmed.")
        print("   ->", res.message)

    # 3. clarify the final result
    print("\n3. Clarification of the final mark:")
    print("  ", clarify_result(s, "Riya", "Q4", explainer=MockClarifyExplainer()))

    # show the audit trail
    print("\nFull audit trail (tamper-proof history):")
    for h in s.get_audit_history(g.id):
        print(f"  [{h.actor.value:7}] {h.action:14} {h.old_marks} -> {h.new_marks}")


if __name__ == "__main__":
    main()