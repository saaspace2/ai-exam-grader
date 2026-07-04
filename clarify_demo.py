"""Demo: grade an answer, store it, then ask 'why this mark?' (Clarify / RAG).

Usage:
    python clarify_demo.py
"""
from grader.store import Store
from grader.models import Question, StudentAnswer
from grader.grading import grade_answer, MockAIGrader
from grader.retrieval import clarify, build_answer_index
from grader.config import settings


def main():
    s = Store(":memory:")  # temporary DB for the demo

    # set up and grade one essay answer
    q = Question(id="Q4", type="essay", text="Explain photosynthesis.",
                 rubric="sunlight water carbon dioxide glucose oxygen", max_marks=10)
    a = StudentAnswer(id="A4", question_id="Q4", student_id="Riya",
                      answer_text="Plants use sunlight and water to make glucose and release oxygen.")
    s.save_question(q)
    s.save_answer(a)
    g = grade_answer(q, a, ai=MockAIGrader())
    s.save_grade(g)

    print(f"Riya was graded {g.marks_awarded}/{g.max_marks} on Q4.\n")

    # CLARIFY: why this mark?
    explainer = None if settings.has_api_key() else __import__(
        "grader.retrieval", fromlist=["MockClarifyExplainer"]).MockClarifyExplainer()
    print("Student asks: 'Why did I get this mark?'\n")
    print("Clarify answer:")
    print(" ", clarify(s, "Riya", "Q4", "Why did I get this mark?", explainer=explainer))

    # VECTOR SEARCH demo: find by meaning
    print("\nFuzzy search: 'how plants make food using light'")
    for hit in build_answer_index(s).search("how plants make food using light", top_k=1):
        print(f"  best match [{hit.key}] score {hit.score:.2f}: {hit.text[:50]}...")


if __name__ == "__main__":
    main()