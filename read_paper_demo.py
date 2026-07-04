"""Demo: read a student's paper answer from an image, then grade it.

Usage:
    python read_paper_demo.py path/to/answer_image.png

With no real OPENROUTER_API_KEY set, this uses the MockImageReader (offline),
so it returns placeholder text. Set a real key in .env to read a real image
with a free vision model.
"""

import sys
from pathlib import Path

from grader.vision import read_answer_from_image, get_image_reader, MockImageReader
from grader.ingestion import ingest_question, ingest_answer
from grader.grading import grade_answer
from grader.config import settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_paper_demo.py <image_path>")
        return
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        return

    if settings.has_api_key():
        print(f"Reading with REAL vision model ({settings.OPENROUTER_MODEL})...")
        reader = None  # get_image_reader() will pick the real one
    else:
        print("No API key set - using MockImageReader (offline placeholder text).")
        reader = MockImageReader(student_id="Riya", answer_text="Paris")

    # STEP 0: read the paper
    raw = read_answer_from_image(image_path, question_id="Q1",
                                 answer_id="A1", reader=reader)
    print("\nRead from paper:")
    print("  student_id :", raw["student_id"])
    print("  answer_text:", raw["answer_text"])

    # STEP 1+2: ingest + grade against an example question
    q = ingest_question({"id": "Q1", "type": "mcq", "text": "Capital of France?",
                         "correct_answer": "Paris", "max_marks": 2})
    a = ingest_answer(raw)
    g = grade_answer(q, a)
    print(f"\nGraded: {g.student_id} scored {g.marks_awarded}/{g.max_marks} ({g.grading_method})")
    print(f"Why: {g.justification}")


if __name__ == "__main__":
    main()