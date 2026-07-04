"""Tests for the serving API (Component 7).

Uses FastAPI's TestClient (no real server needed) and a temporary database
file so tests never touch real data. All offline via the mock grader.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import grader.config as cfg
from grader.api import app


@pytest.fixture
def client(monkeypatch):
    """A test client pointed at a throwaway database file."""
    db = os.path.join(tempfile.gettempdir(), "test_api.sqlite")
    if os.path.exists(db):
        os.remove(db)
    monkeypatch.setattr(cfg.settings, "GRADER_DB_PATH", db)
    # Force EVERYTHING offline/mock for tests: blank the key on the settings
    # object so the grader, vision, and clarify factories all pick the mock.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cfg.settings, "OPENROUTER_API_KEY", "")
    yield TestClient(app)
    if os.path.exists(db):
        os.remove(db)


def _grade_payload():
    return {
        "question": {"id": "Q1", "type": "mcq", "text": "Capital of France?",
                     "correct_answer": "Paris", "max_marks": 2},
        "answer": {"id": "A1", "question_id": "Q1", "student_id": "Riya",
                   "answer_text": "Paris"},
    }


class TestHealth:
    def test_health_is_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestGrade:
    def test_grade_returns_marks(self, client):
        r = client.post("/grade", json=_grade_payload())
        assert r.status_code == 200
        assert r.json()["marks_awarded"] == 2

    def test_grade_then_fetch(self, client):
        client.post("/grade", json=_grade_payload())
        r = client.get("/grade/Riya/Q1")
        assert r.status_code == 200
        assert r.json()["student_id"] == "Riya"

    def test_bad_input_is_422(self, client):
        r = client.post("/grade", json={"question": {"id": "x"}, "answer": {}})
        assert r.status_code == 422

    def test_missing_grade_is_404(self, client):
        r = client.get("/grade/Nobody/Q99")
        assert r.status_code == 404


class TestClarifyAndAppeal:
    def test_clarify(self, client):
        client.post("/grade", json=_grade_payload())
        r = client.post("/clarify", json={"student_id": "Riya", "question_id": "Q1"})
        assert r.status_code == 200
        # mock explainer grounds its text in the facts (marks appear in it)
        assert "2.0" in r.json()["explanation"] or "full marks" in r.json()["explanation"].lower()

    def test_doubt_then_reevaluate(self, client):
        client.post("/grade", json=_grade_payload())
        d = client.post("/doubt", json={"student_id": "Riya", "question_id": "Q1"})
        assert d.status_code == 200
        r = client.post("/reevaluate", json={"student_id": "Riya", "question_id": "Q1"})
        assert r.status_code == 200
        assert "applied" in r.json()

    def test_doubt_on_missing_grade_is_404(self, client):
        r = client.post("/doubt", json={"student_id": "Ghost", "question_id": "Q1"})
        assert r.status_code == 404


class TestRealWorldFlow:
    """Teacher sets up a paper; student uploads a script; results appear."""

    def _png(self):
        import os, tempfile
        p = os.path.join(tempfile.gettempdir(), "script_test.png")
        with open(p, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        return p

    def test_setup_paper(self, client):
        r = client.post("/setup/paper", json={"questions": [
            {"id": "Q1", "type": "mcq", "text": "Capital of France?",
             "correct_answer": "Paris", "max_marks": 2},
        ]})
        assert r.status_code == 200
        assert r.json()["questions_saved"] == 1

    def test_upload_script_reads_grades_stores(self, client):
        client.post("/setup/paper", json={"questions": [
            {"id": "Q1", "type": "mcq", "text": "Capital?",
             "correct_answer": "Paris", "max_marks": 2},
        ]})
        png = self._png()
        with open(png, "rb") as f:
            r = client.post("/upload/script",
                            data={"question_id": "Q1", "student_id": "Riya"},
                            files={"file": ("script.png", f, "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["student_id"] == "Riya"
        assert "marks_awarded" in body

    def test_upload_for_missing_question_is_404(self, client):
        png = self._png()
        with open(png, "rb") as f:
            r = client.post("/upload/script",
                            data={"question_id": "GHOST", "student_id": "Riya"},
                            files={"file": ("s.png", f, "image/png")})
        assert r.status_code == 404

    def test_results_sheet(self, client):
        client.post("/setup/paper", json={"questions": [
            {"id": "Q1", "type": "mcq", "text": "Capital?",
             "correct_answer": "Paris", "max_marks": 2},
        ]})
        png = self._png()
        with open(png, "rb") as f:
            client.post("/upload/script",
                        data={"question_id": "Q1", "student_id": "Riya"},
                        files={"file": ("s.png", f, "image/png")})
        r = client.get("/results/Riya")
        assert r.status_code == 200
        assert r.json()["student_id"] == "Riya"
        assert len(r.json()["results"]) == 1


class TestDatasetCollection:
    """Uploads save the image + record a prediction; export returns the dataset."""

    def _png(self):
        import os, tempfile
        p = os.path.join(tempfile.gettempdir(), "ds_test.png")
        with open(p, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        return p

    def test_upload_saves_image_and_records_prediction(self, client, monkeypatch, tmp_path):
        import grader.config as cfg
        monkeypatch.setattr(cfg.settings, "UPLOADS_DIR", str(tmp_path / "uploads"))
        client.post("/setup/paper", json={"questions": [
            {"id": "Q1", "type": "mcq", "text": "Capital?",
             "correct_answer": "Paris", "max_marks": 2}]})
        with open(self._png(), "rb") as f:
            r = client.post("/upload/script",
                            data={"question_id": "Q1", "student_id": "Riya"},
                            files={"file": ("script.png", f, "image/png")})
        assert r.status_code == 200
        # the image was saved to disk
        import os
        assert os.path.exists(r.json()["image_saved_as"])
        # a prediction row exists
        ds = client.get("/dataset/export?format=json").json()
        assert ds["count"] >= 1

    def test_export_csv_has_header(self, client, monkeypatch, tmp_path):
        import grader.config as cfg
        monkeypatch.setattr(cfg.settings, "UPLOADS_DIR", str(tmp_path / "uploads"))
        client.post("/setup/paper", json={"questions": [
            {"id": "Q1", "type": "mcq", "text": "Capital?",
             "correct_answer": "Paris", "max_marks": 2}]})
        with open(self._png(), "rb") as f:
            client.post("/upload/script",
                        data={"question_id": "Q1", "student_id": "Riya"},
                        files={"file": ("s.png", f, "image/png")})
        r = client.get("/dataset/export?format=csv")
        assert r.status_code == 200
        assert "student_id" in r.text.splitlines()[0]   # header present


class TestMultiQuestionFlow:
    """Upload paper -> key -> full script; grade all answers by question number."""

    def _png(self):
        import os, tempfile
        p = os.path.join(tempfile.gettempdir(), "multi_test.png")
        with open(p, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 60)
        return p

    def _upload(self, client, endpoint, reader, **data):
        import grader.vision as vision
        vision.get_image_reader = lambda: reader
        with open(self._png(), "rb") as f:
            return client.post(endpoint, data=data,
                               files={"file": ("x.png", f, "image/png")})

    def test_full_multi_flow(self, client, monkeypatch, tmp_path):
        import grader.config as cfg
        from grader.vision import MockImageReader
        monkeypatch.setattr(cfg.settings, "UPLOADS_DIR", str(tmp_path / "up"))

        # 1. paper: two numeric questions
        r = self._upload(client, "/upload/paper",
                         MockImageReader(items=[("Q1", "2+2?"), ("Q2", "3+3?")]),
                         default_type="numeric", default_max_marks=5)
        assert r.status_code == 200
        assert r.json()["questions_read"] == 2

        # 2. key: Q1=4, Q2=6
        r = self._upload(client, "/upload/key",
                         MockImageReader(items=[("Q1", "4"), ("Q2", "6")]))
        assert r.status_code == 200
        assert set(r.json()["questions_updated"]) == {"Q1", "Q2"}

        # 3. student script: Q1=4 (right), Q2=5 (wrong)
        r = self._upload(client, "/upload/script-full",
                         MockImageReader(student_id="Saahil",
                                         items=[("Q1", "4"), ("Q2", "5")]))
        assert r.status_code == 200
        body = r.json()
        assert body["student_id"] == "Saahil"
        assert body["total"] == 5.0      # Q1 right (5), Q2 wrong (0)
        assert body["out_of"] == 10.0


class TestPerQuestionMarks:
    """Marks read from the page win; missing ones use the default; teacher can edit."""

    def _png(self):
        import os, tempfile
        p = os.path.join(tempfile.gettempdir(), "marks_test.png")
        with open(p, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 60)
        return p

    def _upload(self, client, reader, **data):
        import grader.vision as vision
        vision.get_image_reader = lambda: reader
        with open(self._png(), "rb") as f:
            return client.post("/upload/paper", data=data,
                               files={"file": ("x.png", f, "image/png")})

    def test_marks_read_from_page_and_default(self, client, monkeypatch, tmp_path):
        import grader.config as cfg
        from grader.vision import MockImageReader
        monkeypatch.setattr(cfg.settings, "UPLOADS_DIR", str(tmp_path / "up"))
        r = self._upload(client, MockImageReader(
            items=[("Q1", "2+2?", 2), ("Q2", "big essay", 10), ("Q3", "3+3?", None)]),
            default_type="numeric", default_max_marks=5)
        marks = {m["id"]: (m["max_marks"], m["marks_source"]) for m in r.json()["marks"]}
        assert marks["Q1"] == (2, "page")
        assert marks["Q2"] == (10, "page")
        assert marks["Q3"] == (5, "default")

    def test_teacher_can_edit_marks(self, client, monkeypatch, tmp_path):
        import grader.config as cfg
        from grader.vision import MockImageReader
        monkeypatch.setattr(cfg.settings, "UPLOADS_DIR", str(tmp_path / "up"))
        self._upload(client, MockImageReader(items=[("Q1", "2+2?", None)]),
                     default_type="numeric", default_max_marks=5)
        r = client.post("/questions/marks",
                        json={"updates": [{"question_id": "Q1", "max_marks": 8}]})
        assert r.status_code == 200
        qs = client.get("/questions").json()["questions"]
        assert qs[0]["max_marks"] == 8