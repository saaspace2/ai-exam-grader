"""Serving API - the front door (Component 7).

A FastAPI web service exposing the grading system over HTTP, so teachers and
students (or a front-end, or Databricks) can use it. Each endpoint just calls
the functions we already built and tested. FastAPI validates every request
using our Pydantic models, so bad input is rejected automatically.

Run it locally:
    uvicorn grader.api:app --reload
Then open http://127.0.0.1:8000/docs for an interactive test page.
"""

import csv
import io
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path as _Path

from grader.appeals import (
    clarify_result,
    confirm_reevaluation,
    raise_doubt,
    reevaluate,
)
from grader.config import settings
from grader.grading import grade_answer
from grader.ingestion import ingest_answer, ingest_question
from grader.vision import (read_answer_from_image, get_image_reader,
                           read_paper_from_image, read_key_from_image,
                           read_script_from_image, VisionReadError)
from grader.models import GradeRecord
from grader.retrieval import clarify
from grader.store import Store

app = FastAPI(
    title="AI Exam-Grading System",
    description="Grade answers, clarify marks, and handle re-evaluation appeals.",
    version="0.1.0",
)

# Allow a browser page to call this API (fine for local/dev use).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def home():
    """Serve the frontend page at the root URL."""
    index = _Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. API is running - see /docs."}


def get_store() -> Store:
    """Open the shared database. A fresh connection per request keeps it simple."""
    return Store(settings.GRADER_DB_PATH)


# ---------------------------------------------------------------------------
# Request bodies (FastAPI validates these automatically).
# ---------------------------------------------------------------------------

class GradeRequest(BaseModel):
    question: dict      # raw question fields (validated by ingest_question)
    answer: dict        # raw answer fields (validated by ingest_answer)


class ClarifyRequest(BaseModel):
    student_id: str
    question_id: str
    student_question: str = "Why did I get this mark?"


class DoubtRequest(BaseModel):
    student_id: str
    question_id: str
    reason: str | None = None


class ReevalRequest(BaseModel):
    student_id: str
    question_id: str
    student_reason: str | None = None


class ConfirmRequest(BaseModel):
    student_id: str
    question_id: str
    new_marks: float
    justification: str = "Student confirmed."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check - the smoke-test target. Is the service alive?"""
    return {"status": "ok", "using_real_ai": settings.has_api_key()}


@app.post("/grade")
def grade(req: GradeRequest):
    """Ingest a question + answer, grade it, store it, return the grade."""
    try:
        question = ingest_question(req.question)
        answer = ingest_answer(req.answer)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input: {e}")

    store = get_store()
    try:
        store.save_question(question)
        try:
            store.save_answer(answer)
        except ValueError:
            pass  # answer may already be stored; grading can still proceed
        record = grade_answer(question, answer)
        try:
            store.save_grade(record)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return record.model_dump()
    finally:
        store.close()


@app.get("/grade/{student_id}/{question_id}")
def get_grade(student_id: str, question_id: str):
    """Fetch a stored grade."""
    store = get_store()
    try:
        record = store.get_grade(student_id, question_id)
        if record is None:
            raise HTTPException(status_code=404, detail="No grade found.")
        return record.model_dump()
    finally:
        store.close()


@app.post("/clarify")
def clarify_endpoint(req: ClarifyRequest):
    """Explain a stored mark (grounded in the real facts)."""
    store = get_store()
    try:
        return {"explanation": clarify(store, req.student_id, req.question_id,
                                       req.student_question)}
    finally:
        store.close()


@app.post("/doubt")
def doubt_endpoint(req: DoubtRequest):
    """Raise a doubt on a grade (flags it for re-evaluation)."""
    store = get_store()
    try:
        entry = raise_doubt(store, req.student_id, req.question_id, req.reason)
        return {"status": "doubt_raised", "audit_id": entry.id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        store.close()


@app.post("/reevaluate")
def reevaluate_endpoint(req: ReevalRequest):
    """Re-evaluate a disputed grade. A mark that would DROP is held for confirm."""
    store = get_store()
    try:
        result = reevaluate(store, req.student_id, req.question_id,
                            req.student_reason)
        return result.__dict__
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        store.close()


@app.post("/reevaluate/confirm")
def confirm_endpoint(req: ConfirmRequest):
    """Apply a held (lower) mark after the student confirms."""
    store = get_store()
    try:
        result = confirm_reevaluation(store, req.student_id, req.question_id,
                                      req.new_marks, req.justification)
        return result.__dict__
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        store.close()


# ===========================================================================
# REAL-WORLD FLOW: teacher sets up the paper + key; student uploads a script.
# ===========================================================================

class SetupQuestion(BaseModel):
    """One question + its answer key, as the teacher provides it."""
    id: str
    type: str
    text: str
    correct_answer: str | None = None
    rubric: str | None = None
    tolerance: float | None = 0.0
    max_marks: float


class SetupRequest(BaseModel):
    """The teacher's whole question paper + answer key."""
    questions: list[SetupQuestion]


@app.post("/setup/paper")
def setup_paper(req: SetupRequest):
    """Teacher sets up the question paper + answer key (store all questions)."""
    store = get_store()
    saved = 0
    try:
        for sq in req.questions:
            raw = sq.model_dump()
            # drop None optional fields so validation uses defaults
            raw = {k: v for k, v in raw.items() if v is not None}
            question = ingest_question(raw)
            store.save_question(question)
            saved += 1
        return {"status": "ok", "questions_saved": saved}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid question paper: {e}")
    finally:
        store.close()


@app.post("/upload/script")
async def upload_script(
    question_id: str = Form(...),
    student_id: str = Form(""),
    file: UploadFile = File(...),
):
    """Student uploads an answer-script image. The system reads it, grades it,
    stores it, and returns the grade.

    student_id is optional: if blank, the vision reader tries to read it from
    the page (real scripts have the name/ID written on top).
    """
    # SAVE the uploaded image permanently (evidence + dataset), not a temp file.
    import os
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    ext = (file.filename or "img.png").split(".")[-1]
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(settings.UPLOADS_DIR, saved_name)
    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    store = get_store()
    try:
        # STEP 0: read the saved script image -> raw answer dict.
        # The reader tries to read the QUESTION NUMBER from the page; the typed
        # question_id is only a fallback.
        answer_id = f"A_{student_id or 'UNKNOWN'}_{question_id}"
        raw = read_answer_from_image(saved_path, question_id=question_id,
                                     answer_id=answer_id)
        qid_source = raw.pop("_question_id_source", "fallback")
        if student_id:
            raw["student_id"] = student_id   # typed id overrides the read one

        # the question the answer is actually for (read from page, or fallback)
        effective_qid = raw["question_id"]
        question = store.get_question(effective_qid)
        if question is None:
            raise HTTPException(status_code=404,
                detail=(f"The answer refers to question '{effective_qid}' "
                        f"(source: {qid_source}), which is not set up."))

        # If the vision reader returned nothing (e.g. API error/rate limit),
        # do not grade empty text - report a clear message instead.
        if not raw["answer_text"].strip():
            raise HTTPException(
                status_code=502,
                detail=("Could not read the answer from the image. The vision "
                        "model may be rate-limited or unavailable. Try again in "
                        "a moment, or check your OPENROUTER_API_KEY."),
            )

        # STEP 1+2+3: ingest -> grade -> store
        answer = ingest_answer(raw)
        try:
            store.save_answer(answer)
        except ValueError:
            pass
        record = grade_answer(question, answer)
        try:
            store.save_grade(record)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        # RECORD THE PREDICTION in the dataset (image + read + grade + model).
        store.save_prediction(
            prediction_id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            student_id=raw["student_id"],
            question_id=effective_qid,
            image_path=saved_path,
            answer_read=raw["answer_text"],
            marks_awarded=record.marks_awarded,
            max_marks=record.max_marks,
            grading_method=record.grading_method,
            model_used=(settings.OPENROUTER_MODEL if settings.has_api_key() else "mock"),
        )

        return {
            "student_id": raw["student_id"],
            "question_id": effective_qid,
            "question_id_source": qid_source,
            "answer_read": raw["answer_text"],
            "marks_awarded": record.marks_awarded,
            "max_marks": record.max_marks,
            "grading_method": record.grading_method,
            "justification": record.justification,
            "image_saved_as": saved_path,
        }
    finally:
        store.close()


@app.get("/results/{student_id}")
def results(student_id: str):
    """All grades for one student (their results sheet)."""
    store = get_store()
    try:
        rows = store.conn.execute(
            "SELECT question_id, data FROM grades WHERE student_id = ? ORDER BY question_id",
            (student_id,),
        ).fetchall()
        import json as _json
        out = []
        for r in rows:
            g = _json.loads(r["data"])
            out.append({
                "question_id": r["question_id"],
                "marks_awarded": g["marks_awarded"],
                "max_marks": g["max_marks"],
                "status": g["status"],
            })
        return {"student_id": student_id, "results": out,
                "total": sum(x["marks_awarded"] for x in out),
                "out_of": sum(x["max_marks"] for x in out)}
    finally:
        store.close()


@app.get("/dataset/export")
def export_dataset(format: str = "json"):
    """Export the collected dataset (all predictions) as JSON or CSV."""
    store = get_store()
    try:
        rows = store.list_predictions()
    finally:
        store.close()

    if format == "csv":
        buf = io.StringIO()
        cols = ["id", "created_at", "student_id", "question_id", "image_path",
                "answer_read", "marks_awarded", "max_marks", "grading_method", "model_used"]
        writer = csv.DictWriter(buf, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in cols})
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=grader_dataset.csv"},
        )
    return {"count": len(rows), "predictions": rows}



# ---------------------------------------------------------------------------
# MULTI-QUESTION uploads: a single image holds several questions/answers.
# ---------------------------------------------------------------------------

def _save_upload(file: "UploadFile") -> str:
    """Save an uploaded file under UPLOADS_DIR and return its path."""
    import os
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    ext = (file.filename or "img.png").split(".")[-1]
    path = os.path.join(settings.UPLOADS_DIR, f"{uuid.uuid4().hex}.{ext}")
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return path


@app.post("/upload/paper")
async def upload_paper(
    default_type: str = Form("short"),
    default_max_marks: float = Form(5),
    file: UploadFile = File(...),
):
    """Read a QUESTION PAPER image into many questions and store them.

    The reader extracts each numbered question's text. Type and max_marks use
    the given defaults (the teacher can refine, and the answer-key upload adds
    the correct answers / rubric).
    """
    path = _save_upload(file)
    try:
        read = read_paper_from_image(path)
    except VisionReadError as e:
        raise HTTPException(status_code=502, detail=f"Vision model error: {e}")
    if not read:
        raise HTTPException(status_code=502,
            detail="Could not read any questions from the image (model may be unavailable).")

    store = get_store()
    saved = []
    try:
        for q in read:
            # marks read from the page win; else fall back to the default.
            q_marks = q.get("max_marks") or default_max_marks
            raw = {"id": q["id"], "type": default_type, "text": q["text"] or "(unreadable)",
                   "max_marks": q_marks}
            # short/essay need a rubric placeholder; mcq/numeric a correct_answer placeholder
            if default_type in ("mcq", "numeric"):
                raw["correct_answer"] = ""
            else:
                raw["rubric"] = ""
            try:
                question = ingest_question(raw)
                store.save_question(question)
                saved.append(q["id"])
            except Exception:
                pass
        # include the marks used per question, and whether they were read or default
        marks_used = []
        for q in read:
            marks_used.append({
                "id": q["id"],
                "max_marks": q.get("max_marks") or default_max_marks,
                "marks_source": "page" if q.get("max_marks") else "default",
            })
        return {"status": "ok", "questions_read": len(read),
                "questions_saved": saved, "marks": marks_used,
                "image_saved_as": path}
    finally:
        store.close()


@app.post("/upload/key")
async def upload_key(file: UploadFile = File(...)):
    """Read an ANSWER KEY image and attach each answer to its stored question."""
    path = _save_upload(file)
    try:
        key = read_key_from_image(path)
    except VisionReadError as e:
        raise HTTPException(status_code=502, detail=f"Vision model error: {e}")
    if not key:
        raise HTTPException(status_code=502,
            detail="Could not read any answers from the key image.")

    store = get_store()
    updated = []
    try:
        for qid, ans in key.items():
            question = store.get_question(qid)
            if question is None:
                continue
            # put the key into correct_answer (mcq/numeric) or rubric (short/essay)
            if question.type.value in ("mcq", "numeric"):
                question.correct_answer = ans
            else:
                question.rubric = ans
            store.save_question(question)
            updated.append(qid)
        return {"status": "ok", "answers_read": len(key), "questions_updated": updated}
    finally:
        store.close()


@app.post("/upload/script-full")
async def upload_script_full(
    student_id: str = Form(""),
    file: UploadFile = File(...),
):
    """Read a WHOLE student answer script and grade every answer by question number."""
    path = _save_upload(file)
    try:
        read = read_script_from_image(path)
    except VisionReadError as e:
        raise HTTPException(status_code=502, detail=f"Vision model error: {e}")
    sid = student_id or read["student_id"] or "UNKNOWN"
    answers = read["answers"]
    if not answers:
        raise HTTPException(status_code=502,
            detail="Could not read any answers from the script (model may be unavailable).")

    store = get_store()
    results = []
    try:
        for qid, ans_text in answers.items():
            question = store.get_question(qid)
            if question is None:
                results.append({"question_id": qid, "error": "not set up"})
                continue
            raw = {"id": f"A_{sid}_{qid}", "question_id": qid,
                   "student_id": sid, "answer_text": ans_text}
            answer = ingest_answer(raw)
            try:
                store.save_answer(answer)
            except ValueError:
                pass
            record = grade_answer(question, answer)
            try:
                store.save_grade(record)
            except ValueError:
                pass
            store.save_prediction(
                prediction_id=uuid.uuid4().hex,
                created_at=datetime.now(timezone.utc).isoformat(),
                student_id=sid, question_id=qid, image_path=path,
                answer_read=ans_text, marks_awarded=record.marks_awarded,
                max_marks=record.max_marks, grading_method=record.grading_method,
                model_used=(settings.OPENROUTER_MODEL if settings.has_api_key() else "mock"),
            )
            results.append({"question_id": qid, "answer_read": ans_text,
                            "marks_awarded": record.marks_awarded,
                            "max_marks": record.max_marks,
                            "grading_method": record.grading_method})
        total = sum(r.get("marks_awarded", 0) for r in results)
        out_of = sum(r.get("max_marks", 0) for r in results)
        return {"student_id": sid, "results": results,
                "total": total, "out_of": out_of, "image_saved_as": path}
    finally:
        store.close()


class QuestionMarks(BaseModel):
    """One question's marks (and optional type) to set/adjust."""
    question_id: str
    max_marks: float
    type: str | None = None   # optionally change the type too


class SetMarksRequest(BaseModel):
    updates: list[QuestionMarks]


@app.post("/questions/marks")
def set_marks(req: SetMarksRequest):
    """Teacher sets or adjusts marks (and optionally type) per question."""
    store = get_store()
    updated = []
    try:
        for u in req.updates:
            q = store.get_question(u.question_id)
            if q is None:
                continue
            # rebuild the question with new marks/type (re-validate via ingest)
            raw = q.model_dump()
            raw["max_marks"] = u.max_marks
            if u.type:
                raw["type"] = u.type
            raw = {k: v for k, v in raw.items() if v is not None}
            try:
                newq = ingest_question(raw)
                store.save_question(newq)
                updated.append(u.question_id)
            except Exception:
                pass
        return {"status": "ok", "updated": updated}
    finally:
        store.close()


@app.get("/questions")
def list_questions():
    """List all set-up questions (id, type, text, marks) for the teacher to review."""
    store = get_store()
    try:
        rows = store.conn.execute("SELECT data FROM questions ORDER BY id").fetchall()
        import json as _json
        out = []
        for r in rows:
            q = _json.loads(r["data"])
            out.append({"id": q["id"], "type": q["type"],
                        "text": q.get("text", ""), "max_marks": q["max_marks"],
                        "correct_answer": q.get("correct_answer"),
                        "rubric": q.get("rubric")})
        return {"questions": out}
    finally:
        store.close()


@app.get("/students")
def list_students():
    """List every student who has at least one grade (so you know valid IDs)."""
    store = get_store()
    try:
        rows = store.conn.execute(
            "SELECT DISTINCT student_id FROM grades ORDER BY student_id"
        ).fetchall()
        return {"students": [r["student_id"] for r in rows]}
    finally:
        store.close()