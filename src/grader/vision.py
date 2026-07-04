"""Vision / OCR - reading a student's paper answer from an image (Component 3.7).

This is a NEW step that sits at the very FRONT of the pipeline, before ingest:

    image of paper  ->  vision model reads it  ->  {student_id, answer_text}
                                                          |
                    (then the normal pipeline)  ->  ingest -> grade -> ...

The vision model reads BOTH handwritten and printed text, and pulls out the
student ID and the answer from the page. We reuse the SAME OpenRouter setup as
grading (same key from .env). As always: a mock reader is the default so tests
are free and offline; the real reader turns on when a key is present.
"""

import base64
import json
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# The structured result of reading one answer image.
# ---------------------------------------------------------------------------

@dataclass
class ReadResult:
    """What the vision reader pulls off one page."""

    student_id: str        # the student ID/name read from the paper ("" if none)
    answer_text: str       # the answer text read from the paper
    confidence: float      # how sure the reader is (0..1)
    question_id: str = ""  # the question number read from the paper ("" if none)
    raw: str = ""          # the raw model reply, kept for debugging


@dataclass
class MultiReadItem:
    """One numbered item read from a page: a question, a key answer, or an answer."""
    question_id: str       # the question number, e.g. "Q1" (or "" if unreadable)
    text: str              # the text for that item (question text / answer / key)
    max_marks: float | None = None   # marks read from the page, if written (e.g. [2 marks])


@dataclass
class MultiReadResult:
    """The result of reading a WHOLE page of numbered items."""
    student_id: str            # read from the page ("" if none / not a script)
    items: list[MultiReadItem] # one per question number found
    confidence: float
    raw: str = ""


# ---------------------------------------------------------------------------
# The reader interface - a job description: "read an answer image".
# ---------------------------------------------------------------------------

class AnswerImageReader(ABC):
    """Interface: anything that can read a student answer from an image."""

    @abstractmethod
    def read(self, image_path: str) -> ReadResult:
        """Read one image and return a ReadResult (single answer)."""
        ...

    def read_multi(self, image_path: str) -> "MultiReadResult":
        """Read a whole page of numbered items (questions/answers/keys).

        Default: wrap the single read() as a one-item list. Real readers
        override this to extract every numbered item from the page.
        """
        r = self.read(image_path)
        return MultiReadResult(
            student_id=r.student_id,
            items=[MultiReadItem(question_id=r.question_id or "Q1",
                                 text=r.answer_text)],
            confidence=r.confidence, raw=r.raw,
        )


# ---------------------------------------------------------------------------
# The MOCK reader - for building and testing, no network, no key.
# ---------------------------------------------------------------------------

class MockImageReader(AnswerImageReader):
    """A stand-in reader that returns fixed text - free, offline, deterministic.

    It does not actually look at the image; it returns preset values so the
    whole pipeline can be built and tested without any API calls.
    """

    def __init__(self, student_id="STU001", answer_text="This is a mock answer.",
                 confidence=0.5, question_id="", items=None):
        self._student_id = student_id
        self._answer_text = answer_text
        self._confidence = confidence
        self._question_id = question_id
        # items: optional preset list of (question_id, text) tuples for read_multi
        self._items = items

    def read(self, image_path: str) -> ReadResult:
        # Confirm the file exists so tests catch a wrong path, but do not read it.
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return ReadResult(
            student_id=self._student_id,
            answer_text=self._answer_text,
            confidence=self._confidence,
            question_id=self._question_id,
            raw="(mock reader - no model called)",
        )

    def read_multi(self, image_path: str) -> MultiReadResult:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if self._items is not None:
            items = []
            for it in self._items:
                if len(it) == 3:
                    q, t, m = it
                    items.append(MultiReadItem(question_id=q, text=t, max_marks=m))
                else:
                    q, t = it
                    items.append(MultiReadItem(question_id=q, text=t))
        else:
            items = [MultiReadItem(question_id=self._question_id or "Q1",
                                   text=self._answer_text)]
        return MultiReadResult(student_id=self._student_id, items=items,
                               confidence=self._confidence,
                               raw="(mock reader - no model called)")


# ---------------------------------------------------------------------------
# The REAL reader - sends the image to a free vision model via OpenRouter.
# ---------------------------------------------------------------------------

class OpenRouterImageReader(AnswerImageReader):
    """Reads an answer image using a vision-capable model on OpenRouter."""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def read(self, image_path: str) -> ReadResult:
        import requests

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Turn the image file into a base64 data URI the API can accept.
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64}"

        system = (
            "You read scanned exam papers (handwritten or printed). "
            "Extract the student's ID/name, the QUESTION NUMBER the answer is "
            "for (as written on the page, e.g. Q1, 2, iii), and their answer. "
            'Respond ONLY with JSON: '
            '{"student_id": "<id or name, or empty>", '
            '"question_id": "<question number, or empty>", '
            '"answer_text": "<the full answer>", "confidence": <0..1>}.'
        )

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Read this exam page."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]},
                ],
                "temperature": 0,
            }),
            timeout=90,
        )
        response.raise_for_status()
        body = response.json()
        # OpenRouter may return an error object instead of choices (rate limit,
        # invalid key, model unavailable). Handle it gracefully, do not crash.
        if "choices" not in body or not body["choices"]:
            err = body.get("error", {})
            msg = err.get("message", "no choices returned") if isinstance(err, dict) else str(err)
            return ReadResult(student_id="", answer_text="",
                              confidence=0.0, raw=f"[reader error] {msg}")
        content = body["choices"][0]["message"]["content"]
        return self._parse(content)

    def read_multi(self, image_path: str) -> "MultiReadResult":
        """Read a whole page into a LIST of numbered items via the vision model."""
        import requests

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64}"

        system = (
            "You read scanned exam pages (handwritten or printed). The page may "
            "contain SEVERAL numbered items (questions or answers). Extract EVERY "
            "numbered item, its MARKS if written on the page (e.g. [2 marks], "
            "(5)), and, if present, the student's ID/name. "
            'Respond ONLY with JSON: {"student_id": "<id or empty>", '
            '"items": [{"question_id": "<e.g. Q1>", "text": "<the text>", '
            '"max_marks": <number or null>}, ...], "confidence": <0..1>}.'
        )
        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Read every numbered item on this page."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]},
                ],
                "temperature": 0,
            }),
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        if "choices" not in body or not body["choices"]:
            err = body.get("error", {})
            msg = err.get("message", "no choices") if isinstance(err, dict) else str(err)
            return MultiReadResult(student_id="", items=[], confidence=0.0,
                                   raw=f"[reader error] {msg}")
        content = body["choices"][0]["message"]["content"]
        return self._parse_multi(content)

    @staticmethod
    def _parse_multi(content: str) -> "MultiReadResult":
        try:
            start = content.index("{"); end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            items = []
            for it in data.get("items", []):
                mm = it.get("max_marks", None)
                try:
                    mm = float(mm) if mm is not None else None
                except (TypeError, ValueError):
                    mm = None
                items.append(MultiReadItem(
                    question_id=str(it.get("question_id", "")).strip(),
                    text=str(it.get("text", "")).strip(),
                    max_marks=mm))
            return MultiReadResult(
                student_id=str(data.get("student_id", "")).strip(),
                items=items,
                confidence=float(data.get("confidence", 0.5)),
                raw=content,
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            return MultiReadResult(student_id="", items=[], confidence=0.3, raw=content)

    @staticmethod
    def _parse(content: str) -> ReadResult:
        """Pull the JSON out of the model reply, with a safe fallback."""
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            return ReadResult(
                student_id=str(data.get("student_id", "")).strip(),
                answer_text=str(data.get("answer_text", "")).strip(),
                confidence=float(data.get("confidence", 0.5)),
                question_id=str(data.get("question_id", "")).strip(),
                raw=content,
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            # If the model did not return clean JSON, treat the whole reply as
            # the answer text, with low confidence and no ids.
            return ReadResult(student_id="", answer_text=content.strip(),
                              confidence=0.3, question_id="", raw=content)


# ---------------------------------------------------------------------------
# Factory + convenience: pick the real reader if a key is set, else the mock.
# ---------------------------------------------------------------------------

def get_image_reader() -> AnswerImageReader:
    """Return the real reader if an API key is configured, else the mock."""
    from grader.config import settings
    if settings.has_api_key():
        return OpenRouterImageReader(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    return MockImageReader()


def read_answer_from_image(
    image_path: str,
    question_id: str,
    answer_id: str,
    reader: AnswerImageReader | None = None,
) -> dict:
    """Read one image and produce a raw answer dict ready for ingestion.

    Returns a dict with the same shape ingest_answer expects:
        {id, question_id, student_id, answer_text}
    The question_id and student_id are read FROM THE PAGE when the reader finds
    them; the passed-in question_id is used only as a FALLBACK if the page has
    none. answer_id is always supplied by the caller.
    """
    if reader is None:
        reader = get_image_reader()
    result = reader.read(image_path)
    # question number read from the page wins; else fall back to the given one
    final_qid = result.question_id or question_id
    return {
        "id": answer_id,
        "question_id": final_qid,
        "student_id": result.student_id or "UNKNOWN",
        "answer_text": result.answer_text,
        "_question_id_source": "page" if result.question_id else "fallback",
    }


class VisionReadError(Exception):
    """Raised when the vision model could not read the page (with the reason)."""


def _check_read(result) -> None:
    """Raise a clear error if the read failed, carrying the model's message."""
    if not result.items and result.raw and "[reader error]" in result.raw:
        raise VisionReadError(result.raw.replace("[reader error]", "").strip())


def read_paper_from_image(image_path: str, reader: AnswerImageReader | None = None) -> list[dict]:
    """Read a QUESTION PAPER image into a list of raw question dicts (id + text).

    Type/marks/key are filled in later by the teacher or the answer-key upload.
    """
    if reader is None:
        reader = get_image_reader()
    result = reader.read_multi(image_path)
    _check_read(result)
    out = []
    for idx, item in enumerate(result.items, start=1):
        out.append({
            "id": item.question_id or f"Q{idx}",
            "text": item.text,
            "max_marks": item.max_marks,   # None if the paper did not show marks
        })
    return out


def read_key_from_image(image_path: str, reader: AnswerImageReader | None = None) -> dict:
    """Read an ANSWER KEY image into {question_id: key_text}."""
    if reader is None:
        reader = get_image_reader()
    result = reader.read_multi(image_path)
    _check_read(result)
    return {(item.question_id or f"Q{idx}"): item.text
            for idx, item in enumerate(result.items, start=1)}


def read_script_from_image(image_path: str, reader: AnswerImageReader | None = None) -> dict:
    """Read a STUDENT SCRIPT image into {student_id, answers: {question_id: text}}."""
    if reader is None:
        reader = get_image_reader()
    result = reader.read_multi(image_path)
    _check_read(result)
    answers = {(item.question_id or f"Q{idx}"): item.text
               for idx, item in enumerate(result.items, start=1)}
    return {"student_id": result.student_id or "UNKNOWN", "answers": answers}