"""FastAPI app: upload a document, follow its ingestion, read its Markdown, and chat with it."""
import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db, embeddings, modes, ollama, prejudge, reranker, retrieval, rewrite
from .answer_prompt import build_messages
from .config import ALLOWED_EXTENSIONS, UPLOAD_DIR, settings
from .ingest import ingest
from .modes import Mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db.ping()
    db.ensure_indexes()
    embeddings.get_model()  # load the embedding model before accepting traffic
    if any(mode.reranker for mode in modes.MODES.values()):
        reranker.get_model()  # and the reranker (small, CPU)
    # Preload the model the next step needs. A new session starts with an upload, so with contextual embedding
    # that is the context model; after contextualizing, ingest swaps in the answering model.
    if settings.contextual_embedding:
        ollama.load_in_background(settings.context_model, settings.context_num_ctx, "at startup")
    else:
        ollama.load_in_background(settings.llm_model, settings.llm_num_ctx, "at startup")
    log.info("Backend ready (vector index: %s, llm: %s)", db.vector_index_ready, settings.llm_model)
    yield


app = FastAPI(title="RAG AI_chat_application API", lifespan=lifespan)


# ------------------------------------------------------------------ schemas
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []


def _new_document(doc_id: str, filename: str, size: int, mode: Mode) -> dict:
    """A document's record at upload, before ingestion starts."""
    now = datetime.now(timezone.utc)
    return {
        "_id": doc_id,
        "filename": filename,
        "size": size,
        "mode": mode.name,
        **({} if mode.local else {"reranker": mode.reranker, "prejudge": mode.prejudge}),
        "status": "queued",
        "progress": 0,
        "contextual": settings.contextual_embedding,
        "context_model": mode.context_model if settings.contextual_embedding else None,
        "created_at": now,
        "updated_at": now,
    }


def _doc_info(doc: dict) -> dict:
    """What the UI shows about a document (DocInfo in frontend/lib/api.ts)."""
    mode = modes.for_document(doc)
    return {
        "doc_id": doc["_id"],
        "filename": doc["filename"],
        "mode": mode.name,
        "mode_label": mode.label,
        "embed_model": mode.embed_model,
        "llm_model": mode.llm_model,
        "status": doc["status"],
        "progress": doc.get("progress", 0),
        "num_chunks": doc.get("num_chunks"),
        "contextual": doc.get("contextual", False),
        "context_model": doc.get("context_model"),
        "context_done": doc.get("context_done"),
        "error": doc.get("error"),
        "failed_stage": doc.get("failed_stage"),
    }


def _source(number: int, passage: dict) -> dict:
    """A retrieved passage as the UI shows it (Source in frontend/lib/api.ts); `number` is its citation [n]."""
    return {
        "id": number,
        "section": passage["section"],
        "text": passage["text"],
        "start": passage["start"],
        "end": passage["end"],
        "bm25_rank": passage.get("bm25_rank"),
        "vector_rank": passage.get("vector_rank"),
        "fused_rank": passage["fused_rank"] if "rerank_rank" in passage else None,  # position before reranking
    }


def _ndjson(event: dict) -> str:
    return json.dumps(event) + "\n"


def _require_keys(mode: Mode) -> None:
    if error := mode.missing_keys_error():
        raise HTTPException(400, error)


def _answer_events(question: str, search_question: str, passages: list[dict], history: list[dict],
                   mode: Mode) -> Iterator[str]:
    """The chat reply as NDJSON: one "sources" event, a "prejudge" event (if the mode pre-judges), many "token"
    events, then "done" (or "error"). `search_question` is the standalone question the passages were retrieved
    for; the answering model gets `question` as asked, with the history."""
    yield _ndjson({"type": "sources", "sources": [_source(n, p) for n, p in enumerate(passages, start=1)]})

    # Pre-judge: skip the answering LLM when the passages hold none of the answer
    verdict = prejudge.ALL
    if mode.prejudge:
        try:
            verdict = prejudge.verdict(search_question, passages, mode)
            yield _ndjson({"type": "prejudge", "verdict": verdict})
        except Exception:  # noqa: BLE001
            log.exception("pre-judge failed; answering anyway")
        if verdict == prejudge.NONE:
            yield _ndjson({"type": "token", "content": prejudge.NOT_ENOUGH_CONTENT})
            yield _ndjson({"type": "done"})
            return

    messages = build_messages(question, passages, history, partial=verdict == prejudge.PARTIAL)
    try:
        for token in mode.stream_answer(messages):
            yield _ndjson({"type": "token", "content": token})
        yield _ndjson({"type": "done"})
    except Exception as e:  # noqa: BLE001
        log.exception("generation failed")
        yield _ndjson({"type": "error", "message": str(e)})


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_index": db.vector_index_ready,
        "llm_model": settings.llm_model,
        "llm_available": ollama.model_available(),
        "embed_model": settings.embed_model,
        "context_model": settings.context_model if settings.contextual_embedding else None,
        "prejudge_model": settings.llm_model if settings.prejudge_enabled else None,  # private mode
    }


@app.get("/modes")
def list_modes():
    """Private and cloud mode; cloud mode lists the API keys it still needs."""
    return [mode.info() for mode in modes.MODES.values()]


@app.post("/documents", status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    mode_name: str = Form(modes.PRIVATE, alias="mode"),
    use_reranker: bool = Form(True, alias="reranker"),  # cloud mode only; private mode follows .env
    use_prejudge: bool = Form(True, alias="prejudge"),
):
    try:
        mode = modes.for_upload(mode_name, use_reranker, use_prejudge)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _require_keys(mode)
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF, TXT and MD files are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")

    doc_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{doc_id}{ext}"
    path.write_bytes(data)
    doc = _new_document(doc_id, name, len(data), mode)
    db.insert_document(doc)
    background.add_task(ingest, doc_id, path, name, mode)  # sync fn -> runs in threadpool
    return _doc_info(doc)


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return _doc_info(doc)


@app.get("/documents/{doc_id}/markdown", response_class=PlainTextResponse)
def get_markdown(doc_id: str):
    markdown = db.get_markdown(doc_id)
    if not markdown:
        raise HTTPException(404, "Markdown not available.")
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    db.delete_document(doc_id)
    retrieval.drop_document(doc_id)
    for upload in UPLOAD_DIR.glob(f"{doc_id}.*"):
        upload.unlink(missing_ok=True)


@app.post("/chat")
def chat(req: ChatRequest):
    doc = db.get_document(req.doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    if doc["status"] != "ready":
        raise HTTPException(409, "Document is not indexed yet.")
    mode = modes.for_document(doc)  # a document is answered in the mode (and with the switches) it was uploaded in
    _require_keys(mode)

    history = [h.model_dump() for h in req.history]
    search_question = rewrite.standalone_question(req.question, history, mode)  # follow-ups: resolve "it", "that"
    passages = retrieval.search(req.doc_id, search_question, mode)
    return StreamingResponse(
        _answer_events(req.question, search_question, passages, history, mode),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
