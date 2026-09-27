"""FastAPI app: upload a document, follow its ingestion, read its Markdown, and chat with it."""
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db, embeddings, llm, modes, prejudge, reranker, retrieval
from .config import ALLOWED_EXTENSIONS, UPLOAD_DIR, settings
from .ingest import ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db.client().admin.command("ping")
    db.ensure_vector_index()
    embeddings.get_model()  # load the embedding model before accepting traffic
    if settings.reranker_enabled:
        reranker.get_model()  # and the reranker (small, CPU)
    # Preload the model the next step needs. A new session starts with an upload, so with contextual embedding
    # that is the context model; after contextualizing, ingest swaps in the answering model.
    if settings.contextual_embedding:
        llm.load_in_background(settings.context_model, settings.context_num_ctx, "at startup")
    else:
        llm.load_in_background(settings.llm_model, settings.llm_num_ctx, "at startup")
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


def _doc_info(doc: dict) -> dict:
    """What the UI shows about a document (DocInfo in frontend/lib/api.ts)."""
    mode = modes.get(doc.get("mode"))
    return {
        "doc_id": doc["_id"],
        "filename": doc["filename"],
        "mode": mode.name,
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


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_index": db.vector_index_ready,
        "llm_model": settings.llm_model,
        "llm_available": llm.model_available(),
        "embed_model": settings.embed_model,
        "context_model": settings.context_model if settings.contextual_embedding else None,
        "prejudge_model": settings.llm_model if settings.prejudge_enabled else None,
    }


@app.get("/modes")
def list_modes():
    """Private and cloud mode with their models; cloud mode lists the API keys it still needs."""
    return [mode.info() for mode in modes.MODES.values()]


@app.post("/documents", status_code=202)
async def upload_document(
    background: BackgroundTasks, file: UploadFile = File(...), mode_name: str = Form(modes.PRIVATE, alias="mode")
):
    if mode_name not in modes.MODES:
        raise HTTPException(400, f"Unknown mode {mode_name!r}.")
    mode = modes.MODES[mode_name]
    if mode.missing_keys():
        raise HTTPException(400, f"{mode.label} needs {' and '.join(mode.missing_keys())} (see README).")
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

    now = datetime.now(timezone.utc)
    doc = {
        "_id": doc_id,
        "filename": name,
        "size": len(data),
        "mode": mode.name,
        "status": "queued",
        "progress": 0,
        "contextual": settings.contextual_embedding,
        "context_model": mode.context_model if settings.contextual_embedding else None,
        "created_at": now,
        "updated_at": now,
    }
    db.documents().insert_one(doc)
    background.add_task(ingest, doc_id, path, name, mode)  # sync fn -> runs in threadpool
    return _doc_info(doc)


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = db.documents().find_one({"_id": doc_id}, {"markdown": 0})
    if not doc:
        raise HTTPException(404, "Document not found.")
    return _doc_info(doc)


@app.get("/documents/{doc_id}/markdown", response_class=PlainTextResponse)
def get_markdown(doc_id: str):
    doc = db.documents().find_one({"_id": doc_id}, {"markdown": 1})
    if not doc or not doc.get("markdown"):
        raise HTTPException(404, "Markdown not available.")
    return PlainTextResponse(doc["markdown"], media_type="text/markdown; charset=utf-8")


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    db.chunks().delete_many({"doc_id": doc_id})
    db.documents().delete_one({"_id": doc_id})
    retrieval.bm25_cache.drop(doc_id)
    for upload in UPLOAD_DIR.glob(f"{doc_id}.*"):
        upload.unlink(missing_ok=True)


@app.post("/chat")
def chat(req: ChatRequest):
    doc = db.documents().find_one({"_id": req.doc_id}, {"status": 1, "mode": 1})
    if not doc:
        raise HTTPException(404, "Document not found.")
    if doc["status"] != "ready":
        raise HTTPException(409, "Document is not indexed yet.")
    mode = modes.get(doc.get("mode"))  # a document is answered in the mode it was uploaded in
    if mode.missing_keys():
        raise HTTPException(400, f"{mode.label} needs {' and '.join(mode.missing_keys())} (see README).")

    passages = retrieval.search(req.doc_id, req.question, mode)
    messages = llm.build_messages(req.question, passages, [h.model_dump() for h in req.history])

    def events():
        # NDJSON stream: one "sources" event, many "token" events, then "done" (or "error")
        yield _ndjson({"type": "sources", "sources": [_source(n, p) for n, p in enumerate(passages, start=1)]})

        # Pre-judge: skip the answering LLM when the passages can't answer the question
        if settings.prejudge_enabled:
            try:
                can_answer = prejudge.can_answer(req.question, passages, mode)
            except Exception:  # noqa: BLE001
                log.exception("pre-judge failed; answering anyway")
                can_answer = True
            if not can_answer:
                yield _ndjson({"type": "token", "content": prejudge.NOT_ENOUGH_CONTENT})
                yield _ndjson({"type": "done"})
                return

        try:
            for token in mode.stream_answer(messages):
                yield _ndjson({"type": "token", "content": token})
            yield _ndjson({"type": "done"})
        except Exception as e:  # noqa: BLE001
            log.exception("generation failed")
            yield _ndjson({"type": "error", "message": str(e)})

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
