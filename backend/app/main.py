import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db, llm, prejudge, retrieval
from .config import ALLOWED_EXTENSIONS, MARKDOWN_DIR, UPLOAD_DIR, settings
from .embeddings import get_model
from .ingest import ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    db.client().admin.command("ping")
    retrieval.vector_index_ready = db.ensure_vector_index()
    get_model()  # load the embedding model before accepting traffic
    # Load the LLM in the background so the first chat doesn't time out while Ollama loads it (slow on CPU)
    threading.Thread(target=_warm_up_llm, daemon=True).start()
    log.info("Backend ready (vector index: %s, llm: %s)", retrieval.vector_index_ready, settings.llm_model)
    yield


def _warm_up_llm():
    try:
        started = time.perf_counter()
        llm.warm_up()
        log.info("LLM %s loaded in %.0fs", settings.llm_model, time.perf_counter() - started)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not preload %s: %s", settings.llm_model, e)
    if settings.prejudge_enabled:
        try:
            started = time.perf_counter()
            prejudge.warm_up()
            log.info("Pre-judge %s loaded (%s) in %.0fs", settings.prejudge_model,
                     "CPU" if settings.prejudge_on_cpu else "GPU", time.perf_counter() - started)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not preload pre-judge %s: %s", settings.prejudge_model, e)


app = FastAPI(title="RAG AI_chat_application API", lifespan=lifespan)


# ------------------------------------------------------------------ schemas
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []


def _public(doc: dict) -> dict:
    return {
        "doc_id": doc["_id"],
        "filename": doc["filename"],
        "status": doc["status"],
        "progress": doc.get("progress", 0),
        "num_chunks": doc.get("num_chunks"),
        "contextual": doc.get("contextual", False),
        "context_model": doc.get("context_model"),
        "context_done": doc.get("context_done"),
        "error": doc.get("error"),
        "failed_stage": doc.get("failed_stage"),
    }


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_index": retrieval.vector_index_ready,
        "llm_model": settings.llm_model,
        "llm_available": llm.model_available(),
        "embed_model": settings.embed_model,
        "context_model": settings.context_model if settings.contextual_embedding else None,
        "prejudge_model": settings.prejudge_model if settings.prejudge_enabled else None,
    }


@app.post("/documents", status_code=202)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...)):
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
        "status": "queued",
        "progress": 0,
        "contextual": settings.contextual_embedding,
        "context_model": settings.context_model if settings.contextual_embedding else None,
        "created_at": now,
        "updated_at": now,
    }
    db.documents().insert_one(doc)
    background.add_task(ingest, doc_id, path, name)  # sync fn -> runs in threadpool
    return _public(doc)


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = db.documents().find_one({"_id": doc_id}, {"markdown": 0})
    if not doc:
        raise HTTPException(404, "Document not found.")
    return _public(doc)


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
    for p in list(UPLOAD_DIR.glob(f"{doc_id}.*")) + [MARKDOWN_DIR / f"{doc_id}.md"]:
        p.unlink(missing_ok=True)


@app.post("/chat")
def chat(req: ChatRequest):
    doc = db.documents().find_one({"_id": req.doc_id}, {"status": 1})
    if not doc:
        raise HTTPException(404, "Document not found.")
    if doc["status"] != "ready":
        raise HTTPException(409, "Document is not indexed yet.")

    passages = retrieval.hybrid_search(req.doc_id, req.question)
    messages = llm.build_messages(req.question, passages, [h.model_dump() for h in req.history])

    def events():
        # NDJSON stream: one "sources" event, many "token" events, then "done" (or "error")
        sources = [
            {
                "id": i,
                "section": p["section"],
                "text": p["text"],
                "bm25_rank": p.get("bm25_rank"),
                "vector_rank": p.get("vector_rank"),
                "rrf": round(p["rrf"], 5),
            }
            for i, p in enumerate(passages, start=1)
        ]
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        # Pre-judge: skip the answering LLM when the passages can't answer the question
        if settings.prejudge_enabled:
            try:
                ok = prejudge.can_answer(req.question, passages)
            except Exception:  # noqa: BLE001
                log.exception("pre-judge failed; answering anyway")
                ok = True
            yield json.dumps({"type": "prejudge", "can_answer": ok}) + "\n"
            if not ok:
                yield json.dumps({"type": "token", "content": prejudge.NOT_ENOUGH_CONTENT}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
                return

        try:
            for token in llm.stream_chat(messages):
                yield json.dumps({"type": "token", "content": token}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:  # noqa: BLE001
            log.exception("generation failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
