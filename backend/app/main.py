"""FastAPI app: upload a document, follow its ingestion, read its Markdown, and chat with it."""
import json
import logging
import uuid
from collections.abc import Iterator
from typing import Literal
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


app = FastAPI(title="AI Chat Application (ACAP) API", lifespan=lifespan)


# ------------------------------------------------------------------ schemas
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = []


class StoredMessage(BaseModel):
    """A question or an answer as a chat keeps it (the UI shows it again when the chat is reopened)."""
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100_000)
    sources: list[dict] | None = None
    verdict: str | None = None
    error: str | None = None


class NewMessages(BaseModel):
    messages: list[StoredMessage] = Field(min_length=1, max_length=2)  # a question and its answer


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
        "created_at": doc.get("created_at"),
    }


TITLE_CHARS = 60


def _chat_title(question: str) -> str:
    """A chat is named after its first question."""
    title = " ".join(question.split())
    return title if len(title) <= TITLE_CHARS else title[:TITLE_CHARS - 1].rstrip() + "…"


def _chat_info(chat: dict) -> dict:
    """What the chat list shows about a chat (ChatInfo in frontend/lib/api.ts)."""
    return {
        "chat_id": chat["_id"],
        "doc_id": chat["doc_id"],
        "title": chat["title"],
        "created_at": chat["created_at"],
        "updated_at": chat["updated_at"],
        "message_count": chat["message_count"] if "message_count" in chat else len(chat.get("messages", [])),
    }


def _require_chat(chat_id: str) -> dict:
    chat = db.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return chat


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


def _answer_events(doc_id: str, question: str, history: list[dict], mode: Mode) -> Iterator[str]:
    """The chat reply as NDJSON: one "sources" event, a "prejudge" event (if the mode pre-judges), many "token"
    events, then "done" (or "error").

    Private mode first loads the answering model if it isn't in memory (the UI shows the load through
    GET /answering-model). That happens inside the stream, so the response starts right away however long the
    load takes. Retrieval and the pre-judge use the standalone version of a follow-up; the answering model gets
    `question` as asked, with the history."""
    try:
        if mode.local:
            ollama.ensure_loaded(settings.llm_model, settings.llm_num_ctx)
        search_question = rewrite.standalone_question(question, history, mode)  # follow-ups: resolve "it", "that"
        passages = retrieval.search(doc_id, search_question, mode)
    except Exception as e:  # noqa: BLE001
        log.exception("could not prepare the answer")
        yield _ndjson({"type": "error", "message": str(e)})
        return
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


@app.get("/answering-model")
def answering_model():
    """Private mode's answering model: loaded, loading (seconds so far, and how long its last load took) or
    not_loaded. The UI polls it to show the load."""
    return ollama.load_status(settings.llm_model)


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


@app.get("/documents")
def list_documents():
    """Every uploaded document, newest first, with its number of chats."""
    return [{**_doc_info(d), "chat_count": d["chat_count"]} for d in db.list_documents()]


@app.delete("/documents", status_code=204)
def delete_all_documents():
    """New session: every document, with its chunks, chats and uploaded file. Uploads still being ingested stop."""
    for doc_id in db.delete_all_documents():
        for upload in UPLOAD_DIR.glob(f"{doc_id}.*"):
            upload.unlink(missing_ok=True)
    retrieval.drop_all_documents()


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


@app.get("/documents/{doc_id}/chats")
def list_chats(doc_id: str):
    """The document's chats, most recently used first, and how many a document can have."""
    if not db.get_document(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"chats": [_chat_info(c) for c in db.list_chats(doc_id)], "max_chats": settings.max_chats_per_document}


@app.post("/documents/{doc_id}/chats")
def new_chat(doc_id: str):
    """A new, empty chat about the document. If the document already has an empty chat, that one is returned
    instead, so there is never more than one empty chat."""
    if not db.get_document(doc_id):
        raise HTTPException(404, "Document not found.")
    chats = db.list_chats(doc_id)
    if empty := next((c for c in chats if not c["message_count"]), None):
        return _chat_info(empty)
    if len(chats) >= settings.max_chats_per_document:
        raise HTTPException(409, f"A document can have at most {settings.max_chats_per_document} chats. "
                                 "Delete one to start a new chat.")
    now = datetime.now(timezone.utc)
    chat = {"_id": uuid.uuid4().hex, "doc_id": doc_id, "title": "", "messages": [], "created_at": now, "updated_at": now}
    db.insert_chat(chat)
    return _chat_info(chat)


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = _require_chat(chat_id)
    return {**_chat_info(chat), "messages": chat["messages"]}


@app.post("/chats/{chat_id}/messages")
def add_messages(chat_id: str, req: NewMessages):
    """Save a question and its answer (once the answer has ended) at the end of the chat."""
    chat = _require_chat(chat_id)
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    fields = {"updated_at": datetime.now(timezone.utc)}
    if not chat["title"] and messages[0]["role"] == "user":
        fields["title"] = _chat_title(messages[0]["content"])
    db.append_messages(chat_id, messages, **fields)
    return _chat_info(db.get_chat_summary(chat_id))


@app.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: str):
    db.delete_chat(chat_id)


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
    return StreamingResponse(
        _answer_events(req.doc_id, req.question, history, mode),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
