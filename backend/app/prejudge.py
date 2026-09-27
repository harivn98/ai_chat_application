"""Pre-judge: decide whether the retrieved passages can answer the question before calling the answering LLM.

A small model (PREJUDGE_MODEL) reads the top-k passages and the question and replies YES or NO.
On NO the answering LLM is skipped and the user gets NOT_ENOUGH_CONTENT instead of a guess.

With PREJUDGE_ON_CPU it runs entirely on the CPU (num_gpu=0), so it stays loaded in RAM next to the
answering model on the GPU and neither of them is ever swapped out.
"""
import logging
import os
import re

import httpx

from .config import settings

log = logging.getLogger("prejudge")

NOT_ENOUGH_CONTENT = "There isn't enough content in the document to answer this question."

PROMPT = """You check whether retrieved passages from a document contain the information needed to answer a question.

Passages:
{passages}

Question: {question}

Do the passages contain enough information to answer the question (fully or mostly)?
Reply with exactly one word: YES or NO."""


def _options() -> dict:
    """Model options; warm_up() must send the same ones, or Ollama reloads the model."""
    opts = {"num_ctx": settings.prejudge_num_ctx}
    if settings.prejudge_on_cpu:
        opts.update(num_gpu=0, num_thread=os.cpu_count() or 4)
    return opts


def warm_up() -> None:
    """Load the pre-judge model (on CPU if configured) and keep it loaded.

    If the same model is loaded on the GPU (e.g. it just wrote chunk contexts), Ollama moves it,
    which also frees that GPU memory for the answering model.
    """
    payload = {"model": settings.prejudge_model, "keep_alive": settings.llm_keep_alive, "options": _options()}
    timeout = httpx.Timeout(connect=10, read=settings.llm_load_timeout, write=30, pool=10)
    r = httpx.post(f"{settings.ollama_url}/api/generate", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")


def can_answer(question: str, passages: list[dict]) -> bool:
    """True if the passages can answer the question. Fails open (True) if the model gives no verdict."""
    if not passages:
        return False
    text = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))
    payload = {
        "model": settings.prejudge_model,
        "messages": [{"role": "user", "content": PROMPT.format(passages=text, question=question)}],
        "stream": False,
        "think": False,
        "keep_alive": settings.llm_keep_alive,
        "options": {**_options(), "temperature": 0, "num_predict": 3},
    }
    timeout = httpx.Timeout(connect=10, read=settings.llm_timeout, write=30, pool=10)
    r = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    reply = r.json()["message"]["content"].strip().upper()
    m = re.match(r"\W*(YES|NO)\b", reply)
    if not m:
        log.warning("Pre-judge gave no YES/NO verdict (%r); answering anyway", reply[:50])
        return True
    return m.group(1) == "YES"
