"""Pre-judge: decide whether the retrieved passages can answer the question before generating an answer.

The answering model (LLM_MODEL) reads the top-k passages and the question and replies YES or NO.
On NO, answer generation is skipped and the user gets NOT_ENOUGH_CONTENT instead of a guess.

It uses the same model and the same num_ctx as answer generation, so the model that is already on the
GPU does both steps and Ollama never has to swap or reload a model between them.
"""
import logging
import re

import httpx

from .config import settings

log = logging.getLogger("prejudge")

NOT_ENOUGH_CONTENT = "There isn't enough content in the document to answer this question."

# Strict: the passages must contain the specific information asked for, not just be on the same topic,
# so on-topic questions the document doesn't actually answer are caught. Yes/no questions and answers that
# follow directly from stated facts still count, so answerable questions aren't blocked for their wording.
PROMPT = """You check whether passages retrieved from a document contain the answer to a question.

Passages:
{passages}

Question: {question}

Reply YES if the passages contain the specific information the question asks for, either stated directly or following clearly from what they state (for a yes/no question, facts that settle the yes or no count).
Reply NO if the passages are only about the same topic but do not contain that specific information, or if answering would require guessing or knowledge from outside the passages.
Reply with exactly one word: YES or NO."""


def can_answer(question: str, passages: list[dict]) -> bool:
    """True if the passages can answer the question. Fails open (True) if the model gives no verdict."""
    if not passages:
        return False
    text = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": PROMPT.format(passages=text, question=question)}],
        "stream": False,
        "think": False,
        "keep_alive": settings.llm_keep_alive,
        # num_ctx must match answer generation (llm.stream_chat), or Ollama reloads the model
        "options": {"num_ctx": settings.llm_num_ctx, "temperature": 0, "num_predict": 3},
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
