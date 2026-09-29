"""OpenRouter client for cloud mode: every cloud model (Gemini embeddings and contexts, DeepSeek answers) runs
through it.

Everything passed to these functions leaves this machine: it goes to OpenRouter and on to the model providers.
"""
import json
import logging
import time
from collections.abc import Iterator

import httpx
import numpy as np

from .config import settings

log = logging.getLogger("openrouter")

EMBED_BATCH = 100      # texts per embeddings request
RETRIES = 5            # on rate limits (429) and overload (5xx), waiting 2, 4, 8, 16, 32 s
TIMEOUT = httpx.Timeout(connect=10, read=120, write=60, pool=10)
# For models whose reasoning can't be switched off (Gemini 3.5 Flash-Lite): the lowest effort, left out of the reply
MINIMAL_REASONING = {"effort": "minimal", "exclude": True}


class OpenRouterError(RuntimeError):
    pass


def missing_keys() -> list[str]:
    """API keys cloud mode still needs."""
    return [] if settings.openrouter_api_key else ["OPENROUTER_API_KEY"]


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.openrouter_api_key}", "X-Title": "AI Chat Application (ACAP)"}


def _post(path: str, payload: dict) -> dict:
    for attempt in range(RETRIES + 1):
        r = httpx.post(f"{settings.openrouter_url}{path}", json=payload, headers=_headers(), timeout=TIMEOUT)
        if r.status_code == 200:
            reply = r.json()
            if "error" in reply:
                raise OpenRouterError(f"OpenRouter error: {reply['error']}")
            return reply
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == RETRIES:
            raise OpenRouterError(f"OpenRouter error {r.status_code} for {payload['model']}: {r.text[:300]}")
        wait = 2 ** (attempt + 1)
        log.warning("OpenRouter answered %s for %s; retrying in %ss", r.status_code, payload["model"], wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def embed(texts: list[str], kind: str) -> np.ndarray:
    """L2-normalised embeddings of CLOUD_EMBED_DIM values: kind "document" for indexed chunks, "query" for questions.

    Gemini Embedding 2 takes the retrieval task as a text prefix. OpenRouter returns its full 3072 values; the model
    is trained so that a prefix of the vector is an embedding too, so it is cut to CLOUD_EMBED_DIM and renormalised
    (what Gemini's own output_dimensionality does).
    """
    prefix = "task: search result | query: " if kind == "query" else "title: none | text: "
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        reply = _post("/embeddings", {"model": settings.cloud_embed_model,
                                      "input": [prefix + t for t in texts[i:i + EMBED_BATCH]]})
        vectors.extend(e["embedding"] for e in sorted(reply["data"], key=lambda e: e["index"]))
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.shape[1] < settings.cloud_embed_dim:
        raise OpenRouterError(f"{settings.cloud_embed_model} returned {vecs.shape[1]} values, "
                         f"fewer than CLOUD_EMBED_DIM={settings.cloud_embed_dim}")
    vecs = vecs[:, :settings.cloud_embed_dim]
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _chat_payload(model: str, messages: list[dict], stream: bool, temperature: float,
                  reasoning: dict | None = None, max_tokens: int | None = None) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        # by default reasoning follows LLM_THINK; it would never be shown anyway
        "reasoning": reasoning or {"enabled": settings.llm_think},
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return payload


def complete(model: str, prompt: str, max_tokens: int, reasoning: dict | None = None) -> str:
    """One prompt, one deterministic (temperature 0) reply. max_tokens includes any reasoning tokens."""
    reply = _post("/chat/completions", _chat_payload(model, [{"role": "user", "content": prompt}], stream=False,
                                                    temperature=0, reasoning=reasoning, max_tokens=max_tokens))
    text = reply["choices"][0]["message"].get("content") or ""
    if not text:
        raise OpenRouterError(f"{model} returned no text (finish reason: {reply['choices'][0].get('finish_reason')})")
    return text


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """The cloud answering model's reply to `messages`, streamed piece by piece (server-sent events)."""
    payload = _chat_payload(settings.cloud_llm_model, messages, stream=True, temperature=settings.llm_temperature)
    with httpx.stream("POST", f"{settings.openrouter_url}/chat/completions", json=payload, headers=_headers(),
                      timeout=TIMEOUT) as r:
        if r.status_code != 200:
            raise OpenRouterError(f"OpenRouter error {r.status_code}: {r.read().decode(errors='ignore')[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue  # blank lines and ": OPENROUTER PROCESSING" keep-alive comments
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise OpenRouterError(f"OpenRouter error: {event['error']}")
            choices = event.get("choices") or []
            piece = (choices[0].get("delta") or {}).get("content") if choices else None
            if piece:
                yield piece
