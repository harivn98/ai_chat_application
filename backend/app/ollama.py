"""Ollama client for private mode: model loading, one-shot completions and streamed answers."""
import json
import logging
import threading
import time
from collections.abc import Iterator

import httpx

from .config import settings

log = logging.getLogger("ollama")


class OllamaError(RuntimeError):
    """Ollama answered with an error status (404: the model is not installed)."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"Ollama error {status}: {detail[:300]}")
        self.status = status


def _timeout(read: float) -> httpx.Timeout:
    return httpx.Timeout(connect=10, read=read, write=60, pool=10)


def _post(path: str, payload: dict, read_timeout: float) -> dict:
    r = httpx.post(f"{settings.ollama_url}{path}", json=payload, timeout=_timeout(read_timeout))
    if r.status_code != 200:
        raise OllamaError(r.status_code, r.text)
    return r.json()


# ------------------------------------------------------------------ model loading
# Loads in progress, so the UI can show them (private mode): model -> [start time, callers waiting on the load].
# Ollama doesn't report how far a load is, so the UI compares the time so far with the model's last load.
_loading: dict[str, list] = {}
_loading_lock = threading.Lock()
LOAD_TIMES_FILE = settings.data_dir / "model_load_times.json"  # seconds each model's last load took


def _load_times() -> dict[str, float]:
    try:
        return json.loads(LOAD_TIMES_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _record_load_time(model: str, seconds: float) -> None:
    times = _load_times()
    times[model] = round(seconds, 1)
    try:
        LOAD_TIMES_FILE.write_text(json.dumps(times))
    except OSError as e:
        log.warning("Could not save the load time of %s: %s", model, e)


def load(model: str, num_ctx: int) -> None:
    """Load a model into Ollama's memory without generating anything.

    num_ctx must be the one the model's requests use, or Ollama loads it again on the first request.
    On CPU, loading an 8B model can take longer than a request timeout; if the client gives up, Ollama
    aborts the half-finished load and the next request starts over. Loading it once up front, with a
    generous timeout, avoids that loop.
    """
    payload = {"model": model, "keep_alive": settings.llm_keep_alive, "options": {"num_ctx": num_ctx}}
    with _loading_lock:
        entry = _loading.setdefault(model, [time.time(), 0])
        entry[1] += 1
        first = entry[1] == 1  # the caller that started the load times it; the others joined it midway
    try:
        _post("/api/generate", payload, settings.llm_load_timeout)
    finally:
        with _loading_lock:
            entry[1] -= 1
            if not entry[1]:
                del _loading[model]
    took = time.time() - entry[0]
    if first and took >= 1:  # quicker: the model was already loaded
        _record_load_time(model, took)


def _is_running(model: str) -> bool:
    """Whether Ollama has the model in memory."""
    r = httpx.get(f"{settings.ollama_url}/api/ps", timeout=3)
    r.raise_for_status()
    names = {m.get("name") for m in r.json().get("models", [])}
    return model in names or f"{model}:latest" in names


def ensure_loaded(model: str, num_ctx: int) -> None:
    """Load the model if it isn't in memory, or wait for its load in progress. Loading it here, rather than
    letting the first request load it, is what lets load_status() show the load."""
    with _loading_lock:
        loading = model in _loading
    try:
        running = not loading and _is_running(model)
    except Exception:  # noqa: BLE001
        running = False
    if not running:
        load(model, num_ctx)


def load_status(model: str) -> dict:
    """loaded, loading or not_loaded; while loading, the seconds so far and the seconds the last load took
    (expected_s, None if it was never measured)."""
    with _loading_lock:
        started = _loading[model][0] if model in _loading else None
    if started is not None:
        state = "loading"
    else:
        try:
            state = "loaded" if _is_running(model) else "not_loaded"
        except Exception:  # noqa: BLE001
            state = "not_loaded"
    return {
        "model": model,
        "state": state,
        "elapsed_s": round(time.time() - started, 1) if started is not None else None,
        "expected_s": _load_times().get(model),
    }


def load_in_background(model: str, num_ctx: int, reason: str) -> None:
    """Start loading a model without waiting for it; logs how long it took."""

    def run():
        started = time.perf_counter()
        try:
            load(model, num_ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not preload %s: %s", model, e)
            return
        log.info("%s loaded (%s) in %.0fs", model, reason, time.perf_counter() - started)

    threading.Thread(target=run, daemon=True).start()


def unload(model: str) -> None:
    """Free a model's GPU/RAM memory right away (keep_alive=0) instead of waiting for LLM_KEEP_ALIVE."""
    _post("/api/generate", {"model": model, "keep_alive": 0}, read_timeout=60)


def model_available() -> bool:
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3)
        names = {m.get("name") for m in r.json().get("models", [])}
        return settings.llm_model in names or f"{settings.llm_model}:latest" in names
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ generation
def complete(model: str, prompt: str, num_ctx: int, read_timeout: float, num_predict: int | None = None) -> str:
    """One prompt, one deterministic reply (temperature 0, no thinking)."""
    options = {"temperature": 0, "num_ctx": num_ctx}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": settings.llm_keep_alive,
        "options": options,
    }
    return _post("/api/chat", payload, read_timeout)["message"]["content"]


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """The answering model's reply to `messages`, streamed piece by piece.

    With LLM_THINK=true, Ollama sends the reasoning in a separate `thinking` field, so it is never shown.
    """
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": True,
        "think": settings.llm_think,
        "keep_alive": settings.llm_keep_alive,
        "options": {"temperature": settings.llm_temperature, "num_ctx": settings.llm_num_ctx},
    }
    first = True
    with httpx.stream("POST", f"{settings.ollama_url}/api/chat", json=payload,
                      timeout=_timeout(settings.llm_timeout)) as r:
        if r.status_code != 200:
            raise OllamaError(r.status_code, r.read().decode(errors="ignore"))
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise RuntimeError(f"Ollama error: {data['error']}")
            piece = data.get("message", {}).get("content", "")
            if first:
                piece = piece.lstrip()  # no leading blank lines in the answer
                first = not piece
            if piece:
                yield piece
            if data.get("done"):
                break
