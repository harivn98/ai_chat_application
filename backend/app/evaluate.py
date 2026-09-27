"""QASPER evaluation of the RAG pipeline (allenai/qasper: NLP papers + questions + gold answers + evidence).

Runs only when triggered:
    python -m app.evaluate --run-id <id> [--papers 5] [--seed 0] [--split test]

Runs in four phases, one Ollama model at a time: (1) each sampled paper is converted to Markdown and
ingested like an upload (chunk + optional Contextual Retrieval contexts + embed + store), (2) every
question goes through hybrid retrieval + the pre-judge, (3) the questions the pre-judge let through are
answered, (4) the judge grades the answers. Each stage is timed; model switches are not.

Metrics:
  - Answer F1     official QASPER token F1 against the best-matching annotator answer
  - Evidence F1   official QASPER paragraph F1; predicted evidence = paper paragraphs inside the cited passages
  - Retrieval recall@k  share of gold evidence paragraphs found anywhere in the top-k retrieved chunks
  - Judge correct LLM judge: does the answer convey the same information as a reference answer?

Each run writes to <EVAL_DIR> (evaluation_metrics/):
  <run_id>.json  complete details: metrics, timings, environment variables, and every question's
                 answer, references, scores and judge output
  results.md     one row per run, for side-by-side comparison
"""
import argparse
import dataclasses
import io
import json
import logging
import random
import re
import string
import sys
import tarfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import httpx

from . import db, llm, prejudge, reranker, retrieval
from .chunker import chunk_markdown
from .config import settings
from .contextual import contextualize, hand_over, indexed_content
from .embeddings import embed_passages, get_model
from .ingest import _wait_until_searchable

log = logging.getLogger("evaluate")

S3 = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
SPLITS = {
    "test": (f"{S3}/qasper-test-and-evaluator-v0.3.tgz", "qasper-test-v0.3.json"),
    "validation": (f"{S3}/qasper-train-dev-v0.3.tgz", "qasper-dev-v0.3.json"),
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
CITE_RE = re.compile(r"\[(\d+)\]")
NOT_FOUND_RE = re.compile(
    r"could(?: not|n't) find|not (?:mentioned|found|provided|specified|stated|available|included)"
    r"|does(?: not|n't) (?:mention|say|specify|state|provide|contain|include)|no information",
    re.IGNORECASE,
)
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")

RESULTS_HEADER = (
    "# QASPER evaluation results\n\n"
    "Dataset: [allenai/qasper](https://huggingface.co/datasets/allenai/qasper). "
    "Contextualization = the context model writing a context for every chunk of one paper, "
    "embedding = embedding one paper (seconds per paper); retrieval = query embedding + BM25 + "
    "vector search + RRF, rerank = cross-encoder re-scoring of the fused candidates, pre-judge = the YES/NO check whether the passages can answer, "
    "generation = full LLM answer (answered questions only), judging = one judge call (seconds per question). "
    "Pre-judge rejected = questions answered with 'not enough content' instead of calling the LLM, and how many "
    "of those an annotator also marked unanswerable. "
    "Answer F1 and Evidence F1 are the official QASPER metrics (evidence = paragraphs in the passages the "
    "answer cites). Retrieval recall@k = share of gold evidence paragraphs present in the top-k chunks. "
    "Judge correct = the judge model says the answer matches a reference answer. All scores are 0-100.\n\n"
    "| Run ID | Date (UTC) | Split · papers · questions | LLM / judge | Retrieval config "
    "| Contextualization (s/paper) | Embedding (s/paper) "
    "| Retrieval (s/q) | Rerank (s/q) | Pre-judge (s/q) | Generation (s/q) | Judging (s/q) | Total (min) | Answer F1 "
    "| F1 extractive / abstractive / yes-no / unanswerable | Evidence F1 | Retrieval recall@k "
    "| Judge correct | Pre-judge rejected | Errors |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _upgrade_results_table(path) -> None:
    """Rewrite an older results.md for the current columns; columns an old run lacks get '–'."""
    header_line, sep_line = RESULTS_HEADER.rstrip("\n").split("\n")[-2:]
    new_cols = _cells(header_line)
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        h = next(i for i, ln in enumerate(lines) if ln.startswith("| Run ID |"))
    except StopIteration:
        return
    old_cols = _cells(lines[h])
    if old_cols == new_cols:
        return
    rows = []
    for ln in lines[h + 2:]:
        if ln.startswith("|"):
            old = dict(zip(old_cols, _cells(ln)))
            if "Retrieval config" in old and "ctx" not in old["Retrieval config"]:
                old["Retrieval config"] += " · no ctx"  # runs before contextual embedding existed
            if "Retrieval config" in old and "prejudge" not in old["Retrieval config"]:
                old["Retrieval config"] += " · no prejudge"  # runs before the pre-judge existed
            if "Retrieval config" in old and "rerank" not in old["Retrieval config"]:
                old["Retrieval config"] += " · no rerank"  # runs before the reranker existed
            rows.append("| " + " | ".join(old.get(c, "–") for c in new_cols) + " |")
    path.write_text(RESULTS_HEADER + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

JUDGE_PROMPT = """You are grading an answer to a question about a research paper.

Question:
{question}

Reference answers (written by different annotators; any one of them is acceptable):
{references}

Candidate answer:
{response}

Is the candidate answer correct, i.e. does it convey the same key information as at least one reference answer?
- Extra correct detail is fine. Missing the key information or contradicting it is not.
- If a reference says the paper does not answer the question, the candidate is correct only if it also says the answer is not in the paper.
- Ignore citation markers such as [1].

Reply with exactly one word: CORRECT or INCORRECT."""


# ------------------------------------------------------------------ official QASPER scoring (qasper_evaluator.py)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def _references(qa: dict) -> list[dict]:
    """Gold answers per annotator, as in the official evaluator with --text_evidence_only."""
    refs = []
    for annotation in qa["answers"]:
        a = annotation["answer"]
        if a["unanswerable"]:
            refs.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
            continue
        if a["extractive_spans"]:
            answer, kind = ", ".join(a["extractive_spans"]), "extractive"
        elif a["free_form_answer"]:
            answer, kind = a["free_form_answer"], "abstractive"
        elif a["yes_no"] is not None:
            answer, kind = ("Yes" if a["yes_no"] else "No"), "boolean"
        else:
            continue
        evidence = [_one_line(t) for t in a["evidence"] if "FLOAT SELECTED" not in t]
        refs.append({"answer": answer, "evidence": evidence, "type": kind})
    return refs


# ------------------------------------------------------------------ dataset
def _load_split(split: str) -> dict:
    url, member = SPLITS[split]
    cache = settings.data_dir / "eval" / member
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        _say(f"Downloading the QASPER {split} split (first run only)…")
        r = httpx.get(url, follow_redirects=True, timeout=300)
        r.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            cache.write_bytes(tar.extractfile(member).read())
    return json.loads(cache.read_text(encoding="utf-8"))


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _paper_markdown(paper: dict) -> tuple[str, list[str]]:
    """Paper -> Markdown (nested 'A ::: B' section names become nested headings) + its paragraphs."""
    abstract = _one_line(paper["abstract"])
    lines = [f"# {_one_line(paper['title'])}", "", "## Abstract", "", abstract, ""]
    paragraphs = [abstract] if abstract else []
    prev: list[str] = []
    for sec in paper["full_text"]:
        parts = [p.strip() for p in (sec["section_name"] or "").split(":::") if p.strip()]
        same = 0
        while same < min(len(parts), len(prev)) and parts[same] == prev[same]:
            same += 1
        for depth in range(same, len(parts)):
            lines += [f"{'#' * min(depth + 2, 6)} {parts[depth]}", ""]
        prev = parts
        for para in map(_one_line, sec["paragraphs"]):
            if para:
                lines += [para, ""]
                paragraphs.append(para)
    return "\n".join(lines), paragraphs


def _paragraphs_in(texts: list[str], paragraphs: list[str]) -> list[str]:
    """Paper paragraphs that (fully or partly) appear in the given chunk texts."""
    blocks = [b for t in texts for b in map(_one_line, t.split("\n\n")) if len(b) >= 40]
    return [p for p in paragraphs if any(b in p for b in blocks)]


# ------------------------------------------------------------------ judge
def _judge(prompt: str) -> str:
    payload = {
        "model": settings.judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": settings.llm_keep_alive,
        "options": {"temperature": 0, "num_ctx": settings.llm_num_ctx},
    }
    timeout = httpx.Timeout(connect=10, read=600, write=60, pool=10)
    r = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"]


def _load_model(model: str) -> None:
    """Load a model into Ollama before timing anything; on CPU this alone can take several minutes."""
    st = _Status(f"Loading {model} into Ollama (instant if already loaded; minutes if Ollama runs on CPU)…")
    llm.warm_up(model)
    st.done(f"{model} loaded in {_fmt_secs(st.elapsed)}")


def _parse_verdict(text: str) -> bool | None:
    m = re.findall(r"\b(INCORRECT|CORRECT)\b", text.upper())
    return m[-1] == "CORRECT" if m else None


# ------------------------------------------------------------------ progress output
_TTY = sys.stdout.isatty()


def _say(text: str = "") -> None:
    print(text, flush=True)


def _short(text: str, n: int = 70) -> str:
    text = _one_line(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_secs(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"


class _Status:
    """A status line that keeps updating in place (with elapsed time) until done() is called."""

    def __init__(self, label: str):
        self.label, self.detail, self.t0 = label, "", time.perf_counter()
        self._width, self._stopped, self._lock = 0, False, threading.Lock()
        if _TTY:
            threading.Thread(target=self._tick, daemon=True).start()
        else:
            _say(label)

    def _tick(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                if self._stopped:
                    return
                self._write(f"{self.label} {self.detail} [{_fmt_secs(self.elapsed)}]")

    def _write(self, text: str):
        sys.stdout.write("\r" + text.ljust(self._width))
        sys.stdout.flush()
        self._width = len(text)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def set(self, detail: str):
        self.detail = detail

    def done(self, text: str):
        with self._lock:
            self._stopped = True
            if _TTY:
                self._write(text)
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                _say(text)


# ------------------------------------------------------------------ one question, in two steps
# The steps run as separate passes over all questions (all pre-judges, then all answers). Both use the
# answering model, which is loaded once, right after the context model is unloaded.
def _retrieve_and_prejudge(doc_id: str, qa: dict, indent: str) -> dict:
    question = qa["question"].strip()
    t0 = time.perf_counter()
    if settings.reranker_enabled:
        candidates = retrieval.hybrid_search(doc_id, question, settings.rerank_candidates)
    else:
        candidates = passages = retrieval.hybrid_search(doc_id, question)
    retrieval_s = time.perf_counter() - t0
    _say(f"{indent}retrieval: {len(candidates)} chunks in {retrieval_s:.2f}s")

    rerank_s = None
    if settings.reranker_enabled:
        t0 = time.perf_counter()
        passages = reranker.rerank(question, candidates, settings.top_k)
        rerank_s = time.perf_counter() - t0
        moved = [p["fused_rank"] for p in passages]
        _say(f"{indent}rerank: kept fused #{', #'.join(map(str, moved))} in {rerank_s:.2f}s")

    can_answer, prejudge_s = True, None
    if settings.prejudge_enabled:
        st = _Status(f"{indent}pre-judge…")
        t0 = time.perf_counter()
        can_answer = prejudge.can_answer(question, passages)
        prejudge_s = time.perf_counter() - t0
        st.done(f"{indent}pre-judge: {'YES, will answer' if can_answer else 'NO, not enough content'} "
                f"({_fmt_secs(prejudge_s)})")
    return {"question": question, "passages": passages, "candidates": candidates, "retrieval_s": retrieval_s,
            "rerank_s": rerank_s, "can_answer": can_answer, "prejudge_s": prejudge_s}


def _answer_and_score(step: dict, qa: dict, paragraphs: list[str], indent: str) -> dict:
    question, passages, can_answer = step["question"], step["passages"], step["can_answer"]
    retrieval_s, prejudge_s = step["retrieval_s"], step["prejudge_s"]
    refs = _references(qa)

    generation_s = None
    if can_answer:
        st = _Status(f"{indent}generating…")
        st.set("waiting for first token")
        pieces: list[str] = []
        t0 = time.perf_counter()
        for piece in llm.stream_chat(llm.build_messages(question, passages, [])):
            pieces.append(piece)
            st.set(f"{len(pieces)} tokens")
        generation_s = time.perf_counter() - t0
        response = "".join(pieces)
        st.done(f"{indent}generation: {len(pieces)} tokens in {_fmt_secs(generation_s)}")
        _say(f"{indent}answer: {_short(response, 90)}")
    else:
        response = prejudge.NOT_ENOUGH_CONTENT
        _say(f"{indent}skipped: pre-judge said not enough content")

    cited = sorted({int(n) for n in CITE_RE.findall(response) if 1 <= int(n) <= len(passages)})
    if not can_answer or (NOT_FOUND_RE.search(response) and not cited):
        predicted_answer, predicted_evidence = "Unanswerable", []
    else:
        predicted_answer = CITE_RE.sub("", response)
        predicted_evidence = _paragraphs_in([passages[n - 1]["text"] for n in cited], paragraphs)

    f1, answer_type = max(((token_f1_score(predicted_answer, r["answer"]), r["type"]) for r in refs),
                          key=lambda x: x[0])
    evidence_f1 = max(paragraph_f1_score(predicted_evidence, r["evidence"]) for r in refs)

    retrieved_paras = set(_paragraphs_in([p["text"] for p in passages], paragraphs))
    recalls = [len(retrieved_paras & set(r["evidence"])) / len(r["evidence"]) for r in refs if r["evidence"]]
    recall = f"{100 * max(recalls):.0f}%" if recalls else "n/a"
    pool_recall = None
    if settings.reranker_enabled:
        pool = set(_paragraphs_in([c["text"] for c in step["candidates"]], paragraphs))
        pool_recalls = [len(pool & set(r["evidence"])) / len(r["evidence"]) for r in refs if r["evidence"]]
        pool_recall = max(pool_recalls) if pool_recalls else None
        if pool_recall is not None:
            recall += f" (in the {len(step['candidates'])} candidates: {100 * pool_recall:.0f}%)"
    _say(f"{indent}scores: retrieval recall {recall} · answer F1 {100 * f1:.0f} · evidence F1 {100 * evidence_f1:.0f}")

    return {
        "question_id": qa["question_id"],
        "question": question,
        "references": refs,
        "response": response,
        "predicted_answer": predicted_answer,
        "retrieved_chunks": [p["index"] for p in passages],
        "fused_ranks": [p.get("fused_rank") for p in passages],  # where the kept chunks were before reranking
        "cited_passages": cited,
        "answer_f1": f1,
        "answer_type": answer_type,
        "evidence_f1": evidence_f1,
        "retrieval_recall": max(recalls) if recalls else None,  # None: no text evidence (e.g. unanswerable)
        "candidate_recall": pool_recall,  # same, over the RERANK_CANDIDATES chunks the reranker chose from
        "prejudge_can_answer": can_answer if settings.prejudge_enabled else None,
        "gold_unanswerable": any(r["type"] == "none" for r in refs),  # some annotator says the paper can't answer
        "retrieval_s": retrieval_s,
        "rerank_s": step["rerank_s"],
        "prejudge_s": prejudge_s,
        "generation_s": generation_s,  # None when the pre-judge skipped generation
    }


def _cleanup(doc_id: str):
    db.chunks().delete_many({"doc_id": doc_id})
    retrieval.bm25_cache.drop(doc_id)


# ------------------------------------------------------------------ results
def _existing_run_ids(path) -> set[str]:
    if not path.exists():
        return set()
    return {m.group(1) for m in re.finditer(r"^\| (\S+) \|", path.read_text(encoding="utf-8"), re.MULTILINE)}


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _secs(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


def _environment() -> dict[str, str]:
    """Every backend setting under its environment variable name (credentials masked)."""
    env = {}
    for f in dataclasses.fields(settings):
        value = str(getattr(settings, f.name))
        if f.name == "mongo_uri":
            value = re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", value)
        env[f.name.upper()] = value
    return env


def run(run_id: str, num_papers: int, seed: int, split: str) -> None:
    results_md = settings.eval_dir / "results.md"
    if run_id in _existing_run_ids(results_md) or (settings.eval_dir / f"{run_id}.json").exists():
        raise SystemExit(f"Run ID '{run_id}' already exists in {settings.eval_dir}; choose another.")

    data = _load_split(split)
    paper_ids = sorted(random.Random(seed).sample(sorted(data), min(num_papers, len(data))))
    total_q = sum(len(data[p]["qas"]) for p in paper_ids)
    _say(f"\n=== Run {run_id}: {len(paper_ids)} papers, {total_q} questions (QASPER {split}, seed {seed}) ===")
    _say(f"LLM {settings.llm_model} · judge {settings.judge_model} · top {settings.top_k} · "
         f"BM25 {settings.bm25_candidates} · vec ≥ {settings.vector_min_score} · "
         f"{'context ' + settings.context_model if settings.contextual_embedding else 'no context'} · "
         f"{'rerank top ' + str(settings.rerank_candidates) + ' with ' + settings.reranker_model if settings.reranker_enabled else 'no rerank'}")

    st = _Status("Connecting to MongoDB and loading the embedding model…")
    retrieval.vector_index_ready = db.ensure_vector_index()
    get_model()
    if settings.reranker_enabled:
        reranker.get_model()  # load it now so the first question's rerank time is not a model load
    st.done(f"Ready (vector index: {'yes' if retrieval.vector_index_ready else 'no, using local fallback'})")
    started = time.perf_counter()
    rows: list[dict] = []
    embedding_times: list[float] = []
    context_times: list[float] = []
    rag_times: list[float] = []
    ingested: dict[str, list[str]] = {}  # paper id -> its paragraphs, for papers that ingested fine
    ingest_errors: dict[str, str] = {}

    # The phases run model by model (context model, then answering model, then judge), so Ollama
    # doesn't swap models in and out of GPU memory for every paper.
    try:
        # Phase 1: ingest every paper (optional contexts, embeddings, vector index)
        _say(f"\n--- Phase 1/4: ingesting papers "
             f"({'with contexts from ' + settings.context_model if settings.contextual_embedding else 'no contexts'}) ---")
        if settings.contextual_embedding:
            _load_model(settings.context_model)
        for n, pid in enumerate(paper_ids, start=1):
            paper = data[pid]
            doc_id = f"eval-{run_id}-{pid}"
            markdown, paragraphs = _paper_markdown(paper)
            _say(f"\n[paper {n}/{len(paper_ids)}] {_short(paper['title'])} ({len(paper['qas'])} questions)")
            try:
                _cleanup(doc_id)
                chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
                contexts = [""] * len(chunks)
                if settings.contextual_embedding:
                    st = _Status("  adding context…")
                    t0 = time.perf_counter()
                    contexts = contextualize(markdown, chunks,
                                             lambda done, total: st.set(f"{done}/{total} chunks"))
                    context_times.append(time.perf_counter() - t0)
                    st.done(f"  context: {len(chunks)} chunks in {_fmt_secs(context_times[-1])} "
                            f"(e.g. \"{_short(contexts[len(contexts) // 2], 80)}\")")
                contents = [indexed_content(c, ctx) for c, ctx in zip(chunks, contexts)]

                st = _Status("  embedding…")
                t0 = time.perf_counter()
                vectors = embed_passages(contents)
                embedding_times.append(time.perf_counter() - t0)
                st.done(f"  embedding: {len(chunks)} chunks in {_fmt_secs(embedding_times[-1])}")
                db.chunks().insert_many(
                    [
                        {"doc_id": doc_id, "index": c.index, "section": c.section, "text": c.text,
                         "context": ctx, "content": content, "embedding": v.tolist()}
                        for c, ctx, content, v in zip(chunks, contexts, contents, vectors)
                    ]
                )
                st = _Status("  waiting for the vector index…")
                _wait_until_searchable(doc_id, len(chunks))  # index sync is excluded from the timings
                st.done(f"  vector index ready in {_fmt_secs(st.elapsed)}")
                ingested[pid] = paragraphs
            except Exception as e:  # noqa: BLE001
                log.exception("Ingesting paper %s failed", pid)
                _say(f"  FAILED to ingest paper: {e}")
                ingest_errors[pid] = str(e)

        questions = [(pid, qa) for pid in paper_ids for qa in data[pid]["qas"]]
        steps: dict[str, dict] = {}  # question id -> retrieval + pre-judge result

        # Phase 2: retrieve and pre-judge every question. The answering model does the pre-judge, so it is
        # loaded once here, straight after the context model is unloaded, and stays for phase 3.
        _say("\n--- Phase 2/4: retrieval + pre-judge "
             f"({'with ' + settings.llm_model if settings.prejudge_enabled else 'pre-judge off'}) ---")
        if settings.contextual_embedding:
            st = _Status(f"Unloading {settings.context_model}…")
            st.done(f"After contextualizing: {hand_over() or 'nothing to hand over'}")
        _load_model(settings.llm_model)
        for n, (pid, qa) in enumerate(questions, start=1):
            if pid in ingest_errors:
                continue
            _say(f"  [Q {n}/{total_q}] {_short(qa['question'])}")
            try:
                steps[qa["question_id"]] = _retrieve_and_prejudge(f"eval-{run_id}-{pid}", qa, indent="      ")
            except Exception as e:  # noqa: BLE001
                log.exception("Question %s failed", qa["question_id"])
                _say(f"      FAILED: {e}")
                steps[qa["question_id"]] = {"error": f"rag: {e}"}

        # Phase 3: answer the questions the pre-judge let through (same model, already loaded)
        to_answer = sum(1 for s in steps.values() if s.get("can_answer"))
        _say(f"\n--- Phase 3/4: answering {to_answer} of {total_q} questions with {settings.llm_model} ---")
        for n, (pid, qa) in enumerate(questions, start=1):
            base = {"paper_id": pid, "question_id": qa["question_id"], "question": qa["question"]}
            if pid in ingest_errors:
                rows.append({**base, "error": f"ingest: {ingest_errors[pid]}"})
                continue
            step = steps[qa["question_id"]]
            if "error" in step:
                rows.append({**base, "error": step["error"]})
                continue
            eta = ""
            if rag_times and step["can_answer"]:
                left = sum(1 for p, q in questions[n - 1:] if steps.get(q["question_id"], {}).get("can_answer"))
                eta = f" · ~{_fmt_secs(sum(rag_times) / len(rag_times) * left)} left in phase 3"
            _say(f"  [Q {n}/{total_q}{eta}] {_short(qa['question'])}")
            t0 = time.perf_counter()
            try:
                row = _answer_and_score(step, qa, ingested[pid], indent="      ")
            except Exception as e:  # noqa: BLE001
                log.exception("Question %s failed", qa["question_id"])
                _say(f"      FAILED: {e}")
                row = {"question_id": qa["question_id"], "question": qa["question"], "error": f"rag: {e}"}
            if step.get("can_answer"):
                rag_times.append(time.perf_counter() - t0)
            rows.append({"paper_id": pid, **row})
    finally:
        for pid in paper_ids:
            _cleanup(f"eval-{run_id}-{pid}")

    # Phase 4: LLM judge
    _say(f"\n--- Phase 4/4: judging answers with {settings.judge_model} ---")
    if settings.judge_model != settings.llm_model:
        _load_model(settings.judge_model)
    for n, row in enumerate(rows, start=1):
        if "error" in row:
            _say(f"  [judge {n}/{len(rows)}] skipped (question failed)")
            continue
        refs = "\n".join(
            "- The paper does not answer this question." if r["type"] == "none" else f"- {r['answer']}"
            for r in row["references"]
        )
        st = _Status(f"  [judge {n}/{len(rows)}] grading…")
        try:
            t0 = time.perf_counter()
            out = _judge(JUDGE_PROMPT.format(question=row["question"], references=refs, response=row["response"]))
            row["judging_s"] = time.perf_counter() - t0
            row["judge_correct"], row["judge_output"] = _parse_verdict(out), out
            verdict = {True: "CORRECT", False: "INCORRECT", None: "no verdict"}[row["judge_correct"]]
            st.done(f"  [judge {n}/{len(rows)}] {verdict:<9} ({_fmt_secs(row['judging_s'])}) {_short(row['question'], 60)}")
        except Exception as e:  # noqa: BLE001
            log.exception("Judging failed for %s", row["question_id"])
            row["judge_error"] = str(e)
            st.done(f"  [judge {n}/{len(rows)}] FAILED: {e}")

    total_min = (time.perf_counter() - started) / 60
    ok = [r for r in rows if "error" not in r]
    summary = {
        "run_id": run_id,
        "date_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}",
        "split": split,
        "seed": seed,
        "papers": len(paper_ids),
        "questions": len(rows),
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "top_k": settings.top_k,
        "bm25_candidates": settings.bm25_candidates,
        "vector_min_score": settings.vector_min_score,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "contextual_embedding": settings.contextual_embedding,
        "context_model": settings.context_model if settings.contextual_embedding else None,
        "prejudge_enabled": settings.prejudge_enabled,
        "prejudge_model": settings.llm_model if settings.prejudge_enabled else None,  # the answering model
        "reranker_enabled": settings.reranker_enabled,
        "reranker_model": settings.reranker_model if settings.reranker_enabled else None,
        "rerank_candidates": settings.rerank_candidates if settings.reranker_enabled else None,
        "contextualization_s_per_paper": _mean(context_times),
        "embedding_s_per_paper": _mean(embedding_times),
        "retrieval_s_per_question": _mean([r["retrieval_s"] for r in ok]),
        "rerank_s_per_question": _mean([r.get("rerank_s") for r in ok]),
        # recall over the chunks the reranker chose from: the most reranking can put into the top k
        "candidate_recall": _mean([r.get("candidate_recall") for r in ok]),
        "prejudge_s_per_question": _mean([r.get("prejudge_s") for r in ok]),
        "generation_s_per_question": _mean([r["generation_s"] for r in ok]),  # answered questions only
        # rejected by the pre-judge, and how many of those some annotator also marked unanswerable
        "prejudge_rejected": sum(1 for r in ok if r.get("prejudge_can_answer") is False),
        "prejudge_rejected_unanswerable": sum(1 for r in ok if r.get("prejudge_can_answer") is False
                                              and r.get("gold_unanswerable")),
        "judging_s_per_question": _mean([r.get("judging_s") for r in ok]),
        "total_minutes": total_min,
        "answer_f1": _mean([r["answer_f1"] for r in ok]),
        "answer_f1_by_type": {t: _mean([r["answer_f1"] for r in ok if r["answer_type"] == t]) for t in ANSWER_TYPES},
        "evidence_f1": _mean([r["evidence_f1"] for r in ok]),
        "retrieval_recall": _mean([r["retrieval_recall"] for r in ok]),
        "judge_correct": _mean([None if r.get("judge_correct") is None else float(r["judge_correct"]) for r in ok]),
        "errors": len(rows) - len(ok),
        "judge_errors": sum(1 for r in ok if r.get("judge_correct") is None),
    }

    env = _environment()
    summary["environment"] = env
    s = summary
    by_type = " / ".join(_pct(s["answer_f1_by_type"][t]) for t in ANSWER_TYPES)
    ctx = f"ctx {s['context_model']}" if s["contextual_embedding"] else "no ctx"
    pj = f"prejudge {s['prejudge_model']}/gpu" if s["prejudge_enabled"] else "no prejudge"
    rr = (f"rerank {s['reranker_model'].split('/')[-1]} top {s['rerank_candidates']}"
          if s["reranker_enabled"] else "no rerank")
    rejected = (f"{s['prejudge_rejected']}/{len(ok)} ({s['prejudge_rejected_unanswerable']} gold unanswerable)"
                if s["prejudge_enabled"] else "–")
    line = (
        f"| {run_id} | {s['date_utc']} | {split} · {s['papers']} · {s['questions']} (seed {seed}) "
        f"| {s['llm_model']} / {s['judge_model']} "
        f"| top {s['top_k']} · BM25 {s['bm25_candidates']} · vec ≥ {s['vector_min_score']} "
        f"· chunk {s['chunk_size']}/{s['chunk_overlap']} · {ctx} · {pj} · {rr} "
        f"| {_secs(s['contextualization_s_per_paper'])} "
        f"| {_secs(s['embedding_s_per_paper'])} | {_secs(s['retrieval_s_per_question'])} "
        f"| {_secs(s['rerank_s_per_question'])} "
        f"| {_secs(s['prejudge_s_per_question'])} "
        f"| {_secs(s['generation_s_per_question'])} | {_secs(s['judging_s_per_question'])} | {total_min:.1f} "
        f"| **{_pct(s['answer_f1'])}** | {by_type} | {_pct(s['evidence_f1'])} | {_pct(s['retrieval_recall'])} "
        f"| {_pct(s['judge_correct'])} | {rejected} | {s['errors']} failed · {s['judge_errors']} unjudged |\n"
    )

    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    if results_md.exists():
        _upgrade_results_table(results_md)
    else:
        results_md.write_text(RESULTS_HEADER, encoding="utf-8")
    with results_md.open("a", encoding="utf-8") as f:
        f.write(line)

    details = settings.eval_dir / f"{run_id}.json"
    details.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    _say(f"\n=== Done in {total_min:.1f} min: {len(ok)}/{len(rows)} questions answered ===")
    for name, key in [(f"retrieval recall@{settings.top_k}", "retrieval_recall"), ("evidence F1", "evidence_f1"),
                      ("answer F1", "answer_f1"), ("judge correct", "judge_correct")]:
        _say(f"  {name:<21}{_pct(summary[key])}")
    if settings.reranker_enabled:
        _say(f"  {'candidate recall@' + str(settings.rerank_candidates):<21}{_pct(summary['candidate_recall'])}"
             f"  (evidence among the chunks the reranker chose from)")
    if settings.prejudge_enabled:
        _say(f"  {'pre-judge rejected':<21}{rejected}")
    _say(f"\nSaved {details} (complete details)\n      {results_md} (side-by-side comparison)")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on QASPER.")
    p.add_argument("--run-id", required=True, help="unique label for this run (letters, digits, . _ -)")
    p.add_argument("--papers", type=int, default=5, help="number of papers to sample (~3.5 questions each)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed; keep it fixed to compare runs")
    p.add_argument("--split", choices=sorted(SPLITS), default="test")
    args = p.parse_args()
    if not RUN_ID_RE.match(args.run_id):
        p.error("--run-id may only contain letters, digits, '.', '_' and '-' (max 64 chars)")
    if args.papers < 1:
        p.error("--papers must be at least 1")
    run(args.run_id, args.papers, args.seed, args.split)


if __name__ == "__main__":
    main()
