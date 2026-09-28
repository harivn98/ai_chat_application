"""QASPER evaluation of the RAG pipeline (allenai/qasper: NLP papers + questions + gold answers + evidence).

Runs only when triggered:
    python -m app.evaluate --run-id <id> [--papers 5] [--seed 0] [--split test]
                           [--mode private|cloud-rerank|cloud-prejudge]

Runs in four phases, one Ollama model at a time: (1) each sampled paper is converted to Markdown and
ingested like an upload (chunk + optional Contextual Retrieval contexts + embed + store), (2) every
question goes through hybrid retrieval + the pre-judge, (3) the questions the pre-judge let through are
answered, (4) the judge grades the answers. Each stage is timed; model switches are not.
The cloud modes run steps 1-3 with the cloud models (see modes.py); the judge is always JUDGE_MODEL in Ollama.

Metrics:
  - Answer F1     official QASPER token F1 against the best-matching annotator answer
  - Evidence F1   official QASPER paragraph F1; predicted evidence = paper paragraphs inside the cited passages
  - Retrieval recall@k  share of gold evidence paragraphs found anywhere in the top-k retrieved chunks
  - Judge correct LLM judge: does the answer convey the same information as a reference answer?

Each run writes to <EVAL_DIR> (evaluation_metrics/):
  <run_id>.json  complete details: metrics, timings, environment variables, and every question's
                 answer, references, scores and judge output
  result_40.md   one row per run, for side-by-side comparison
"""
import argparse
import dataclasses
import json
import logging
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import db, embeddings, llm, prejudge, reranker, retrieval
from .chunker import chunk_markdown
from .config import settings
from .contextual import contextualize, hand_over, indexed_content
from .ingest import store_chunks
from .modes import MODES, PRIVATE, Mode
from .qasper import (
    ANSWER_TYPES,
    SPLITS,
    load_split,
    one_line,
    paper_markdown,
    paragraph_f1_score,
    paragraphs_in,
    references,
    token_f1_score,
)

log = logging.getLogger("evaluate")

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
CITE_RE = re.compile(r"\[(\d+)\]")
NOT_FOUND_RE = re.compile(
    r"could(?: not|n't) find|not (?:mentioned|found|provided|specified|stated|available|included)"
    r"|does(?: not|n't) (?:mention|say|specify|state|provide|contain|include)|no information",
    re.IGNORECASE,
)

VERDICT_TEXT = {prejudge.ALL: "ALL, will answer", prejudge.PARTIAL: "PARTIAL, will answer and say what is missing",
                prejudge.NONE: "NONE, not enough content"}

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
    "Judge correct = the judge model says the answer matches a reference answer. All scores are 0-100. "
    "BM25 chunks = chunks BM25 returned per question, vector chunks = chunks vector search returned at or above "
    "the minimum score per question (both as average (fewest–most)); vector score range = lowest–highest cosine "
    "similarity of those vector chunks over the whole run, and in brackets the average of each question's "
    "lowest and highest.\n\n"
    "| Run ID | Date (UTC) | Split · papers · questions | LLM / judge | Retrieval config "
    "| BM25 chunks (per q) | Vector chunks ≥ min score (per q) | Vector score range "
    "| Contextualization (s/paper) | Embedding (s/paper) "
    "| Retrieval (s/q) | Rerank (s/q) | Pre-judge (s/q) | Generation (s/q) | Judging (s/q) | Total (min) | Answer F1 "
    "| F1 extractive / abstractive / yes-no / unanswerable | Evidence F1 | Retrieval recall@k "
    "| Judge correct | Pre-judge rejected | Errors |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


# ------------------------------------------------------------------ console output
_TTY = sys.stdout.isatty()


def _say(text: str = "") -> None:
    print(text, flush=True)


def _short(text: str, n: int = 70) -> str:
    text = one_line(text)
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


# ------------------------------------------------------------------ models
def _load_model(model: str, num_ctx: int) -> None:
    """Load a model into Ollama before timing anything; on CPU this alone can take several minutes.

    num_ctx must be the one the model's requests use, or Ollama loads it again on the first (timed) request.
    """
    st = _Status(f"Loading {model} into Ollama (instant if already loaded; minutes if Ollama runs on CPU)…")
    llm.load(model, num_ctx)
    st.done(f"{model} loaded in {_fmt_secs(st.elapsed)}")


def _judge(prompt: str) -> str:
    return llm.complete(settings.judge_model, prompt, settings.llm_num_ctx, read_timeout=settings.llm_timeout)


def _parse_verdict(text: str) -> bool | None:
    m = re.findall(r"\b(INCORRECT|CORRECT)\b", text.upper())
    return m[-1] == "CORRECT" if m else None


def _cleanup(doc_id: str):
    db.chunks().delete_many({"doc_id": doc_id})
    retrieval.bm25_cache.drop(doc_id)


# ------------------------------------------------------------------ phase 1: ingest the papers
@dataclasses.dataclass
class _Ingested:
    paragraphs: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # paper id -> its paragraphs
    errors: dict[str, str] = dataclasses.field(default_factory=dict)  # paper id -> why it failed to ingest
    context_times: list[float] = dataclasses.field(default_factory=list)  # seconds per paper
    embedding_times: list[float] = dataclasses.field(default_factory=list)  # seconds per paper


def _ingest_papers(data: dict, paper_ids: list[str], doc_ids: dict[str, str], mode: Mode) -> _Ingested:
    _say(f"\n--- Phase 1/4: ingesting papers "
         f"({'with contexts from ' + mode.context_model if settings.contextual_embedding else 'no contexts'}) ---")
    if settings.contextual_embedding and mode.local:
        _load_model(settings.context_model, settings.context_num_ctx)
    ingested = _Ingested()
    for n, pid in enumerate(paper_ids, start=1):
        paper = data[pid]
        markdown, paragraphs = paper_markdown(paper)
        _say(f"\n[paper {n}/{len(paper_ids)}] {_short(paper['title'])} ({len(paper['qas'])} questions)")
        try:
            _ingest_paper(doc_ids[pid], markdown, ingested, mode)
            ingested.paragraphs[pid] = paragraphs
        except Exception as e:  # noqa: BLE001
            log.exception("Ingesting paper %s failed", pid)
            _say(f"  FAILED to ingest paper: {e}")
            ingested.errors[pid] = str(e)
    return ingested


def _ingest_paper(doc_id: str, markdown: str, ingested: _Ingested, mode: Mode) -> None:
    """Chunk, add contexts, embed and store one paper like an upload, timing the context and embedding steps."""
    chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
    contexts = [""] * len(chunks)
    if settings.contextual_embedding:
        st = _Status("  adding context…")
        t0 = time.perf_counter()
        contexts = contextualize(markdown, chunks, mode, lambda done, total: st.set(f"{done}/{total} chunks"))
        ingested.context_times.append(time.perf_counter() - t0)
        st.done(f"  context: {len(chunks)} chunks in {_fmt_secs(ingested.context_times[-1])} "
                f"(e.g. \"{_short(contexts[len(contexts) // 2], 80)}\")")
    contents = [indexed_content(c, ctx) for c, ctx in zip(chunks, contexts)]

    st = _Status("  embedding…")
    t0 = time.perf_counter()
    vectors = mode.embed_passages(contents)
    ingested.embedding_times.append(time.perf_counter() - t0)
    st.done(f"  embedding: {len(chunks)} chunks in {_fmt_secs(ingested.embedding_times[-1])}")
    store_chunks(doc_id, chunks, contexts, contents, vectors, mode)

    st = _Status("  waiting for the vector index…")
    db.wait_until_searchable(doc_id, len(chunks), mode)  # index sync is excluded from the timings
    st.done(f"  vector index ready in {_fmt_secs(st.elapsed)}")


# ------------------------------------------------------------------ phase 2: retrieve + pre-judge
def _retrieve_all(questions: list[tuple[str, dict]], doc_ids: dict[str, str], failed: dict[str, str],
                  mode: Mode) -> dict:
    """Question id -> retrieval + pre-judge result. The answering model does the pre-judge; in private mode it is
    loaded once here, straight after the context model is unloaded, and stays loaded for phase 3."""
    _say("\n--- Phase 2/4: retrieval + pre-judge "
         f"({'with ' + mode.prejudge_model if mode.prejudge else 'pre-judge off'}) ---")
    if mode.local:
        if settings.contextual_embedding:
            st = _Status(f"Unloading {settings.context_model}…")
            st.done(f"After contextualizing: {hand_over() or 'nothing to hand over'}")
        _load_model(settings.llm_model, settings.llm_num_ctx)
    steps: dict[str, dict] = {}
    for n, (pid, qa) in enumerate(questions, start=1):
        if pid in failed:
            continue
        _say(f"  [Q {n}/{len(questions)}] {_short(qa['question'])}")
        try:
            steps[qa["question_id"]] = _retrieve_and_prejudge(doc_ids[pid], qa, mode, indent="      ")
        except Exception as e:  # noqa: BLE001
            log.exception("Question %s failed", qa["question_id"])
            _say(f"      FAILED: {e}")
            steps[qa["question_id"]] = {"error": f"rag: {e}"}
    return steps


def _retrieve_and_prejudge(doc_id: str, qa: dict, mode: Mode, indent: str) -> dict:
    question = qa["question"].strip()
    t0 = time.perf_counter()
    rankings = retrieval.candidate_rankings(doc_id, question, mode)
    if mode.reranker:
        candidates = retrieval.hybrid_search(doc_id, question, mode, mode.rerank_candidates, rankings)
    else:
        candidates = passages = retrieval.hybrid_search(doc_id, question, mode, rankings=rankings)
    retrieval_s = time.perf_counter() - t0
    vector_scores = [score for _, score in rankings["vector"]]
    hits = {
        "bm25_hits": len(rankings["bm25"]),  # chunks BM25 returned (at most BM25_CANDIDATES)
        "vector_hits": len(vector_scores),  # vector hits at or above the minimum score (at most VECTOR_CANDIDATES)
        "vector_score_min": min(vector_scores, default=None),  # cosine range of those hits
        "vector_score_max": max(vector_scores, default=None),
    }
    score_range = f" {hits['vector_score_min']:.3f}–{hits['vector_score_max']:.3f}" if vector_scores else ""
    _say(f"{indent}retrieval: {len(candidates)} chunks in {retrieval_s:.2f}s "
         f"(BM25 {hits['bm25_hits']} · vector {hits['vector_hits']} ≥ {mode.vector_min_score}{score_range})")

    rerank_s = None
    if mode.reranker:
        t0 = time.perf_counter()
        passages = reranker.rerank(question, candidates, mode.top_k)
        rerank_s = time.perf_counter() - t0
        moved = [p["fused_rank"] for p in passages]
        _say(f"{indent}rerank: kept fused #{', #'.join(map(str, moved))} in {rerank_s:.2f}s")

    verdict, prejudge_s = prejudge.ALL, None
    if mode.prejudge:
        st = _Status(f"{indent}pre-judge…")
        t0 = time.perf_counter()
        verdict = prejudge.verdict(question, passages, mode)
        prejudge_s = time.perf_counter() - t0
        st.done(f"{indent}pre-judge: {VERDICT_TEXT[verdict]} ({_fmt_secs(prejudge_s)})")
    return {"question": question, "passages": passages, "candidates": candidates, "retrieval_s": retrieval_s,
            "rerank_s": rerank_s, "verdict": verdict, "can_answer": verdict != prejudge.NONE,
            "prejudge_s": prejudge_s, **hits}


# ------------------------------------------------------------------ phase 3: answer + score
def _answer_all(questions: list[tuple[str, dict]], steps: dict, ingested: _Ingested, mode: Mode) -> list[dict]:
    """One result row per question: the answer and its scores, or the error that stopped it."""
    to_answer = sum(1 for s in steps.values() if s.get("can_answer"))
    _say(f"\n--- Phase 3/4: answering {to_answer} of {len(questions)} questions with {mode.llm_model} ---")
    rows: list[dict] = []
    answer_times: list[float] = []  # for the time-left estimate
    for n, (pid, qa) in enumerate(questions, start=1):
        base = {"paper_id": pid, "question_id": qa["question_id"], "question": qa["question"]}
        if pid in ingested.errors:
            rows.append({**base, "error": f"ingest: {ingested.errors[pid]}"})
            continue
        step = steps[qa["question_id"]]
        if "error" in step:
            rows.append({**base, "error": step["error"]})
            continue
        eta = ""
        if answer_times and step["can_answer"]:
            left = sum(1 for _, q in questions[n - 1:] if steps.get(q["question_id"], {}).get("can_answer"))
            eta = f" · ~{_fmt_secs(sum(answer_times) / len(answer_times) * left)} left in phase 3"
        _say(f"  [Q {n}/{len(questions)}{eta}] {_short(qa['question'])}")
        t0 = time.perf_counter()
        try:
            row = _answer_and_score(step, qa, ingested.paragraphs[pid], mode, indent="      ")
        except Exception as e:  # noqa: BLE001
            log.exception("Question %s failed", qa["question_id"])
            _say(f"      FAILED: {e}")
            row = {"question_id": qa["question_id"], "question": qa["question"], "error": f"rag: {e}"}
        if step.get("can_answer"):
            answer_times.append(time.perf_counter() - t0)
        rows.append({"paper_id": pid, **row})
    return rows


def _answer_and_score(step: dict, qa: dict, paragraphs: list[str], mode: Mode, indent: str) -> dict:
    question, passages, can_answer = step["question"], step["passages"], step["can_answer"]
    retrieval_s, prejudge_s = step["retrieval_s"], step["prejudge_s"]
    refs = references(qa)

    generation_s = None
    if can_answer:
        st = _Status(f"{indent}generating…")
        st.set("waiting for first token")
        pieces: list[str] = []
        t0 = time.perf_counter()
        messages = llm.build_messages(question, passages, [], partial=step["verdict"] == prejudge.PARTIAL)
        for piece in mode.stream_answer(messages):
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
        predicted_evidence = paragraphs_in([passages[n - 1]["text"] for n in cited], paragraphs)

    f1, answer_type = max(((token_f1_score(predicted_answer, r["answer"]), r["type"]) for r in refs),
                          key=lambda x: x[0])
    evidence_f1 = max(paragraph_f1_score(predicted_evidence, r["evidence"]) for r in refs)

    retrieved_paras = set(paragraphs_in([p["text"] for p in passages], paragraphs))
    recalls = [len(retrieved_paras & set(r["evidence"])) / len(r["evidence"]) for r in refs if r["evidence"]]
    recall = f"{100 * max(recalls):.0f}%" if recalls else "n/a"
    pool_recall = None
    if mode.reranker:
        pool = set(paragraphs_in([c["text"] for c in step["candidates"]], paragraphs))
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
        "bm25_hits": step["bm25_hits"],
        "vector_hits": step["vector_hits"],
        "vector_score_min": step["vector_score_min"],
        "vector_score_max": step["vector_score_max"],
        "retrieved_chunks": [p["index"] for p in passages],
        "fused_ranks": [p.get("fused_rank") for p in passages],  # where the kept chunks were before reranking
        "cited_passages": cited,
        "answer_f1": f1,
        "answer_type": answer_type,
        "evidence_f1": evidence_f1,
        "retrieval_recall": max(recalls) if recalls else None,  # None: no text evidence (e.g. unanswerable)
        "candidate_recall": pool_recall,  # same, over the RERANK_CANDIDATES chunks the reranker chose from
        "prejudge_can_answer": can_answer if mode.prejudge else None,
        "prejudge_verdict": step["verdict"] if mode.prejudge else None,  # all / partial / none
        "gold_unanswerable": any(r["type"] == "none" for r in refs),  # some annotator says the paper can't answer
        "retrieval_s": retrieval_s,
        "rerank_s": step["rerank_s"],
        "prejudge_s": prejudge_s,
        "generation_s": generation_s,  # None when the pre-judge skipped generation
    }


# ------------------------------------------------------------------ phase 4: judge
def _judge_all(rows: list[dict], mode: Mode) -> None:
    """Adds the judge's verdict (judge_correct, judge_output, judging_s) to every answered row."""
    _say(f"\n--- Phase 4/4: judging answers with {settings.judge_model} ---")
    if not mode.local or settings.judge_model != settings.llm_model:  # else it is loaded since phase 2
        _load_model(settings.judge_model, settings.llm_num_ctx)
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


# ------------------------------------------------------------------ results
def _existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {m.group(1) for m in re.finditer(r"^\| (\S+) \|", path.read_text(encoding="utf-8"), re.MULTILINE)}


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _spread(vals: list) -> dict | None:
    vals = [v for v in vals if v is not None]
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)} if vals else None


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
        if f.name.endswith("_api_key"):
            value = "***" if value else ""
        env[f.name.upper()] = value
    return env


def _summary(run_id: str, split: str, seed: int, papers: int, rows: list[dict], ingested: _Ingested,
             total_min: float, mode: Mode) -> dict:
    ok = [r for r in rows if "error" not in r]
    return {
        "run_id": run_id,
        "date_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}",
        "split": split,
        "seed": seed,
        "papers": papers,
        "questions": len(rows),
        "mode": mode.name,
        "embed_model": mode.embed_model,
        "llm_model": mode.llm_model,
        "judge_model": settings.judge_model,
        "top_k": mode.top_k,
        "bm25_candidates": mode.bm25_candidates,
        "vector_candidates": mode.vector_candidates,
        "vector_min_score": mode.vector_min_score,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "contextual_embedding": settings.contextual_embedding,
        "context_model": mode.context_model if settings.contextual_embedding else None,
        "prejudge_enabled": mode.prejudge,
        "prejudge_model": mode.prejudge_model if mode.prejudge else None,
        "reranker_enabled": mode.reranker,
        "reranker_model": settings.reranker_model if mode.reranker else None,
        "rerank_candidates": mode.rerank_candidates if mode.reranker else None,
        "contextualization_s_per_paper": _mean(ingested.context_times),
        "embedding_s_per_paper": _mean(ingested.embedding_times),
        # per question: chunks BM25 returned, vector hits at or above vector_min_score, and their cosine range
        "bm25_hits": _spread([r["bm25_hits"] for r in ok]),
        "vector_hits": _spread([r["vector_hits"] for r in ok]),
        "vector_score_min": _spread([r["vector_score_min"] for r in ok]),
        "vector_score_max": _spread([r["vector_score_max"] for r in ok]),
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
        "prejudge_partial": sum(1 for r in ok if r.get("prejudge_verdict") == prejudge.PARTIAL),
        "judging_s_per_question": _mean([r.get("judging_s") for r in ok]),
        "total_minutes": total_min,
        "answer_f1": _mean([r["answer_f1"] for r in ok]),
        "answer_f1_by_type": {t: _mean([r["answer_f1"] for r in ok if r["answer_type"] == t]) for t in ANSWER_TYPES},
        "evidence_f1": _mean([r["evidence_f1"] for r in ok]),
        "retrieval_recall": _mean([r["retrieval_recall"] for r in ok]),
        "judge_correct": _mean([None if r.get("judge_correct") is None else float(r["judge_correct"]) for r in ok]),
        "errors": len(rows) - len(ok),
        "judge_errors": sum(1 for r in ok if r.get("judge_correct") is None),
        "environment": _environment(),
    }


def _prejudge_rejected(s: dict) -> str:
    answered = s["questions"] - s["errors"]
    if not s["prejudge_enabled"]:
        return "–"
    partial = f" · {s['prejudge_partial']} partial" if s.get("prejudge_partial") else ""
    return f"{s['prejudge_rejected']}/{answered} ({s['prejudge_rejected_unanswerable']} gold unanswerable){partial}"


def _hit_counts(spread: dict | None) -> str:
    return "–" if spread is None else f"{spread['mean']:.1f} ({spread['min']}–{spread['max']})"


def _score_range(s: dict) -> str:
    lo, hi = s["vector_score_min"], s["vector_score_max"]
    if lo is None or hi is None:
        return "–"
    return f"{lo['min']:.3f}–{hi['max']:.3f} (avg {lo['mean']:.3f}–{hi['mean']:.3f})"


def _results_row(s: dict) -> str:
    """The run's row in result_40.md (columns as in RESULTS_HEADER)."""
    by_type = " / ".join(_pct(s["answer_f1_by_type"][t]) for t in ANSWER_TYPES)
    local = s["mode"] == PRIVATE
    cloud = "" if local else f"cloud · emb {s['embed_model']} · vec top {s['vector_candidates']} · "
    ctx = f"ctx {s['context_model']}" if s["contextual_embedding"] else "no ctx"
    pj = f"prejudge {s['prejudge_model']}{'/gpu' if local else ''}" if s["prejudge_enabled"] else "no prejudge"
    rr = (f"rerank {s['reranker_model'].split('/')[-1]} top {s['rerank_candidates']}"
          if s["reranker_enabled"] else "no rerank")
    return (
        f"| {s['run_id']} | {s['date_utc']} | {s['split']} · {s['papers']} · {s['questions']} (seed {s['seed']}) "
        f"| {s['llm_model']} / {s['judge_model']} "
        f"| {cloud}top {s['top_k']} · BM25 {s['bm25_candidates']} · vec ≥ {s['vector_min_score']} "
        f"· chunk {s['chunk_size']}/{s['chunk_overlap']} · {ctx} · {pj} · {rr} "
        f"| {_hit_counts(s['bm25_hits'])} | {_hit_counts(s['vector_hits'])} | {_score_range(s)} "
        f"| {_secs(s['contextualization_s_per_paper'])} "
        f"| {_secs(s['embedding_s_per_paper'])} | {_secs(s['retrieval_s_per_question'])} "
        f"| {_secs(s['rerank_s_per_question'])} "
        f"| {_secs(s['prejudge_s_per_question'])} "
        f"| {_secs(s['generation_s_per_question'])} | {_secs(s['judging_s_per_question'])} "
        f"| {s['total_minutes']:.1f} "
        f"| **{_pct(s['answer_f1'])}** | {by_type} | {_pct(s['evidence_f1'])} | {_pct(s['retrieval_recall'])} "
        f"| {_pct(s['judge_correct'])} | {_prejudge_rejected(s)} "
        f"| {s['errors']} failed · {s['judge_errors']} unjudged |\n"
    )


def _save(summary: dict, rows: list[dict], results_md: Path, details_json: Path) -> None:
    results_md.parent.mkdir(parents=True, exist_ok=True)
    if not results_md.exists():
        results_md.write_text(RESULTS_HEADER, encoding="utf-8")
    with results_md.open("a", encoding="utf-8") as f:
        f.write(_results_row(summary))
    details_json.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False),
                            encoding="utf-8")


def _print_summary(s: dict, results_md: Path, details_json: Path) -> None:
    _say(f"\n=== Done in {s['total_minutes']:.1f} min: {s['questions'] - s['errors']}/{s['questions']} "
         f"questions answered ===")
    for name, key in [(f"retrieval recall@{s['top_k']}", "retrieval_recall"), ("evidence F1", "evidence_f1"),
                      ("answer F1", "answer_f1"), ("judge correct", "judge_correct")]:
        _say(f"  {name:<21}{_pct(s[key])}")
    _say(f"  {'BM25 chunks/q':<21}{_hit_counts(s['bm25_hits'])}")
    _say(f"  {'vector chunks/q':<21}{_hit_counts(s['vector_hits'])}  (at or above {s['vector_min_score']})")
    _say(f"  {'vector score range':<21}{_score_range(s)}")
    if s["reranker_enabled"]:
        _say(f"  {'candidate recall@' + str(s['rerank_candidates']):<21}{_pct(s['candidate_recall'])}"
             f"  (evidence among the chunks the reranker chose from)")
    if s["prejudge_enabled"]:
        _say(f"  {'pre-judge rejected':<21}{_prejudge_rejected(s)}")
    _say(f"\nSaved {details_json} (complete details)\n      {results_md} (side-by-side comparison)")


# ------------------------------------------------------------------ run
def run(run_id: str, num_papers: int, seed: int, split: str, mode_name: str = PRIVATE) -> None:
    mode = MODES[mode_name]
    if mode.missing_keys():
        raise SystemExit(f"{mode.label} needs {' and '.join(mode.missing_keys())} (see README).")
    results_md = settings.eval_dir / "result_40.md"
    details_json = settings.eval_dir / f"{run_id}.json"
    if run_id in _existing_run_ids(results_md) or details_json.exists():
        raise SystemExit(f"Run ID '{run_id}' already exists in {settings.eval_dir}; choose another.")

    data = load_split(split)
    paper_ids = sorted(random.Random(seed).sample(sorted(data), min(num_papers, len(data))))
    questions = [(pid, qa) for pid in paper_ids for qa in data[pid]["qas"]]
    doc_ids = {pid: f"eval-{run_id}-{pid}" for pid in paper_ids}
    _say(f"\n=== Run {run_id}: {len(paper_ids)} papers, {len(questions)} questions (QASPER {split}, seed {seed}) ===")
    _say(f"{mode.label}: LLM {mode.llm_model} · embeddings {mode.embed_model} · judge {settings.judge_model} · "
         f"top {mode.top_k} · BM25 {mode.bm25_candidates} · vec {mode.vector_candidates} ≥ {mode.vector_min_score} · "
         f"{'context ' + mode.context_model if settings.contextual_embedding else 'no context'} · "
         f"{'rerank top ' + str(mode.rerank_candidates) + ' with ' + settings.reranker_model if mode.reranker else 'no rerank'} · "
         f"{'pre-judge ' + mode.prejudge_model if mode.prejudge else 'no pre-judge'}")

    st = _Status("Connecting to MongoDB and loading the embedding model…")
    db.ensure_vector_index()
    embeddings.get_model()
    if mode.reranker:
        reranker.get_model()  # load it now so the first question's rerank time is not a model load
    st.done(f"Ready (vector index: {'yes' if db.vector_index_ready else 'no, using local fallback'})")

    # The phases run model by model (context model, then answering model, then judge), so Ollama
    # doesn't swap models in and out of GPU memory for every paper.
    started = time.perf_counter()
    try:
        ingested = _ingest_papers(data, paper_ids, doc_ids, mode)
        steps = _retrieve_all(questions, doc_ids, ingested.errors, mode)
        rows = _answer_all(questions, steps, ingested, mode)
    finally:
        for doc_id in doc_ids.values():
            _cleanup(doc_id)
    _judge_all(rows, mode)

    summary = _summary(run_id, split, seed, len(paper_ids), rows, ingested, (time.perf_counter() - started) / 60,
                       mode)
    _save(summary, rows, results_md, details_json)
    _print_summary(summary, results_md, details_json)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on QASPER.")
    p.add_argument("--run-id", required=True, help="unique label for this run (letters, digits, . _ -)")
    p.add_argument("--papers", type=int, default=5, help="number of papers to sample (~3.5 questions each)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed; keep it fixed to compare runs")
    p.add_argument("--split", choices=sorted(SPLITS), default="test")
    p.add_argument("--mode", choices=sorted(MODES), default=PRIVATE,
                   help="private: local models (default); cloud-rerank / cloud-prejudge: Gemini + DeepSeek via "
                        "OpenRouter with the local reranker / the Flash-Lite pre-judge (sends the papers)")
    args = p.parse_args()
    if not RUN_ID_RE.match(args.run_id):
        p.error("--run-id may only contain letters, digits, '.', '_' and '-' (max 64 chars)")
    if args.papers < 1:
        p.error("--papers must be at least 1")
    run(args.run_id, args.papers, args.seed, args.split, args.mode)


if __name__ == "__main__":
    main()
