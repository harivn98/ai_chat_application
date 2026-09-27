"""QASPER evaluation of the RAG pipeline (allenai/qasper: NLP papers + questions + gold answers + evidence).

Runs only when triggered:
    python -m app.evaluate --run-id <id> [--papers 5] [--seed 0] [--split test]

Each sampled paper is converted to Markdown and ingested once (chunk + embed + store, like an upload);
every question on it then goes through hybrid retrieval + generation. Each stage is timed.

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
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone

import httpx

from . import db, llm, retrieval
from .chunker import chunk_markdown
from .config import settings
from .embeddings import embed_passages
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
    "Embedding = chunking + embedding one paper (seconds per paper); retrieval = query embedding + BM25 + "
    "vector search + RRF, generation = full LLM answer, judging = one judge call (seconds per question). "
    "Answer F1 and Evidence F1 are the official QASPER metrics (evidence = paragraphs in the passages the "
    "answer cites). Retrieval recall@k = share of gold evidence paragraphs present in the top-k chunks. "
    "Judge correct = the judge model says the answer matches a reference answer. All scores are 0-100.\n\n"
    "| Run ID | Date (UTC) | Split · papers · questions | LLM / judge | Retrieval config | Embedding (s/paper) "
    "| Retrieval (s/q) | Generation (s/q) | Judging (s/q) | Total (min) | Answer F1 "
    "| F1 extractive / abstractive / yes-no / unanswerable | Evidence F1 | Retrieval recall@k "
    "| Judge correct | Errors |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)

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
        log.info("Downloading QASPER %s split", split)
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
        "options": {"temperature": 0, "num_ctx": settings.llm_num_ctx},
    }
    timeout = httpx.Timeout(connect=10, read=600, write=60, pool=10)
    r = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"]


def _parse_verdict(text: str) -> bool | None:
    m = re.findall(r"\b(INCORRECT|CORRECT)\b", text.upper())
    return m[-1] == "CORRECT" if m else None


# ------------------------------------------------------------------ one question
def _ask(doc_id: str, qa: dict, paragraphs: list[str]) -> dict:
    question = qa["question"].strip()
    refs = _references(qa)

    t0 = time.perf_counter()
    passages = retrieval.hybrid_search(doc_id, question)
    retrieval_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    response = "".join(llm.stream_chat(llm.build_messages(question, passages, [])))
    generation_s = time.perf_counter() - t0

    cited = sorted({int(n) for n in CITE_RE.findall(response) if 1 <= int(n) <= len(passages)})
    if NOT_FOUND_RE.search(response) and not cited:
        predicted_answer, predicted_evidence = "Unanswerable", []
    else:
        predicted_answer = CITE_RE.sub("", response)
        predicted_evidence = _paragraphs_in([passages[n - 1]["text"] for n in cited], paragraphs)

    f1, answer_type = max(((token_f1_score(predicted_answer, r["answer"]), r["type"]) for r in refs),
                          key=lambda x: x[0])
    evidence_f1 = max(paragraph_f1_score(predicted_evidence, r["evidence"]) for r in refs)

    retrieved_paras = set(_paragraphs_in([p["text"] for p in passages], paragraphs))
    recalls = [len(retrieved_paras & set(r["evidence"])) / len(r["evidence"]) for r in refs if r["evidence"]]

    return {
        "question_id": qa["question_id"],
        "question": question,
        "references": refs,
        "response": response,
        "predicted_answer": predicted_answer,
        "retrieved_chunks": [p["index"] for p in passages],
        "cited_passages": cited,
        "answer_f1": f1,
        "answer_type": answer_type,
        "evidence_f1": evidence_f1,
        "retrieval_recall": max(recalls) if recalls else None,  # None: no text evidence (e.g. unanswerable)
        "retrieval_s": retrieval_s,
        "generation_s": generation_s,
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
    log.info("Run %s: %d %s papers (seed %d), %d questions", run_id, len(paper_ids), split, seed,
             sum(len(data[p]["qas"]) for p in paper_ids))

    retrieval.vector_index_ready = db.ensure_vector_index()
    started = time.perf_counter()
    rows: list[dict] = []
    embedding_times: list[float] = []

    # Phase 1: ingest each paper once, then ask all of its questions
    for n, pid in enumerate(paper_ids, start=1):
        paper = data[pid]
        doc_id = f"eval-{run_id}-{pid}"
        markdown, paragraphs = _paper_markdown(paper)
        try:
            _cleanup(doc_id)
            t0 = time.perf_counter()
            chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
            vectors = embed_passages([c.content for c in chunks])
            embedding_times.append(time.perf_counter() - t0)
            db.chunks().insert_many(
                [
                    {"doc_id": doc_id, "index": c.index, "section": c.section, "text": c.text,
                     "content": c.content, "embedding": v.tolist()}
                    for c, v in zip(chunks, vectors)
                ]
            )
            _wait_until_searchable(doc_id, len(chunks))  # index sync is excluded from the timings

            for qa in paper["qas"]:
                try:
                    row = _ask(doc_id, qa, paragraphs)
                except Exception as e:  # noqa: BLE001
                    log.exception("Question %s failed", qa["question_id"])
                    row = {"question_id": qa["question_id"], "question": qa["question"], "error": f"rag: {e}"}
                rows.append({"paper_id": pid, **row})
        except Exception as e:  # noqa: BLE001
            log.exception("Ingesting paper %s failed", pid)
            rows += [{"paper_id": pid, "question_id": qa["question_id"], "question": qa["question"],
                      "error": f"ingest: {e}"} for qa in paper["qas"]]
        finally:
            _cleanup(doc_id)
        log.info("[RAG %d/%d] paper %s: %d questions", n, len(paper_ids), pid, len(paper["qas"]))

    # Phase 2: LLM judge (kept separate so Ollama doesn't swap models between every question)
    for n, row in enumerate(rows, start=1):
        if "error" in row:
            continue
        refs = "\n".join(
            "- The paper does not answer this question." if r["type"] == "none" else f"- {r['answer']}"
            for r in row["references"]
        )
        try:
            t0 = time.perf_counter()
            out = _judge(JUDGE_PROMPT.format(question=row["question"], references=refs, response=row["response"]))
            row["judging_s"] = time.perf_counter() - t0
            row["judge_correct"], row["judge_output"] = _parse_verdict(out), out
        except Exception as e:  # noqa: BLE001
            log.exception("Judging failed for %s", row["question_id"])
            row["judge_error"] = str(e)
        log.info("[judge %d/%d] %s: F1=%.2f correct=%s", n, len(rows), row["question_id"],
                 row["answer_f1"], row.get("judge_correct"))

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
        "embedding_s_per_paper": _mean(embedding_times),
        "retrieval_s_per_question": _mean([r["retrieval_s"] for r in ok]),
        "generation_s_per_question": _mean([r["generation_s"] for r in ok]),
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
    line = (
        f"| {run_id} | {s['date_utc']} | {split} · {s['papers']} · {s['questions']} (seed {seed}) "
        f"| {s['llm_model']} / {s['judge_model']} "
        f"| top {s['top_k']} · BM25 {s['bm25_candidates']} · vec ≥ {s['vector_min_score']} "
        f"· chunk {s['chunk_size']}/{s['chunk_overlap']} "
        f"| {_secs(s['embedding_s_per_paper'])} | {_secs(s['retrieval_s_per_question'])} "
        f"| {_secs(s['generation_s_per_question'])} | {_secs(s['judging_s_per_question'])} | {total_min:.1f} "
        f"| **{_pct(s['answer_f1'])}** | {by_type} | {_pct(s['evidence_f1'])} | {_pct(s['retrieval_recall'])} "
        f"| {_pct(s['judge_correct'])} | {s['errors']} failed · {s['judge_errors']} unjudged |\n"
    )

    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    if not results_md.exists():
        results_md.write_text(RESULTS_HEADER, encoding="utf-8")
    with results_md.open("a", encoding="utf-8") as f:
        f.write(line)

    details = settings.eval_dir / f"{run_id}.json"
    details.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    print(RESULTS_HEADER.splitlines()[-2])
    print(line, end="")
    print(f"\nSaved {details} (complete details)\n      {results_md} (side-by-side comparison)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
