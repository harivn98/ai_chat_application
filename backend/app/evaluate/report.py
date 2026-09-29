"""A run's outputs in EVAL_DIR: the summary (metrics, timings, settings), its row in result_40.md, <run_id>.json
with every question, and the summary printed at the end."""
import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .. import prejudge
from ..config import settings
from ..modes import PRIVATE, Mode
from .console import say
from .phases import IngestedPapers
from .qasper import ANSWER_TYPES

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
    "BM25 chunks = chunks BM25 returned per question, vector chunks = chunks vector search returned per question "
    "(at or above the minimum score, if the mode has one; both as average (fewest–most)); vector score range = "
    "lowest–highest cosine "
    "similarity of those vector chunks over the whole run, and in brackets the average of each question's "
    "lowest and highest.\n\n"
    "| Run ID | Date (UTC) | Split · papers · questions | LLM / judge | Retrieval config "
    "| BM25 chunks (per q) | Vector chunks (per q) | Vector score range "
    "| Contextualization (s/paper) | Embedding (s/paper) "
    "| Retrieval (s/q) | Rerank (s/q) | Pre-judge (s/q) | Generation (s/q) | Judging (s/q) | Total (min) | Answer F1 "
    "| F1 extractive / abstractive / yes-no / unanswerable | Evidence F1 | Retrieval recall@k "
    "| Judge correct | Pre-judge rejected | Errors |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _results_md() -> Path:
    return settings.eval_dir / "result_40.md"


def _details_json(run_id: str) -> Path:
    return settings.eval_dir / f"{run_id}.json"


def run_exists(run_id: str) -> bool:
    """result_40.md already has a row with this run ID, or <run_id>.json exists."""
    return run_id in _existing_run_ids(_results_md()) or _details_json(run_id).exists()


def _existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {m.group(1) for m in re.finditer(r"^\| (\S+) \|", path.read_text(encoding="utf-8"), re.MULTILINE)}


# ------------------------------------------------------------------ summary
def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _spread(vals: list) -> dict | None:
    vals = [v for v in vals if v is not None]
    return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)} if vals else None


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


def summarize(run_id: str, split: str, seed: int, papers: int, rows: list[dict], ingested: IngestedPapers,
              total_min: float, mode: Mode) -> dict:
    ok = [r for r in rows if "error" not in r]
    return {
        "run_id": run_id,
        "date_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}",
        "split": split,
        "seed": seed,
        "papers": papers,
        "questions": len(rows),
        "mode": mode.variant,  # private, cloud-rerank, cloud-prejudge or cloud-rerank-prejudge
        "embed_model": mode.embed_model,
        "llm_model": mode.llm_model,
        "judge_model": settings.judge_model,
        "top_k": mode.top_k,
        "bm25_candidates": mode.bm25_candidates,
        "vector_candidates": mode.vector_candidates,
        "vector_min_score": mode.vector_min_score,  # None: no cutoff
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "contextual_embedding": settings.contextual_embedding,
        "context_model": mode.context_model if settings.contextual_embedding else None,
        "prejudge_enabled": mode.prejudge,
        "prejudge_model": mode.prejudge_model if mode.prejudge else None,
        "reranker_enabled": mode.reranker,
        "reranker_model": settings.reranker_model if mode.reranker else None,
        "contextualization_s_per_paper": _mean(ingested.context_times),
        "embedding_s_per_paper": _mean(ingested.embedding_times),
        # per question: chunks BM25 returned, vector hits (after the cutoff, if any), and their cosine range
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


# ------------------------------------------------------------------ formatting
def _secs(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


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
    cloud = "" if local else f"cloud · emb {s['embed_model']} · "
    ctx = f"ctx {s['context_model']}" if s["contextual_embedding"] else "no ctx"
    pj = f"prejudge {s['prejudge_model']}{'/gpu' if local else ''}" if s["prejudge_enabled"] else "no prejudge"
    rr = f"rerank {s['reranker_model'].split('/')[-1]} all fused" if s["reranker_enabled"] else "no rerank"
    vec = f"vec top {s['vector_candidates']}" + ("" if s["vector_min_score"] is None else f" ≥ {s['vector_min_score']}")
    return (
        f"| {s['run_id']} | {s['date_utc']} | {s['split']} · {s['papers']} · {s['questions']} (seed {s['seed']}) "
        f"| {s['llm_model']} / {s['judge_model']} "
        f"| {cloud}top {s['top_k']} · BM25 {s['bm25_candidates']} · {vec} "
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


# ------------------------------------------------------------------ output
def save(summary: dict, rows: list[dict]) -> None:
    """Append the run's row to result_40.md (created with its header if missing) and write <run_id>.json."""
    results_md = _results_md()
    results_md.parent.mkdir(parents=True, exist_ok=True)
    if not results_md.exists():
        results_md.write_text(RESULTS_HEADER, encoding="utf-8")
    with results_md.open("a", encoding="utf-8") as f:
        f.write(_results_row(summary))
    _details_json(summary["run_id"]).write_text(
        json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(s: dict) -> None:
    say(f"\n=== Done in {s['total_minutes']:.1f} min: {s['questions'] - s['errors']}/{s['questions']} "
        f"questions answered ===")
    for name, key in [(f"retrieval recall@{s['top_k']}", "retrieval_recall"), ("evidence F1", "evidence_f1"),
                      ("answer F1", "answer_f1"), ("judge correct", "judge_correct")]:
        say(f"  {name:<21}{_pct(s[key])}")
    say(f"  {'BM25 chunks/q':<21}{_hit_counts(s['bm25_hits'])}")
    cutoff = "no score cutoff" if s["vector_min_score"] is None else f"at or above {s['vector_min_score']}"
    say(f"  {'vector chunks/q':<21}{_hit_counts(s['vector_hits'])}  ({cutoff})")
    say(f"  {'vector score range':<21}{_score_range(s)}")
    if s["reranker_enabled"]:
        say(f"  {'candidate recall':<21}{_pct(s['candidate_recall'])}"
            f"  (evidence among all the fused chunks the reranker chose from)")
    if s["prejudge_enabled"]:
        say(f"  {'pre-judge rejected':<21}{_prejudge_rejected(s)}")
    say(f"\nSaved {_details_json(s['run_id'])} (complete details)\n      {_results_md()} (side-by-side comparison)")
