"""One evaluation run: sample the papers, ingest them once, then run phases 2-4 and save a result row per variant
(one variant, or one per --sweep size or --compare cloud setting)."""
import dataclasses
import random
import time

from .. import db, embeddings, reranker, retrieval
from ..config import settings
from ..contextual import release_context_model
from ..modes import CLOUD, PRIVATE, Mode, by_variant
from . import phases, report
from .console import Status, say
from .qasper import load_split


def _variants(run_id: str, mode: Mode, sweep: list[int] | None,
              compare: list[str] | None) -> list[tuple[str, Mode]]:
    """(run id, mode) per result row: the mode as configured, one BM25 top N + vector top N per sweep value, or
    one cloud variant per compare value. All of them search the same ingested papers."""
    if sweep:
        return [(f"{run_id}-b{n}v{n}", dataclasses.replace(mode, bm25_candidates=n, vector_candidates=n))
                for n in sweep]
    if compare:
        return [(f"{run_id}-{c}", by_variant(f"{CLOUD}-{c}")) for c in compare]
    return [(run_id, mode)]


def _describe(mode: Mode) -> str:
    cutoff = "" if mode.vector_min_score is None else f" ≥ {mode.vector_min_score}"
    return (f"top {mode.top_k} · BM25 {mode.bm25_candidates} · vec {mode.vector_candidates}{cutoff} · "
            f"{'context ' + mode.context_model if settings.contextual_embedding else 'no context'} · "
            f"{'rerank all fused with ' + settings.reranker_model if mode.reranker else 'no rerank'} · "
            f"{'pre-judge ' + mode.prejudge_model if mode.prejudge else 'no pre-judge'}")


def run(run_id: str, num_papers: int, seed: int, split: str, mode_name: str = PRIVATE,
        sweep: list[int] | None = None, compare: list[str] | None = None) -> None:
    variants = _variants(run_id, by_variant(mode_name), sweep, compare)
    mode = variants[0][1]  # ingests the papers; every variant shares its embeddings and contexts
    if error := mode.missing_keys_error():
        raise SystemExit(error)
    for vid, _ in variants:
        if report.run_exists(vid):
            raise SystemExit(f"Run ID '{vid}' already exists in {settings.eval_dir}; choose another.")

    data = load_split(split)
    paper_ids = sorted(random.Random(seed).sample(sorted(data), min(num_papers, len(data))))
    questions = [(pid, qa) for pid in paper_ids for qa in data[pid]["qas"]]
    doc_ids = {pid: f"eval-{run_id}-{pid}" for pid in paper_ids}
    say(f"\n=== Run {run_id}: {len(paper_ids)} papers, {len(questions)} questions (QASPER {split}, seed {seed}) ===")
    say(f"{'Cloud mode' if compare else mode.label}: LLM {mode.llm_model} · embeddings {mode.embed_model} · "
        f"judge {settings.judge_model}")
    for vid, variant in variants:
        say(f"  {vid}: {_describe(variant)}")

    st = Status("Connecting to MongoDB and loading the embedding model…")
    db.ensure_indexes()
    embeddings.get_model()
    if any(v.reranker for _, v in variants):
        reranker.get_model()  # load it now so the first question's rerank time is not a model load
    st.done(f"Ready (vector index: {'yes' if db.vector_index_ready else 'no, using local fallback'})")

    # The phases run model by model (context model, then answering model, then judge), so Ollama
    # doesn't swap models in and out of GPU memory for every paper. A sweep or comparison ingests once.
    started = time.perf_counter()
    try:
        ingested = phases.ingest_papers(data, paper_ids, doc_ids, mode)
        ingest_min = (time.perf_counter() - started) / 60
        if mode.local and settings.contextual_embedding:
            st = Status(f"Unloading {settings.context_model}…")
            st.done(f"After contextualizing: {release_context_model() or 'nothing to hand over'}")
        for n, (vid, variant) in enumerate(variants, start=1):
            if sweep:
                say(f"\n=== Sweep {n}/{len(variants)}: {vid} (BM25 top {variant.bm25_candidates} + "
                    f"vector top {variant.vector_candidates}) ===")
            elif compare:
                say(f"\n=== Compare {n}/{len(variants)}: {vid} ({variant.label}) ===")
            t0 = time.perf_counter()
            retrievals = phases.retrieve_all(questions, doc_ids, ingested, variant)
            rows = phases.answer_all(questions, retrievals, ingested, variant)
            phases.judge_all(rows, variant)
            # each row's total counts the shared ingestion once, so it compares with a single run
            total_min = ingest_min + (time.perf_counter() - t0) / 60
            summary = report.summarize(vid, split, seed, len(paper_ids), rows, ingested, total_min, variant)
            report.save(summary, rows)
            report.print_summary(summary)
    finally:
        for doc_id in doc_ids.values():
            db.delete_chunks(doc_id)
            retrieval.drop_document(doc_id)
