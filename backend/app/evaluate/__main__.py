"""Command line: python -m app.evaluate --run-id <id> [options] (see the package docstring)."""
import argparse
import logging
import re

from ..modes import CLOUD, CLOUD_VARIANTS, PRIVATE
from .qasper import SPLITS
from .run import run

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _sweep_values(text: str) -> list[int]:
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated whole numbers, e.g. 30,25,20,15") from None
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("use distinct numbers of at least 1, e.g. 30,25,20,15")
    return values


def _compare_values(text: str) -> list[str]:
    choices = [name.removeprefix(f"{CLOUD}-") for name in CLOUD_VARIANTS]
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values or len(set(values)) != len(values) or any(v not in choices for v in values):
        raise argparse.ArgumentTypeError(f"use distinct values from {', '.join(choices)}, "
                                         "e.g. rerank,prejudge,rerank-prejudge")
    return values


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m app.evaluate", description="Evaluate the RAG pipeline on QASPER.")
    p.add_argument("--run-id", required=True, help="unique label for this run (letters, digits, . _ -)")
    p.add_argument("--papers", type=int, default=5, help="number of papers to sample (~3.5 questions each)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed; keep it fixed to compare runs")
    p.add_argument("--split", choices=sorted(SPLITS), default="test")
    p.add_argument("--mode", choices=[PRIVATE, *CLOUD_VARIANTS],
                   help="private: local models (default); cloud-rerank / cloud-prejudge / cloud-rerank-prejudge: "
                        "Gemini + DeepSeek via OpenRouter with the local reranker, the Flash-Lite pre-judge, or both "
                        "(sends the papers)")
    extra = p.add_mutually_exclusive_group()
    extra.add_argument("--sweep", type=_sweep_values, metavar="N,M,...",
                       help="ingest once, then evaluate BM25 top N + vector top N for each value (one row each, "
                            "run IDs <run-id>-bNvN), e.g. 30,25,20,15")
    extra.add_argument("--compare", type=_compare_values, metavar="VARIANT,...",
                       help="ingest once in cloud mode, then evaluate each cloud variant (one row each, run IDs "
                            "<run-id>-<variant>), e.g. rerank,prejudge,rerank-prejudge; don't combine with --mode")
    args = p.parse_args()
    if args.compare and args.mode:
        p.error("--compare picks the cloud variants itself; leave out --mode")
    if not RUN_ID_RE.match(args.run_id):
        p.error("--run-id may only contain letters, digits, '.', '_' and '-' (max 64 chars)")
    if args.papers < 1:
        p.error("--papers must be at least 1")
    run(args.run_id, args.papers, args.seed, args.split, args.mode or PRIVATE, args.sweep, args.compare)


if __name__ == "__main__":
    main()
