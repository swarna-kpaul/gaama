"""
Baseline evaluation for MemoryBench Group Travel Planner.
==========================================================
Full-context baseline: for each step, ALL prior Q&A pairs are
stuffed directly into the prompt as chat history. No SDK retrieval.

Usage:
    python run_baseline.py                    # all entries
    python run_baseline.py --limit 5          # first 5 entries
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_FILE_DIR))

from config import (
    DATA_PATH, DATA_DIR, DB_NAME,
    LLM_MODEL, EVAL_MODEL, MAX_WORKERS,
)
from group_travel_eval import (
    load_dataset, compute_summary,
    GroupTravelBaselineEval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryBench Group Travel Planner baseline evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Max entries to evaluate")
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL, help="Eval judge model")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output directory name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    output_dir = (_FILE_DIR / "data" / "results" / args.output) if args.output else None

    dataset = load_dataset(DATA_PATH)
    if args.limit:
        dataset = dataset[:args.limit]

    print("=" * 70, flush=True)
    print("MemoryBench Group Travel Planner — Baseline (Full Chat History)", flush=True)
    print(f"  entries:  {len(dataset)}", flush=True)
    print("=" * 70, flush=True)

    evaluator = GroupTravelBaselineEval(
        data_dir=DATA_DIR,
        llm_model=LLM_MODEL,
        eval_model=args.eval_model,
        db_name=DB_NAME,
        max_workers=args.workers,
        output_dir=output_dir,
    )

    results = evaluator.run(dataset)
    if results:
        compute_summary(results)


if __name__ == "__main__":
    main()
