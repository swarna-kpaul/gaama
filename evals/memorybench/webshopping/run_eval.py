"""
SDK retrieval evaluation for MemoryBench WebShopping.
=====================================================
Usage:
    python run_eval.py                              # all entries
    python run_eval.py --domains baking electronics # specific domains
    python run_eval.py --limit 5                    # first 5 entries
    python run_eval.py --semantic                   # semantic-only retrieval
    python run_eval.py --ppr 0.1                    # custom PPR weight
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
    DATA_PATH, DATA_DIR, DB_NAME, BUDGET,
    PPR_WEIGHT, SIM_WEIGHT, MAX_MEMORY_WORDS,
    LLM_MODEL, EVAL_MODEL, MAX_WORKERS,
)
from webshopping_eval import (
    load_dataset, extract_domain, compute_summary,
    WebShoppingSDKEval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryBench WebShopping SDK evaluation")
    parser.add_argument("--domains", nargs="+", default=None, help="Filter by domain(s)")
    parser.add_argument("--limit", type=int, default=None, help="Max entries to evaluate")
    parser.add_argument("--ppr", type=float, default=PPR_WEIGHT, help="PPR weight")
    parser.add_argument("--sim", type=float, default=SIM_WEIGHT, help="Similarity weight")
    parser.add_argument("--semantic", action="store_true", help="Semantic-only retrieval")
    parser.add_argument("--max-words", type=int, default=MAX_MEMORY_WORDS, help="Max memory words")
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL, help="Eval judge model")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output directory name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    output_dir = (_FILE_DIR / "data" / "results" / args.output) if args.output else None

    dataset = load_dataset(DATA_PATH)
    if args.domains:
        dataset = [e for e in dataset if extract_domain(e["category"]) in args.domains]
    if args.limit:
        dataset = dataset[:args.limit]

    print("=" * 70, flush=True)
    print("MemoryBench WebShopping — SDK Evaluation", flush=True)
    print(f"  entries:  {len(dataset)}", flush=True)
    print(f"  domains:  {args.domains or 'all'}", flush=True)
    print(f"  ppr:      {args.ppr}", flush=True)
    print(f"  semantic: {args.semantic}", flush=True)
    print("=" * 70, flush=True)

    evaluator = WebShoppingSDKEval(
        data_dir=DATA_DIR,
        llm_model=LLM_MODEL,
        eval_model=args.eval_model,
        db_name=DB_NAME,
        budget=BUDGET,
        ppr_weight=args.ppr if not args.semantic else 0.0,
        sim_weight=args.sim,
        max_memory_words=args.max_words,
        semantic_only=args.semantic,
        max_workers=args.workers,
        output_dir=output_dir,
    )

    results = evaluator.run(dataset)
    if results:
        compute_summary(results)


if __name__ == "__main__":
    main()
