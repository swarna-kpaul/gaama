"""
Run SDK eval for remaining entries not already in evaluation_log.jsonl.
Skips cleanup to preserve existing LTM databases.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_FILE_DIR))

from config import (
    DATA_PATH, DATA_DIR, DB_NAME, BUDGET,
    PPR_WEIGHT, SIM_WEIGHT, MAX_MEMORY_WORDS,
    LLM_MODEL, EVAL_MODEL,
)
from group_travel_eval import (
    load_dataset, compute_summary, save_results_jsonl,
    GroupTravelSDKEval,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = _FILE_DIR / "data" / "results" / "sdk" / "evaluation_log.jsonl"
MAX_WORKERS = 5


def load_completed_ids() -> set:
    if not RESULTS_PATH.exists():
        return set()
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return {str(json.loads(l)["entry_id"]) for l in f if l.strip()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dataset = load_dataset(DATA_PATH)
    completed = load_completed_ids()
    remaining = [e for e in dataset if str(e["id"]) not in completed]

    print("=" * 70, flush=True)
    print("MemoryBench Group Travel Planner - SDK Eval (remaining)", flush=True)
    print(f"  total:     {len(dataset)}", flush=True)
    print(f"  completed: {len(completed)}", flush=True)
    print(f"  remaining: {len(remaining)}", flush=True)
    print(f"  workers:   {MAX_WORKERS}", flush=True)
    print("=" * 70, flush=True)

    if not remaining:
        print("Nothing to do.", flush=True)
        return

    evaluator = GroupTravelSDKEval(
        data_dir=DATA_DIR,
        llm_model=LLM_MODEL,
        eval_model=EVAL_MODEL,
        db_name=DB_NAME,
        budget=BUDGET,
        ppr_weight=PPR_WEIGHT,
        sim_weight=SIM_WEIGHT,
        max_memory_words=MAX_MEMORY_WORDS,
        max_workers=MAX_WORKERS,
    )

    # --- Run WITHOUT cleanup ---
    output_dir = evaluator.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = min(MAX_WORKERS, len(remaining))
    print(f"\nSDK Eval: {len(remaining)} entries, {workers} workers\n", flush=True)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluator.evaluate_entry, entry): str(entry["id"])
            for entry in remaining
        }
        for fut in as_completed(futures):
            eid = futures[fut]
            try:
                entry_results = fut.result()
                all_results.extend(entry_results)
                avg_r = sum(r["reward"] for r in entry_results) / len(entry_results)
                print(f"[done] entry_{eid}: avg_reward={avg_r:.3f}", flush=True)
            except Exception:
                logger.error("Failed: entry_%s", eid, exc_info=True)
                print(f"ERROR: entry_{eid} failed (see log)", flush=True)

    elapsed = time.time() - t0
    print(f"\nSDK Eval done. {len(all_results)} new results in {elapsed:.1f}s", flush=True)

    if all_results:
        # Append to existing results
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Appended {len(all_results)} results to {RESULTS_PATH}", flush=True)

        # Summary over ALL results (old + new)
        with open(RESULTS_PATH, encoding="utf-8") as f:
            full_results = [json.loads(l) for l in f if l.strip()]
        print(f"\n--- Full summary ({len(full_results)} total results) ---")
        compute_summary(full_results)


if __name__ == "__main__":
    main()
