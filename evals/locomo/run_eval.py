"""Unified evaluation wrapper for LoCoMo samples.

Runs evaluation on either the original (baseline) DB or the GEL-modified DB
for each sample.  Both DBs live side-by-side under data/gel_run/<sample_id>/:

    locomo_original_db.sqlite   -- pristine LTM (baseline)
    locomo_gels_db.sqlite       -- LTM after GEL edits

Usage:
    # Baseline evaluation for one sample
    python run_eval.py --mode baseline --sample conv-41

    # GEL evaluation for all samples (except conv-26)
    python run_eval.py --mode gel --sample conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50

    # Both modes for comparison
    python run_eval.py --mode both --sample conv-41 conv-42

    # All samples, both modes
    python run_eval.py --mode both --sample all

    # Custom PPR weight
    python run_eval.py --mode gel --sample conv-41 --ppr 0.3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_FILE_DIR))

from config import (
    SAMPLE_IDS, PPR_BUDGET, PPR_WEIGHT, PPR_SIM_WEIGHT, PPR_EXPANSION_DEPTH,
    MAX_MEMORY_WORDS, LLM_MODEL, EVAL_MODEL, EMBEDDING_MODEL, MAX_WORKERS,
    CATEGORY_NAMES, DATA_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
GEL_ROOT = _FILE_DIR / "data" / "gel_run"
OUTPUT_DIR = _FILE_DIR / "data" / "results"

DB_NAMES = {
    "baseline": "locomo_original_db.sqlite",
    "gel":      "locomo_gels_db.sqlite",
}


# ---------------------------------------------------------------------------
# SDK factory
# ---------------------------------------------------------------------------

def _make_sdk(db_path: Path, blob_root: Path):
    from gaama.api import create_default_sdk
    from gaama.config.settings import (
        SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
    )
    settings = SDKSettings(
        storage=StorageSettings(
            sqlite_path=db_path,
            blob_root=blob_root,
        ),
        llm=LLMSettings(
            provider="openai",
            model=LLM_MODEL,
            api_key_env="OPENAI_API_KEY",
            max_tokens=16000,
        ),
        embedding=EmbeddingSettings(
            model=EMBEDDING_MODEL,
            api_key_env="OPENAI_API_KEY",
        ),
    )
    return create_default_sdk(settings, agent_id=AGENT_ID)


# ---------------------------------------------------------------------------
# Per-sample evaluation (runs in subprocess for parallelism)
# ---------------------------------------------------------------------------

def _eval_one(args_tuple) -> Dict[str, Any]:
    """Evaluate a single (sample_id, mode) pair.

    args_tuple: (sample_id, mode, ppr_w)
    """
    sample_id, mode, ppr_w = args_tuple

    from locomo_eval import evaluate_sample, save_results_jsonl

    sdir = GEL_ROOT / sample_id
    db_path = sdir / DB_NAMES[mode]
    blob_root = sdir / "blobs"

    if not db_path.exists():
        print(f"  [{sample_id}/{mode}] DB not found: {db_path}", flush=True)
        return {"sample_id": sample_id, "mode": mode, "error": "DB not found"}

    sdk = _make_sdk(db_path, blob_root)

    t0 = time.time()
    results = evaluate_sample(
        sdk=sdk,
        data_path=DATA_PATH,
        sample_id=sample_id,
        categories=list(CATEGORY_NAMES.keys()),
        budget=PPR_BUDGET,
        max_memory_words=MAX_MEMORY_WORDS,
        ppr_weight=ppr_w,
        sim_weight=PPR_SIM_WEIGHT,
        semantic_only=False,
        expansion_depth=PPR_EXPANSION_DEPTH,
        eval_model=EVAL_MODEL,
        max_workers=MAX_WORKERS,
    )
    results = [r for r in results if r.get("sample_id") == sample_id]
    elapsed = time.time() - t0

    n = len(results)
    avg = sum(r["reward"] for r in results) / n if n else 0
    print(f"  [{sample_id}/{mode}] {n} questions, reward={avg:.4f} ({elapsed:.1f}s)",
          flush=True)

    # Save results
    out_dir = OUTPUT_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sample_id}.jsonl"
    save_results_jsonl(results, out_file)

    # Per-category breakdown
    cats = {}
    for r in results:
        cat = r.get("question_type", "?")
        cats.setdefault(cat, []).append(r["reward"])

    return {
        "sample_id": sample_id,
        "mode": mode,
        "n": n,
        "reward": avg,
        "elapsed": elapsed,
        "categories": {c: sum(v)/len(v) for c, v in cats.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoCoMo samples on baseline or GEL databases.",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["baseline", "gel", "both"],
        help="Which DB to evaluate: 'baseline' (original), 'gel' (post-GEL), or 'both'",
    )
    parser.add_argument(
        "--sample", type=str, required=True, nargs="+",
        help="Sample ID(s) or 'all' for all 10 samples",
    )
    parser.add_argument(
        "--ppr", type=float, default=PPR_WEIGHT,
        help=f"PPR weight (default: {PPR_WEIGHT})",
    )
    args = parser.parse_args()

    samples = SAMPLE_IDS if args.sample == ["all"] else args.sample
    modes = ["baseline", "gel"] if args.mode == "both" else [args.mode]
    ppr_w = args.ppr

    # Build work items
    work: List[Tuple[str, str, float]] = []
    for mode in modes:
        for sid in samples:
            work.append((sid, mode, ppr_w))

    print("=" * 70)
    print(f"GAAMA EVAL -- {len(work)} job(s)")
    print(f"  Samples: {', '.join(samples)}")
    print(f"  Modes:   {', '.join(modes)}")
    print(f"  PPR={ppr_w}, sim={PPR_SIM_WEIGHT}, depth={PPR_EXPANSION_DEPTH}")
    print(f"  Budget:  {PPR_BUDGET.max_facts}F/{PPR_BUDGET.max_reflections}R/"
          f"{PPR_BUDGET.max_skills}S/{PPR_BUDGET.max_episodes}E")
    print(f"  Words:   {MAX_MEMORY_WORDS}")
    print(f"  Eval:    {EVAL_MODEL}")
    print(f"  DBs:     {', '.join(DB_NAMES[m] for m in modes)}")
    print(f"  Output:  {OUTPUT_DIR}/<mode>/<sample>.jsonl")
    print("=" * 70)

    t0 = time.time()

    if len(work) == 1:
        summaries = [_eval_one(work[0])]
    else:
        n_procs = min(len(work), 10)
        with multiprocessing.Pool(processes=n_procs) as pool:
            async_results = {
                (sid, mode): pool.apply_async(_eval_one, (w,))
                for (sid, mode, _), w in zip(work, work)
            }
            summaries = []
            for key in async_results:
                try:
                    summaries.append(async_results[key].get())
                except Exception as e:
                    print(f"  [FAIL] {key}: {e}")

    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"EVAL SUMMARY ({elapsed:.1f}s)")
    print(f"{'=' * 70}")

    for mode in modes:
        mode_results = [s for s in summaries if s.get("mode") == mode and "error" not in s]
        if not mode_results:
            continue

        print(f"\n--- {mode.upper()} ---")
        print(f"{'Sample':<12} {'N':>5} {'Reward':>8}"
              f"  {'factual':>8} {'temporal':>8} {'infer':>8} {'detail':>8}")
        print("-" * 72)

        for s in sorted(mode_results, key=lambda x: x["sample_id"]):
            cats = s.get("categories", {})
            print(f"  {s['sample_id']:<10} {s['n']:>5} {s['reward']:>8.4f}"
                  f"  {cats.get('cat1_factual', 0):>8.4f}"
                  f" {cats.get('cat2_temporal', 0):>8.4f}"
                  f" {cats.get('cat3_inference', 0):>8.4f}"
                  f" {cats.get('cat4_detailed', 0):>8.4f}")

        if len(mode_results) > 1:
            total_n = sum(s["n"] for s in mode_results)
            total_r = sum(s["reward"] * s["n"] for s in mode_results) / total_n
            print("-" * 72)
            print(f"  {'OVERALL':<10} {total_n:>5} {total_r:>8.4f}")

    # -----------------------------------------------------------------------
    # Side-by-side comparison when mode=both
    # -----------------------------------------------------------------------
    if args.mode == "both":
        base_map = {s["sample_id"]: s for s in summaries
                    if s.get("mode") == "baseline" and "error" not in s}
        gel_map = {s["sample_id"]: s for s in summaries
                   if s.get("mode") == "gel" and "error" not in s}

        common = sorted(set(base_map) & set(gel_map))
        if common:
            print(f"\n--- COMPARISON (baseline vs gel) ---")
            print(f"{'Sample':<12} {'N':>5} {'Baseline':>10} {'GEL':>10} {'Delta':>8}")
            print("-" * 50)
            for sid in common:
                b = base_map[sid]
                g = gel_map[sid]
                print(f"  {sid:<10} {b['n']:>5} {b['reward']:>10.4f}"
                      f" {g['reward']:>10.4f} {g['reward']-b['reward']:>+8.4f}")
            if len(common) > 1:
                bn = sum(base_map[s]["n"] for s in common)
                br = sum(base_map[s]["reward"] * base_map[s]["n"] for s in common) / bn
                gr = sum(gel_map[s]["reward"] * gel_map[s]["n"] for s in common) / bn
                print("-" * 50)
                print(f"  {'OVERALL':<10} {bn:>5} {br:>10.4f}"
                      f" {gr:>10.4f} {gr-br:>+8.4f}")

    print("=" * 70)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
