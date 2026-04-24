"""PPR baseline evaluation for LoCoMo -- parallel per-sample.

Runs PPR-based retrieval (ppr_weight=1.0 by default) with graph expansion,
then evaluates with fractional hypothesis judge. Each sample runs in a
separate process. Results saved to per-sample JSONL + CSV files.

Usage:
    python run_ppr_baseline.py                                    # all samples, ppr=1.0
    python run_ppr_baseline.py --ppr-weight 0.5                   # custom PPR weight
    python run_ppr_baseline.py --sample-ids conv-26 conv-30       # specific samples
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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
    CATEGORY_NAMES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
DATA_PATH = _FILE_DIR / "locomo10.json"
DATA_DIR = _FILE_DIR / "data" / "ltm"
DB_NAME = "locomo_memory.sqlite"


# ---------------------------------------------------------------------------
# Per-sample worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _eval_one_sample(args_tuple) -> List[Dict[str, Any]]:
    """Evaluate a single sample. Runs in its own process."""
    sample_id, data_dir, max_words, eval_model, max_workers, ppr_weight, sim_weight, expansion_depth = args_tuple

    from gaama.api import create_default_sdk
    from gaama.config.settings import (
        SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
    )
    from locomo_eval import evaluate_sample

    settings = SDKSettings(
        storage=StorageSettings(
            sqlite_path=data_dir / DB_NAME,
            blob_root=data_dir / "blobs",
        ),
        llm=LLMSettings(
            provider="openai", model=LLM_MODEL,
            api_key_env="OPENAI_API_KEY", max_tokens=16000,
        ),
        embedding=EmbeddingSettings(
            model=EMBEDDING_MODEL, api_key_env="OPENAI_API_KEY",
        ),
    )
    sdk = create_default_sdk(settings, agent_id=AGENT_ID)

    results = evaluate_sample(
        sdk=sdk,
        data_path=DATA_PATH,
        sample_id=sample_id,
        categories=list(CATEGORY_NAMES.keys()),
        budget=PPR_BUDGET,
        max_memory_words=max_words,
        ppr_weight=ppr_weight,
        sim_weight=sim_weight,
        semantic_only=False,
        expansion_depth=expansion_depth,
        eval_model=eval_model,
        max_workers=max_workers,
    )
    return [r for r in results if r.get("sample_id") == sample_id]


# ---------------------------------------------------------------------------
# Result writing
# ---------------------------------------------------------------------------

def _write_sample_jsonl(results: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_summary_csv(results: List[Dict[str, Any]], csv_path: Path):
    stats = defaultdict(list)
    sample_stats = defaultdict(list)
    category_stats = defaultdict(list)

    for r in results:
        sid = r["sample_id"]
        cat_name = r.get("question_type", f"cat{r['category']}_unknown")
        reward = r["reward"]
        stats[(sid, cat_name)].append(reward)
        sample_stats[sid].append(reward)
        category_stats[cat_name].append(reward)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "category", "num_questions", "mean_reward", "num_correct", "accuracy"])
        for (sid, cat_name) in sorted(stats.keys()):
            rewards = stats[(sid, cat_name)]
            n = len(rewards)
            correct = sum(1 for r in rewards if r > 0.5)
            mean_r = sum(rewards) / n if n else 0.0
            writer.writerow([sid, cat_name, n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])
        writer.writerow([])
        writer.writerow(["TOTAL_BY_CATEGORY", "category", "num_questions", "mean_reward", "num_correct", "accuracy"])
        for cat_name in sorted(category_stats.keys()):
            rewards = category_stats[cat_name]
            n = len(rewards)
            correct = sum(1 for r in rewards if r > 0.5)
            mean_r = sum(rewards) / n if n else 0.0
            writer.writerow(["ALL", cat_name, n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])
        writer.writerow([])
        writer.writerow(["TOTAL_BY_SAMPLE", "category", "num_questions", "mean_reward", "num_correct", "accuracy"])
        for sid in sorted(sample_stats.keys()):
            rewards = sample_stats[sid]
            n = len(rewards)
            correct = sum(1 for r in rewards if r > 0.5)
            mean_r = sum(rewards) / n if n else 0.0
            writer.writerow([sid, "ALL", n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])
        all_rewards = [r["reward"] for r in results]
        n = len(all_rewards)
        correct = sum(1 for r in all_rewards if r > 0.5)
        mean_r = sum(all_rewards) / n if n else 0.0
        writer.writerow([])
        writer.writerow(["OVERALL", "ALL", n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPR baseline eval for LoCoMo (parallel).")
    parser.add_argument("--sample-ids", nargs="*", default=None,
                        help="Specific sample IDs (default: all 10)")
    parser.add_argument("--ppr-weight", type=float, default=1.0,
                        help="PPR score weight (default: 1.0)")
    parser.add_argument("--sim-weight", type=float, default=PPR_SIM_WEIGHT,
                        help=f"Similarity score weight (default: {PPR_SIM_WEIGHT})")
    parser.add_argument("--expansion-depth", type=int, default=PPR_EXPANSION_DEPTH,
                        help=f"PPR expansion depth (default: {PPR_EXPANSION_DEPTH})")
    parser.add_argument("--max-words", type=int, default=MAX_MEMORY_WORDS)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: data/results/ppr_<weight>)")
    args = parser.parse_args()

    sample_ids = args.sample_ids if args.sample_ids else SAMPLE_IDS
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Build output dir name from PPR weight (e.g. ppr_1.0, ppr_0.1)
    ppr_tag = f"ppr_{args.ppr_weight}"
    output_dir = Path(args.output_dir) if args.output_dir else (_FILE_DIR / "data" / "results" / ppr_tag)

    # Safety: refuse to overwrite existing output
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"ERROR: Output directory already exists and is non-empty: {output_dir}", flush=True)
        print("       Remove it manually or use --output-dir to specify a different path.", flush=True)
        sys.exit(1)

    print("=" * 70, flush=True)
    print("PPR Baseline Evaluation (parallel)", flush=True)
    print(f"  Samples:         {', '.join(sample_ids)}", flush=True)
    print(f"  Categories:      {list(CATEGORY_NAMES.keys())}", flush=True)
    print(f"  Budget:          facts={PPR_BUDGET.max_facts} refl={PPR_BUDGET.max_reflections} "
          f"skills={PPR_BUDGET.max_skills} ep={PPR_BUDGET.max_episodes}", flush=True)
    print(f"  ppr_weight:      {args.ppr_weight}", flush=True)
    print(f"  sim_weight:      {args.sim_weight}", flush=True)
    print(f"  expansion_depth: {args.expansion_depth}", flush=True)
    print(f"  semantic_only:   False", flush=True)
    print(f"  max_words:       {args.max_words}", flush=True)
    print(f"  eval_model:      {args.eval_model}", flush=True)
    print(f"  max_workers:     {args.max_workers}", flush=True)
    print(f"  Data dir:        {data_dir}", flush=True)
    print(f"  Output:          {output_dir}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    work = [
        (sid, data_dir, args.max_words, args.eval_model, args.max_workers,
         args.ppr_weight, args.sim_weight, args.expansion_depth)
        for sid in sample_ids
    ]

    if len(sample_ids) == 1:
        all_results = _eval_one_sample(work[0])
    else:
        with multiprocessing.Pool(processes=len(sample_ids)) as pool:
            async_results = {
                sid: pool.apply_async(_eval_one_sample, (w,))
                for sid, w in zip(sample_ids, work)
            }
            all_results = []
            for sid in sample_ids:
                try:
                    r = async_results[sid].get()
                    all_results.extend(r)
                    n = len(r)
                    avg = sum(x["reward"] for x in r) / n if n else 0
                    print(f"  [DONE] {sid}: {n} questions, mean_reward={avg:.4f}", flush=True)
                except Exception as e:
                    print(f"  [FAIL] {sid}: {e}", flush=True)

    elapsed = time.time() - t0

    # Write per-sample files
    output_dir.mkdir(parents=True, exist_ok=True)
    by_sample = defaultdict(list)
    for r in all_results:
        by_sample[r["sample_id"]].append(r)

    for sid in sorted(by_sample.keys()):
        _write_sample_jsonl(by_sample[sid], output_dir / f"{sid}.jsonl")
        _write_summary_csv(by_sample[sid], output_dir / f"{sid}.csv")
        print(f"  {sid}: {len(by_sample[sid])} questions -> {sid}.jsonl, {sid}.csv", flush=True)

    # Write combined files
    _write_sample_jsonl(all_results, output_dir / "evaluation_log.jsonl")
    _write_summary_csv(all_results, output_dir / "evaluation_summary.csv")
    print(f"  Combined: {len(all_results)} questions -> evaluation_log.jsonl, evaluation_summary.csv", flush=True)

    # Print summary
    n = len(all_results)
    if n == 0:
        print("\nNo results.", flush=True)
        return

    reward = sum(r["reward"] for r in all_results) / n
    correct = sum(1 for r in all_results if r.get("reward", 0) >= 1.0)

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS -- PPR Baseline ppr_weight={args.ppr_weight} ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Sample':<12} {'N':>5} {'Reward':>10} {'Acc%':>10}", flush=True)
    print("-" * 42, flush=True)
    for sid in sorted(by_sample.keys()):
        sr = by_sample[sid]
        sn = len(sr)
        savg = sum(r["reward"] for r in sr) / sn
        sc = sum(1 for r in sr if r.get("reward", 0) >= 1.0)
        print(f"  {sid:<10} {sn:>5} {savg:>9.4f} {sc/sn*100:>9.1f}%", flush=True)
    print("-" * 42, flush=True)
    print(f"  {'OVERALL':<10} {n:>5} {reward:>9.4f} {correct/n*100:>9.1f}%", flush=True)

    print(f"\n{'Category':<25} {'N':>5} {'Reward':>10} {'Acc%':>10}", flush=True)
    print("-" * 55, flush=True)
    for cat in sorted(CATEGORY_NAMES.keys()):
        cat_name = CATEGORY_NAMES[cat]
        cr = [r for r in all_results if r.get("category") == cat]
        cn = len(cr)
        if cn == 0:
            continue
        cavg = sum(r["reward"] for r in cr) / cn
        cc = sum(1 for r in cr if r.get("reward", 0) >= 1.0)
        print(f"  {cat_name:<23} {cn:>5} {cavg:>9.4f} {cc/cn*100:>9.1f}%", flush=True)
    print("-" * 55, flush=True)
    print(f"  {'OVERALL':<23} {n:>5} {reward:>9.4f} {correct/n*100:>9.1f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
