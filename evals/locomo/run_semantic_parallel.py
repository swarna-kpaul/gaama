"""Semantic retrieval evaluation for LoCoMo -- parallel per-sample.

Each sample runs in a separate process with its own SDK/DB connection.
Results are saved to per-sample JSONL + CSV files that are never overwritten.

Usage:
    python run_semantic_parallel.py                              # all samples
    python run_semantic_parallel.py --sample-ids conv-26 conv-30 # specific
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
    SAMPLE_IDS, SEMANTIC_BUDGET, SEMANTIC_PPR_WEIGHT, SEMANTIC_SIM_WEIGHT,
    MAX_MEMORY_WORDS, LLM_MODEL, EVAL_MODEL, EMBEDDING_MODEL, MAX_WORKERS,
    CATEGORY_NAMES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
DATA_PATH = _FILE_DIR / "locomo10.json"
DATA_DIR = _FILE_DIR / "data" / "ltm"
OUTPUT_DIR = _FILE_DIR / "data" / "results" / "semantic"
DB_NAME = "locomo_memory.sqlite"


# ---------------------------------------------------------------------------
# Per-sample worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _eval_one_sample(args_tuple) -> List[Dict[str, Any]]:
    """Evaluate a single sample. Runs in its own process."""
    sample_id, data_dir, max_words, eval_model, max_workers = args_tuple

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
        budget=SEMANTIC_BUDGET,
        max_memory_words=max_words,
        ppr_weight=SEMANTIC_PPR_WEIGHT,
        sim_weight=SEMANTIC_SIM_WEIGHT,
        semantic_only=True,
        expansion_depth=None,
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
    parser = argparse.ArgumentParser(description="Semantic eval for LoCoMo (parallel).")
    parser.add_argument("--sample-ids", nargs="*", default=None,
                        help="Specific sample IDs (default: all 10)")
    parser.add_argument("--max-words", type=int, default=MAX_MEMORY_WORDS)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    sample_ids = args.sample_ids if args.sample_ids else SAMPLE_IDS
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    print("=" * 70, flush=True)
    print("Semantic Retrieval Evaluation (parallel)", flush=True)
    print(f"  Samples: {', '.join(sample_ids)}", flush=True)
    print(f"  Budget: facts={SEMANTIC_BUDGET.max_facts} refl={SEMANTIC_BUDGET.max_reflections} "
          f"skills={SEMANTIC_BUDGET.max_skills} ep={SEMANTIC_BUDGET.max_episodes}", flush=True)
    print(f"  max_words={args.max_words}, semantic_only=True", flush=True)
    print(f"  Data dir: {data_dir}", flush=True)
    print(f"  Output: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    work = [
        (sid, data_dir, args.max_words, args.eval_model, args.max_workers)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_sample = defaultdict(list)
    for r in all_results:
        by_sample[r["sample_id"]].append(r)

    for sid in sorted(by_sample.keys()):
        _write_sample_jsonl(by_sample[sid], OUTPUT_DIR / f"{sid}.jsonl")
        _write_summary_csv(by_sample[sid], OUTPUT_DIR / f"{sid}.csv")
        print(f"  {sid}: {len(by_sample[sid])} questions -> {sid}.jsonl, {sid}.csv", flush=True)

    # Write combined files
    _write_sample_jsonl(all_results, OUTPUT_DIR / "evaluation_log.jsonl")
    _write_summary_csv(all_results, OUTPUT_DIR / "evaluation_summary.csv")
    print(f"  Combined: {len(all_results)} questions -> evaluation_log.jsonl, evaluation_summary.csv", flush=True)

    # Print summary
    n = len(all_results)
    if n == 0:
        print("\nNo results.", flush=True)
        return

    reward = sum(r["reward"] for r in all_results) / n
    correct = sum(1 for r in all_results if r.get("reward", 0) >= 1.0)

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS -- Semantic ({elapsed:.1f}s)", flush=True)
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
