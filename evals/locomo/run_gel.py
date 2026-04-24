"""Run GEL (Graph Edit Learning) for LoCoMo samples, then re-evaluate.

Each sample gets its own DB copy to avoid SQLite write conflicts when
running in parallel.

Usage:
    python run_gel.py --sample conv-42           # single sample
    python run_gel.py --sample all               # all 10 samples in parallel
    python run_gel.py --sample all --ppr 0.1     # custom PPR weight
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent  # gaama/evals/locomo -> repo root
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
USER_ID = "locomo"
DATA_DIR = _FILE_DIR / "data" / "ltm"
GEL_ROOT = _FILE_DIR / "data" / "gel_run"
OUTPUT_DIR = _FILE_DIR / "data" / "results"
DB_NAME = "locomo_memory.sqlite"

# GEL defaults
GEL_FACT_BELIEF = 0.85
GEL_MAX_FACTS_PER_QUERY = 2
GEL_REWARD_THRESHOLD = 0.5
GEL_MAX_ANALYSIS_QUESTIONS = 3

GEL_SUB_BUDGET_FACTS = 5
GEL_SUB_BUDGET_EPISODES = 10
GEL_SUB_MAX_MEMORY_WORDS = 600


# ---------------------------------------------------------------------------
# SDK factory
# ---------------------------------------------------------------------------

def _make_sdk(data_dir: Path):
    from gaama.api import create_default_sdk
    from gaama.config.settings import (
        SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
    )
    settings = SDKSettings(
        storage=StorageSettings(
            sqlite_path=data_dir / DB_NAME,
            blob_root=data_dir / "blobs",
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
# Per-sample GEL pipeline (runs in subprocess)
# ---------------------------------------------------------------------------

def _run_sample_gel(args_tuple) -> Dict[str, Any]:
    """Full GEL pipeline for one sample: setup DB -> baseline -> GEL -> final eval.

    args_tuple: (sample_id, ppr_w)
    """
    sample_id, ppr_w = args_tuple

    from gaama.core import QueryFilters, Scope, RetrievalBudget
    from gaama.core.types import GELConfig
    from gaama.services.interfaces import RetrieveOptions
    from gaama.services.graph_edit_learner import GraphEditLearner
    from gaama.services import answer_from_memory
    from locomo_eval import evaluate_sample, save_results_jsonl

    # 1. Setup per-sample DB copy using SQLite backup API
    sdir = GEL_ROOT / sample_id
    sdir.mkdir(parents=True, exist_ok=True)
    src_db = sqlite3.connect(str(DATA_DIR / DB_NAME))
    dst_db = sqlite3.connect(str(sdir / DB_NAME))
    src_db.backup(dst_db)
    src_db.close()
    dst_db.close()
    # Copy blobs if they exist
    src_blobs = DATA_DIR / "blobs"
    dst_blobs = sdir / "blobs"
    if src_blobs.exists() and not dst_blobs.exists():
        shutil.copytree(src_blobs, dst_blobs)
    print(f"  [{sample_id}] DB ready", flush=True)

    # 2. Baseline evaluation
    sdk = _make_sdk(sdir)
    baseline = evaluate_sample(
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
    baseline = [r for r in baseline if r.get("sample_id") == sample_id]
    b_n = len(baseline)
    b_avg = sum(r["reward"] for r in baseline) / b_n if b_n else 0
    print(f"  [{sample_id}] BASELINE: {b_n} questions, reward={b_avg:.4f}", flush=True)

    # 3. GEL pass
    orch = sdk._orchestrator
    scope = Scope(agent_id=AGENT_ID, user_id=USER_ID, task_id=sample_id)
    filters = QueryFilters(agent_id=AGENT_ID, user_id=USER_ID, task_id=sample_id)

    gel_config = GELConfig(
        enabled=True,
        reward_threshold=GEL_REWARD_THRESHOLD,
        max_analysis_questions=GEL_MAX_ANALYSIS_QUESTIONS,
        max_edits_per_query=4,
        verify_after_edit=False,
        gel_fact_belief=GEL_FACT_BELIEF,
        max_facts_per_query=GEL_MAX_FACTS_PER_QUERY,
        max_concepts_per_query=2,
    )

    _llm = getattr(orch._extractor, "_llm", None)
    gel = GraphEditLearner(
        node_store=orch._node_store,
        graph_store=orch._graph_store,
        vector_store=orch._vector_store,
        embedder=orch._embedder,
        llm=_llm,
        retriever=orch._ltm_retriever,
        config=gel_config,
    )

    sub_budget = RetrievalBudget(
        max_facts=GEL_SUB_BUDGET_FACTS,
        max_reflections=0,
        max_skills=0,
        max_episodes=GEL_SUB_BUDGET_EPISODES,
    )
    sub_opts = RetrieveOptions(
        filters=filters,
        budget=sub_budget,
        sources="ltm",
        semantic_only=False,
        ppr_score_weight=ppr_w,
        sim_score_weight=PPR_SIM_WEIGHT,
        expansion_depth=PPR_EXPANSION_DEPTH,
        max_memory_words=GEL_SUB_MAX_MEMORY_WORDS,
    )

    # Load questions
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    questions = []
    for item in data:
        if item.get("sample_id") == sample_id:
            questions = [
                q for q in item.get("qa", [])
                if q.get("category") in CATEGORY_NAMES and q.get("answer")
            ]
            break

    triggered = 0
    total_facts = 0
    total_concepts = 0
    total = len(questions)

    for i, qa in enumerate(questions):
        question = qa["question"]
        category = qa.get("category", 0)
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}")

        pack, scored_items = orch._ltm_retriever.retrieve(question, sub_opts)

        # Generate hypothesis from retrieved memory so the GEL judge can
        # evaluate whether the current retrieval already answers the question.
        hypothesis = answer_from_memory(
            question, pack, llm=_llm, model=EVAL_MODEL, temperature=0,
        ) or ""

        try:
            report = gel.learn_from_failure(
                query=question,
                ground_truth=None,
                hypothesis=hypothesis,
                reward_before=0.0,
                memory_pack=pack,
                scored_items=scored_items,
                scope=scope,
                filters=filters,
                retrieve_options=sub_opts,
            )
            if report.edits_executed > 0:
                triggered += 1
                nf = sum(1 for op in report.edit_ops if op.op_type == "CREATE_FACT")
                nc = sum(1 for op in report.edit_ops if op.op_type == "CREATE_CONCEPT")
                total_facts += nf
                total_concepts += nc
                print(f"  [GEL {sample_id} {i+1}/{total}] TRIGGERED {cat_name} "
                      f"+{nf}F +{nc}C  Q: {question[:55]}", flush=True)
            else:
                print(f"  [GEL {sample_id} {i+1}/{total}] SKIP {cat_name}", flush=True)
        except Exception as e:
            print(f"  [GEL {sample_id} {i+1}/{total}] ERROR: {e}", flush=True)

    print(f"  [{sample_id}] GEL done: {triggered}/{total} triggered, "
          f"+{total_facts}F +{total_concepts}C", flush=True)

    # 4. Final evaluation (fresh SDK to pick up new facts)
    sdk2 = _make_sdk(sdir)
    final = evaluate_sample(
        sdk=sdk2,
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
    final = [r for r in final if r.get("sample_id") == sample_id]
    f_n = len(final)
    f_avg = sum(r["reward"] for r in final) / f_n if f_n else 0
    print(f"  [{sample_id}] FINAL: {f_n} questions, reward={f_avg:.4f}", flush=True)

    # Save results
    out_dir = OUTPUT_DIR / "gel"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results_jsonl(baseline, out_dir / f"{sample_id}_baseline.jsonl")
    save_results_jsonl(final, out_dir / f"{sample_id}_post_gel.jsonl")

    return {
        "sample_id": sample_id,
        "n": b_n,
        "baseline_reward": b_avg,
        "final_reward": f_avg,
        "delta": f_avg - b_avg,
        "gel_triggered": triggered,
        "gel_total": total,
        "gel_facts": total_facts,
        "gel_concepts": total_concepts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run GEL for LoCoMo samples.")
    parser.add_argument("--sample", type=str, required=True, nargs="+",
                        help="Sample ID(s) or 'all' for all 10 samples in parallel")
    parser.add_argument("--ppr", type=float, default=PPR_WEIGHT, help="PPR weight")
    args = parser.parse_args()
    ppr_w = args.ppr

    samples = SAMPLE_IDS if args.sample == ["all"] else args.sample

    print("=" * 70)
    print(f"GAAMA GEL -- {len(samples)} sample(s)")
    print(f"  Samples: {', '.join(samples)}")
    print(f"  PPR={ppr_w}, Budget={PPR_BUDGET.max_facts}F/{PPR_BUDGET.max_reflections}R/"
          f"{PPR_BUDGET.max_skills}S/{PPR_BUDGET.max_episodes}E")
    print(f"  GEL: belief={GEL_FACT_BELIEF}, max_facts={GEL_MAX_FACTS_PER_QUERY}, "
          f"threshold={GEL_REWARD_THRESHOLD}")
    print(f"  Per-sample DB copies under {GEL_ROOT}")
    print("=" * 70)

    t0 = time.time()
    work = [(sid, ppr_w) for sid in samples]

    if len(samples) == 1:
        summaries = [_run_sample_gel(work[0])]
    else:
        with multiprocessing.Pool(processes=len(samples)) as pool:
            async_results = {
                sid: pool.apply_async(_run_sample_gel, (w,))
                for sid, w in zip(samples, work)
            }
            summaries = []
            for sid in samples:
                try:
                    s = async_results[sid].get()
                    summaries.append(s)
                except Exception as e:
                    print(f"  [FAIL] {sid}: {e}")

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*70}")
    print(f"GEL SUMMARY -- {len(summaries)} sample(s) ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"{'Sample':<12} {'N':>5} {'Baseline':>10} {'Final':>10} {'Delta':>8} "
          f"{'Trig':>5} {'Facts':>6} {'Concepts':>9}")
    print("-" * 72)
    for s in sorted(summaries, key=lambda x: x["sample_id"]):
        print(f"  {s['sample_id']:<10} {s['n']:>5} {s['baseline_reward']:>9.4f} "
              f"{s['final_reward']:>9.4f} {s['delta']:>+7.4f} "
              f"{s['gel_triggered']:>5} {s['gel_facts']:>6} {s['gel_concepts']:>9}")

    if summaries:
        total_n = sum(s["n"] for s in summaries)
        b_total = sum(s["baseline_reward"] * s["n"] for s in summaries) / total_n
        f_total = sum(s["final_reward"] * s["n"] for s in summaries) / total_n
        tf = sum(s["gel_facts"] for s in summaries)
        tc = sum(s["gel_concepts"] for s in summaries)
        print("-" * 72)
        print(f"  {'OVERALL':<10} {total_n:>5} {b_total:>9.4f} {f_total:>9.4f} "
              f"{f_total-b_total:>+7.4f} {'':>5} {tf:>6} {tc:>9}")
    print("=" * 70)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
