"""Run neural PPR training pipeline on conv-26.

Steps:
  1. Generate training data: evaluate all questions at ppr_w in {0.01, 0.5, 1.0}
  2. Summarise the CSV
  3. Train neural model (30% train, 70% test)
  4. Evaluate with the trained model via retrieve_and_evaluate
"""
from __future__ import annotations

import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_GAAMA_ROOT = _FILE_DIR.parent.parent          # gaama/
_PROJECT_ROOT = _GAAMA_ROOT.parent              # parent of gaama/ (for `from gaama.xxx` imports)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaama.core import RetrievalBudget

import locomo_pipeline as lp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = _FILE_DIR / "locomo10.json"
DATA_DIR = _FILE_DIR / "data"
OUTPUT_DIR = _FILE_DIR / "data" / "neural_ppr"
DB_NAME = "locomo_memory_with_bl.sqlite"
SAMPLE_IDS = None  # All samples
BUDGET = RetrievalBudget(max_facts=40, max_reflections=20, max_skills=5, max_episodes=80)

# Hand-tuned edge weights (best from earlier experiments)
EDGE_TYPE_WEIGHTS = {
    "SUBJECT":       0.6 ,
    "OBJECT":        0.6,
    "INVOLVES":      0.9,
    "MENTIONS":      0.7,
    "TRIGGERED_BY":  0.7,
    "ABOUT":         0.7,
    "SUPPORTED_BY":  0.6,
    "PRODUCED":      0.6,
    "DERIVED_FROM":  0.6,
    "LEARNED_FROM":  0.6,
    "USES_TOOL":     0.6,
    "APPLIES_TO":    0.6,
    "RELATED_TO":    0.5,
    "CONTRADICTS":   0.5,
    "REFINES":       0.5,
    "NEXT":          0.5
}


def step1_generate_data():
    """Run evaluations at each ppr_weight and generate CSV + embeddings."""
    print("\n" + "=" * 70)
    print("STEP 1: Generate PPR training data")
    print("=" * 70)

    csv_path = lp.generate_ppr_training_data(
        data_path=DATA_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        sample_ids=SAMPLE_IDS,
        categories=[1, 2, 3, 4],
        ppr_weights=[0.01, 0.5, 1.0],
        budget=BUDGET,
        llm_model="gpt-4o-mini",
        eval_model="gpt-4o-mini",
        db_name=DB_NAME,
        max_workers=20,
        max_memory_words=600,
        edge_type_weights=EDGE_TYPE_WEIGHTS,
    )
    return csv_path


def step2_summarize():
    """Print summary of generated data."""
    print("\n" + "=" * 70)
    print("STEP 2: Summarise training data")
    print("=" * 70)
    csv_path = OUTPUT_DIR / "ppr_training_data.csv"
    return lp.summarize_ppr_eval(csv_path)


def step3_train():
    """Train neural PPR model on 70% of data."""
    print("\n" + "=" * 70)
    print("STEP 3: Train neural PPR model (70% train)")
    print("=" * 70)

    model, split_info = lp.train_neural_ppr(
        csv_path=OUTPUT_DIR / "ppr_training_data.csv",
        embeddings_path=OUTPUT_DIR / "ppr_training_embeddings.json",
        model_save_path=OUTPUT_DIR / "neural_ppr_model.json",
        train_ratio=0.4,
        hidden_dim=32,
        epochs=400,
        lr=0.0005,
        batch_size=64,
        seed=42,
    )
    return model, split_info


def step4_evaluate(model=None, split_info=None):
    """Evaluate on the remaining 30% test set (from training split) and full dataset."""
    print("\n" + "=" * 70)
    print("STEP 4: Evaluate neural PPR model")
    print("=" * 70)

    if model is None:
        from gaama.services.neural_ppr import NeuralPPRModel
        model = NeuralPPRModel.load(OUTPUT_DIR / "neural_ppr_model.json")

    # Get test questions from the same split used during training
    if split_info is None:
        # Recreate the same split (seed=42, train_ratio=0.7)
        import csv as _csv
        import json as _json
        from collections import defaultdict as _dd
        _rows = list(_csv.DictReader(open(OUTPUT_DIR / "ppr_training_data.csv", encoding="utf-8")))
        _by_q = _dd(list)
        for r in _rows:
            _by_q[r["question"]].append(r)
        _questions = sorted(_by_q.keys())
        import random as _rng
        _r = _rng.Random(42)
        _r.shuffle(_questions)
        _k = max(1, int(len(_questions) * 0.7))
        test_questions = set(_questions[_k:])
        by_question = dict(_by_q)
    else:
        test_questions = split_info["test_questions"]
        by_question = split_info["by_question"]

    # Build test_qa_override: {sample_id: [qa_dicts]} from the 30% test questions
    from collections import defaultdict
    test_qa_30 = defaultdict(list)
    for q in test_questions:
        rows_for_q = by_question.get(q, [])
        if rows_for_q:
            row = rows_for_q[0]
            test_qa_30[row["sample_id"]].append({
                "question": q,
                "answer": row.get("answer", ""),
                "category": int(row["category"]),
            })
    test_qa_30 = dict(test_qa_30)

    progress_path = _FILE_DIR / "retrieve_evaluate_progress.json"

    # --- Evaluate on 30% test set (remaining from training split) ---
    print(f"\n--- Neural PPR on 30% test set ({sum(len(v) for v in test_qa_30.values())} questions) ---")
    if progress_path.exists():
        progress_path.unlink()

    results_test = lp.retrieve_and_evaluate(
        data_path=DATA_PATH,
        data_dir=DATA_DIR,
        budget=BUDGET,
        llm_model="gpt-4o-mini",
        sample_ids=SAMPLE_IDS,
        db_name=DB_NAME,
        ppr_score_weight=1.0,
        sim_score_weight=1.0,
        adaptive_ppr_model=model,
        evaluator="hypothesis",
        eval_model="gpt-4o-mini",
        max_workers=20,
        max_memory_words=600,
        edge_type_weights=EDGE_TYPE_WEIGHTS,
        test_qa_override=test_qa_30,
        output_path=OUTPUT_DIR / "neural_ppr_test30.jsonl",
    )
    print("\n=== Results on 30% test set ===")
    lp.compute_reward_summary(results_test)

    # --- Evaluate on full dataset ---
    print("\n--- Neural PPR on full dataset ---")
    if progress_path.exists():
        progress_path.unlink()

    results_full = lp.retrieve_and_evaluate(
        data_path=DATA_PATH,
        data_dir=DATA_DIR,
        budget=BUDGET,
        llm_model="gpt-4o-mini",
        sample_ids=SAMPLE_IDS,
        db_name=DB_NAME,
        ppr_score_weight=1.0,
        sim_score_weight=1.0,
        adaptive_ppr_model=model,
        evaluator="hypothesis",
        eval_model="gpt-4o-mini",
        max_workers=20,
        max_memory_words=600,
        edge_type_weights=EDGE_TYPE_WEIGHTS,
        output_path=OUTPUT_DIR / "neural_ppr_full.jsonl",
    )
    print("\n=== Results on full dataset ===")
    lp.compute_reward_summary(results_full)

    return results_test, results_full


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neural PPR training pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run specific step (default: all)")
    args = parser.parse_args()

    model = None
    split_info = None

    if args.step is None or args.step == 1:
        step1_generate_data()
    if args.step is None or args.step == 2:
        step2_summarize()
    if args.step is None or args.step == 3:
        model, split_info = step3_train()
    if args.step is None or args.step == 4:
        step4_evaluate(model, split_info)
