"""
PPR retrieval evaluation for LoCoMo.
=====================================
Runs PPR-based retrieval with graph expansion, then evaluates with fractional
hypothesis judge. Samples run sequentially, questions in parallel.

Usage:
    python run_ppr_eval.py                                   # all samples
    python run_ppr_eval.py --sample-ids conv-26 conv-30      # specific
    python run_ppr_eval.py --ppr-weight 0.5                  # custom PPR weight
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# SDK path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent  # gaama/evals/locomo -> gaama/evals -> gaama -> agentic-memory
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_FILE_DIR))

from gaama.api import create_default_sdk  # noqa: E402
from gaama.config.settings import (  # noqa: E402
    SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
)

from config import (  # noqa: E402
    SAMPLE_IDS, PPR_BUDGET, PPR_WEIGHT, PPR_SIM_WEIGHT, PPR_EXPANSION_DEPTH,
    MAX_MEMORY_WORDS, LLM_MODEL, EVAL_MODEL, EMBEDDING_MODEL, MAX_WORKERS,
    CATEGORY_NAMES,
)
from locomo_eval import (  # noqa: E402
    evaluate_sample, compute_summary, save_results_jsonl,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
DATA_PATH = _FILE_DIR / "locomo10.json"
DATA_DIR = _FILE_DIR / "data" / "ltm"
OUTPUT_DIR = _FILE_DIR / "data" / "results"
DB_NAME = "locomo_memory.sqlite"


# ---------------------------------------------------------------------------
# SDK factory
# ---------------------------------------------------------------------------

def _make_sdk(data_dir: Path, llm_model: str = LLM_MODEL):
    """Create a GAAMA SDK instance."""
    settings = SDKSettings(
        storage=StorageSettings(
            sqlite_path=data_dir / DB_NAME,
            blob_root=data_dir / "blobs",
        ),
        llm=LLMSettings(
            provider="openai",
            model=llm_model,
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPR retrieval evaluation for LoCoMo.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Specific sample IDs to evaluate (default: all).",
    )
    parser.add_argument(
        "--ppr-weight",
        type=float,
        default=PPR_WEIGHT,
        help=f"PPR score weight (default: {PPR_WEIGHT}).",
    )
    parser.add_argument(
        "--sim-weight",
        type=float,
        default=PPR_SIM_WEIGHT,
        help=f"Similarity score weight (default: {PPR_SIM_WEIGHT}).",
    )
    parser.add_argument(
        "--expansion-depth",
        type=int,
        default=PPR_EXPANSION_DEPTH,
        help=f"PPR expansion depth (default: {PPR_EXPANSION_DEPTH}).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=MAX_MEMORY_WORDS,
        help=f"Max words for retrieved memory (default: {MAX_MEMORY_WORDS}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Max parallel workers for question evaluation (default: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=EVAL_MODEL,
        help=f"LLM model for evaluation (default: {EVAL_MODEL}).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Override LTM data directory (default: {DATA_DIR}).",
    )
    args = parser.parse_args()

    sample_ids = args.sample_ids if args.sample_ids else SAMPLE_IDS
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    print("=" * 70, flush=True)
    print("PPR Retrieval Evaluation", flush=True)
    print(f"Samples: {sample_ids}", flush=True)
    print(f"Budget: {PPR_BUDGET}", flush=True)
    print(f"ppr_weight: {args.ppr_weight}, sim_weight: {args.sim_weight}, "
          f"expansion_depth: {args.expansion_depth}", flush=True)
    print(f"max_words: {args.max_words}", flush=True)
    print(f"semantic_only: False", flush=True)
    print(f"Data dir: {data_dir}", flush=True)
    print("=" * 70, flush=True)

    sdk = _make_sdk(data_dir)
    t0 = time.time()
    all_results = []

    for sid in sample_ids:
        try:
            results = evaluate_sample(
                sdk=sdk,
                data_path=DATA_PATH,
                sample_id=sid,
                categories=list(CATEGORY_NAMES.keys()),
                budget=PPR_BUDGET,
                max_memory_words=args.max_words,
                ppr_weight=args.ppr_weight,
                sim_weight=args.sim_weight,
                semantic_only=False,
                expansion_depth=args.expansion_depth,
                eval_model=args.eval_model,
                max_workers=args.max_workers,
            )
            # Filter to only this sample (safety)
            results = [r for r in results if r.get("sample_id") == sid]

            # Save per-sample JSONL
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"{sid}_ppr.jsonl"
            save_results_jsonl(results, out_path)

            all_results.extend(results)
        except Exception as e:
            print(f"  [{sid}] ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.1f}s ({len(all_results)} questions total)", flush=True)

    # Print final summary
    compute_summary(all_results)


if __name__ == "__main__":
    main()
