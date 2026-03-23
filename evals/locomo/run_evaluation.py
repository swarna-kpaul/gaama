"""
Runner script: Full LoCoMo evaluation.
"""
import sys
from pathlib import Path

# Setup paths
_FILE_DIR = Path(__file__).resolve().parent
_GAAMA_ROOT = _FILE_DIR.parent.parent          # gaama/
_PROJECT_ROOT = _GAAMA_ROOT.parent              # parent of gaama/ (for `from gaama.xxx` imports)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from gaama.core import RetrievalBudget
from locomo_pipeline import (
    create_ltm_from_sessions,
    retrieve_and_evaluate,
    compute_reward_summary,
)

# Paths
DATA_PATH = _FILE_DIR / "locomo10.json"
DATA_DIR = _FILE_DIR / "data"
OUTPUT_DIR = _FILE_DIR

# Run evaluation (all 10 conversations)
print("Step 1: Create LTM from sessions...")
sdk = create_ltm_from_sessions(
    data_path=DATA_PATH,
    data_dir=DATA_DIR,
    llm_model="gpt-4o-mini",
    max_tokens_per_chunk=2048,
)

print("\nStep 2: Retrieve and evaluate...")
results = retrieve_and_evaluate(
    data_path=DATA_PATH,
    data_dir=DATA_DIR,
    budget=RetrievalBudget(max_facts=40, max_reflections=20, max_skills=5, max_episodes=80),
    llm_model="gpt-4o-mini",
    evaluator="hypothesis",
    eval_model="gpt-4o-mini",
    max_workers=20,
    max_memory_words=600,
    output_path=OUTPUT_DIR / "evaluation_results.jsonl",
)

print("\nStep 3: Compute reward summary...")
compute_reward_summary(results)

print("\nDone! Check output files in:", OUTPUT_DIR)
