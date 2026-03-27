"""
Default configurations for LoCoMo evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Package path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_GAAMA_PKG = _FILE_DIR.parent.parent  # gaama/evals/locomo -> gaama/evals -> gaama/
_REPO_ROOT = _GAAMA_PKG.parent  # gaama/ -> agentic-memory/ (so 'import gaama' works)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gaama.core import RetrievalBudget  # noqa: E402

# ---------------------------------------------------------------------------
# Semantic retriever defaults (large KNN pool)
# ---------------------------------------------------------------------------
SEMANTIC_BUDGET = RetrievalBudget(
    max_facts=60,
    max_reflections=20,
    max_skills=5,
    max_episodes=80,
)
SEMANTIC_PPR_WEIGHT = 0.0
SEMANTIC_SIM_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# PPR retriever defaults
# ---------------------------------------------------------------------------
PPR_BUDGET = RetrievalBudget(
    max_facts=60,
    max_reflections=20,
    max_skills=5,
    max_episodes=80,
)
PPR_WEIGHT = 0.1
PPR_SIM_WEIGHT = 1.0
PPR_EXPANSION_DEPTH = 2

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------
MAX_MEMORY_WORDS = 1000
LLM_MODEL = "gpt-4o-mini"
EVAL_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_CHUNK = 500
MAX_WORKERS = 20

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_PATH = _FILE_DIR / "locomo10.json"
SAMPLE_IDS = [
    "conv-26", "conv-30", "conv-41", "conv-42", "conv-43",
    "conv-44", "conv-47", "conv-48", "conv-49", "conv-50",
]
CATEGORY_NAMES = {
    1: "cat1_factual",
    2: "cat2_temporal",
    3: "cat3_inference",
    4: "cat4_detailed",
}
