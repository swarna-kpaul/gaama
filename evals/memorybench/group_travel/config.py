"""
Default configurations for MemoryBench Group Travel Planner evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent.parent  # group_travel -> memorybench -> evals -> gaama -> agentic-memory
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gaama.core import RetrievalBudget  # noqa: E402

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
AGENT_ID = "agent-memorybench-gtp"
USER_ID = "memorybench-gtp"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_PATH = _FILE_DIR / "data" / "group_travel_planner.jsonl"

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DATA_DIR = _FILE_DIR / "data" / "ltm"
DB_NAME = "memorybench_gtp.sqlite"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
BUDGET = RetrievalBudget(
    max_facts=40,
    max_reflections=5,
    max_skills=0,
    max_episodes=60,
)
PPR_WEIGHT = 0.1
SIM_WEIGHT = 1.0
MAX_MEMORY_WORDS = 1000

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
EVAL_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_WORKERS = 10
