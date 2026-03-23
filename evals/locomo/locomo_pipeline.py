"""
LoCoMo SDK Pipeline
====================
Main functions:
  1. create_ltm_from_sessions      – Build LTM from conversation sessions for each sample_id.
  2. retrieve_and_evaluate          – Retrieve memories & evaluate all questions per sample.
  3. generate_ppr_training_data    – Generate training data for neural PPR model.
  4. train_neural_ppr              – Train neural PPR reward predictor MLP.

Each sample_id is treated as a separate task_id.  Within each sample there are
many questions (100-260) across 5 categories.  Existing LTM is NOT cleared
before loading new data (additive).

LoCoMo data shape (per entry):
  sample_id          – "conv-26", "conv-30", …
  conversation       – {speaker_a, speaker_b, session_N, session_N_date_time, …}
  qa                 – [{question, answer, evidence, category, adversarial_answer?}, …]
  session_summary    – {session_N_summary: str, …}
  event_summary      – {events_session_N: …}
  observation        – {session_N_observation: …}

Categories:
  1 – Factual recall
  2 – Temporal / date questions
  3 – Inference / reasoning
  4 – Detailed understanding
  5 – Adversarial (has adversarial_answer; usually no answer field)
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Repo / GAAMA path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_GAAMA_ROOT = _FILE_DIR.parent.parent          # gaama/
_PROJECT_ROOT = _GAAMA_ROOT.parent              # parent of gaama/ (for `from gaama.xxx` imports)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaama.api import create_default_sdk, AgenticMemorySDK
from gaama.config.settings import (
    SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
)
from gaama.core import TraceEvent, QueryFilters, RetrievalBudget, MemoryPack
from gaama.services.neural_ppr import NeuralPPRModel, NeuralPPRTrainer
from gaama.services.interfaces import CreateOptions
from gaama.services import answer_from_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent logging & progress helpers
# ---------------------------------------------------------------------------
_LOG_DIR = _FILE_DIR  # log files live next to this script


def _setup_file_logger(log_name: str) -> logging.Logger:
    """Create/reuse a logger that writes to a persistent log file."""
    log_path = _LOG_DIR / f"{log_name}.log"
    flogger = logging.getLogger(f"locomo.{log_name}")
    if not flogger.handlers:
        flogger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        flogger.addHandler(fh)
    return flogger


def _load_progress(progress_path: Path) -> Dict[str, Any]:
    """Load progress state from a JSON file. Returns empty dict if missing."""
    if progress_path.exists():
        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_progress(progress_path: Path, state: Dict[str, Any]) -> None:
    """Atomically save progress state to a JSON file."""
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(progress_path)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
USER_ID = "locomo"
DEFAULT_BUDGET = RetrievalBudget(max_facts=20, max_reflections=10, max_skills=5, max_episodes=40)

CATEGORY_NAMES = {
    1: "cat1_factual",
    2: "cat2_temporal",
    3: "cat3_inference",
    4: "cat4_detailed",
    5: "cat5_adversarial",
}


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def split_train_test_qa(
    data_path: Path,
    train_ratio: float = 0.5,
    categories: Optional[List[int]] = None,
    seed: Optional[int] = 42,
    sample_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """Split original Q&A into training and test sets, **per category**.

    Parameters
    ----------
    data_path      : Path to locomo10.json
    train_ratio    : Fraction of Q&A per category to use for training (rest → test)
    categories     : Categories to include (default: [1, 2, 3, 4])
    seed           : Random seed for reproducible splits (None = random)
    sample_ids     : Only include these sample IDs (None = all)
    limit          : Max number of samples to include (None = all)

    Returns
    -------
    split : Dict keyed by sample_id, each value is::

        {
            "train": {1: [qa_dict, ...], 2: [...], 3: [...], 4: [...]},
            "test":  {1: [qa_dict, ...], 2: [...], 3: [...], 4: [...]},
            "train_flat": [qa_dict, ...],   # all train Q&A concatenated
            "test_flat":  [qa_dict, ...],   # all test Q&A concatenated
        }
    """
    if categories is None:
        categories = [1, 2, 3, 4]
    rng = random.Random(seed)

    dataset = _load_dataset(data_path)
    if sample_ids is not None:
        sid_set = set(sample_ids)
        dataset = [item for item in dataset if item.get("sample_id") in sid_set]
    if limit is not None:
        dataset = dataset[:limit]

    result: Dict[str, Dict[str, Any]] = {}
    for item in dataset:
        sid = item.get("sample_id", "unknown")
        train_by_cat: Dict[int, List[dict]] = {c: [] for c in categories}
        test_by_cat: Dict[int, List[dict]] = {c: [] for c in categories}

        for cat in categories:
            cat_questions = [q for q in item.get("qa", [])
                            if q.get("category") == cat and q.get("answer")]
            rng.shuffle(cat_questions)
            k = max(1, int(len(cat_questions) * train_ratio))
            train_by_cat[cat] = cat_questions[:k]
            test_by_cat[cat] = cat_questions[k:]

        train_flat = [q for cat in categories for q in train_by_cat[cat]]
        test_flat = [q for cat in categories for q in test_by_cat[cat]]

        result[sid] = {
            "train": train_by_cat,
            "test": test_by_cat,
            "train_flat": train_flat,
            "test_flat": test_flat,
        }

        # Print summary
        print(f"  {sid}: ", end="")
        for cat in categories:
            cat_name = CATEGORY_NAMES.get(cat, f"cat{cat}")
            tr = len(train_by_cat[cat])
            te = len(test_by_cat[cat])
            print(f"{cat_name}={tr}tr/{te}te  ", end="")
        print(f"  total={len(train_flat)}tr/{len(test_flat)}te")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(data_path: Path) -> List[dict]:
    """Load locomo10.json."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return data


def _parse_session_date(date_str: str) -> Optional[datetime]:
    """Best-effort parse of LoCoMo date strings like '1:56 pm on 8 May, 2023'."""
    if not date_str:
        return None
    # Normalise
    s = date_str.strip().replace(",", "").replace(".", "")
    patterns = [
        r"(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        r"(\d{1,2}:\d{2}\s*[ap]m)\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            time_s, day, month_s, year = m.groups()
            try:
                return datetime.strptime(f"{day} {month_s} {year} {time_s}", "%d %B %Y %I:%M %p")
            except ValueError:
                pass
    return None


def _get_sorted_sessions(conversation: dict) -> List[Tuple[int, list, Optional[str]]]:
    """Return list of (session_idx, turns, date_string) sorted by session number."""
    sessions = []
    for key in conversation:
        m = re.match(r"^session_(\d+)$", key)
        if not m:
            continue
        idx = int(m.group(1))
        turns = conversation[key]
        date_key = f"session_{idx}_date_time"
        date_str = conversation.get(date_key) or conversation.get(f"session_{idx}_date") or ""
        sessions.append((idx, turns, date_str))
    sessions.sort(key=lambda x: x[0])
    return sessions


def _session_to_events(
    turns: list,
    session_idx: int,
    session_date: Optional[str],
    base_ts: datetime,
) -> List[TraceEvent]:
    """Convert one LoCoMo session (list of speaker/text dicts) to TraceEvents."""
    events: List[TraceEvent] = []
    for turn_idx, turn in enumerate(turns):
        speaker = turn.get("speaker", "user")
        text = turn.get("text", "")
        if not text:
            continue
        ts = base_ts + timedelta(seconds=len(events) * 30)
        metadata: Dict[str, Any] = {
            "session_date": session_date or "",
        }
        # Per-turn LoCoMo attributes (images, captions)
        blip_caption = turn.get("blip_caption")
        if blip_caption:
            metadata["blip_caption"] = blip_caption
        img_url = turn.get("img_url")
        if img_url:
            metadata["img_url"] = img_url
        query = turn.get("query")
        if query:
            metadata["query"] = query
        events.append(
            TraceEvent(
                event_id=f"s{session_idx}_t{turn_idx}_{len(events)}",
                event_type="message",
                actor=speaker,
                content=text,
                ts=ts,
                metadata=metadata,
            )
        )
    return events


def _make_sdk(data_dir: Path, llm_model: str = "gpt-4o-mini",
              db_name: str = "locomo_memory.sqlite") -> AgenticMemorySDK:
    """Create an SDK instance with OpenAI LLM + embedding.

    Parameters
    ----------
    data_dir : Root directory for storage
    llm_model : OpenAI model name
    db_name  : SQLite database filename (allows separate DBs for different runs)
    """
    settings = SDKSettings(
        storage=StorageSettings(
            sqlite_path=data_dir / db_name,
            blob_root=data_dir / "blobs",
        ),
        llm=LLMSettings(
            provider="openai",
            model=llm_model,
            api_key_env="OPENAI_API_KEY",
            max_tokens=16000,
        ),
        embedding=EmbeddingSettings(
            model="text-embedding-3-small",
            api_key_env="OPENAI_API_KEY",
        ),
    )
    return create_default_sdk(settings, agent_id=AGENT_ID)


def _get_llm(sdk: AgenticMemorySDK):
    orch = sdk._orchestrator
    llm = getattr(orch._extractor, "_llm", None)
    if llm is None:
        raise RuntimeError("No LLM adapter on extractor. Ensure OPENAI_API_KEY is set.")
    return llm


# ============================================================================
# 1. CREATE LTM FROM CONVERSATION SESSIONS
# ============================================================================

def create_ltm_from_sessions(
    data_path: Path,
    data_dir: Path,
    limit: Optional[int] = None,
    max_tokens_per_chunk: int = 2048,
    llm_model: str = "gpt-4o-mini",
    sample_ids: Optional[List[str]] = None,
    db_name: str = "locomo_memory.sqlite",
    max_workers: int = 10,
) -> AgenticMemorySDK:
    """
    Build LTM from conversation sessions for every sample_id.

    Each sample_id is treated as a separate task_id.  Existing LTM is NOT
    cleared before loading (additive).  Tasks are processed in parallel
    (up to *max_workers* threads), each with its own SDK instance to avoid
    STM state conflicts.  All instances share the same SQLite database.

    Parameters
    ----------
    data_path  : Path to locomo10.json
    data_dir   : Directory for SQLite + blobs (single shared DB)
    limit      : Max number of samples to process (None = all)
    max_tokens_per_chunk : Token limit per chunk for LTM creation
    llm_model  : OpenAI model for extraction
    sample_ids : If provided, only process these sample IDs
    db_name    : SQLite database filename
    max_workers: Max parallel threads for LTM creation (default 10)

    Returns
    -------
    sdk : The SDK instance (reusable for retrieval / belief learning)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(data_path)
    if sample_ids is not None:
        sid_set = set(sample_ids)
        dataset = [item for item in dataset if item.get("sample_id") in sid_set]
    if limit is not None:
        dataset = dataset[:limit]

    fallback_base = datetime(2023, 5, 1, 12, 0, 0)

    # --- Progress & logging setup ---
    flog = _setup_file_logger("ltm_creation")
    progress_path = _FILE_DIR / "ltm_creation_progress.json"
    progress = _load_progress(progress_path)
    completed: Set[str] = set(progress.get("completed_ids", []))
    if completed:
        flog.info("Resuming LTM creation – %d already completed", len(completed))
        print(f"Resuming: {len(completed)} sample(s) already completed, skipping them.")

    total = len(dataset)

    # Filter out already-completed items
    pending: List[Tuple[int, Dict[str, Any]]] = []
    for idx, item in enumerate(dataset):
        sid = item.get("sample_id", f"conv-{idx}")
        if sid in completed:
            flog.info("[%d/%d] SKIP (already done) sample_id=%s", idx + 1, total, sid)
            print(f"[{idx + 1}/{total}] SKIP sample_id={sid} (already done)")
        else:
            pending.append((idx, item))

    # Thread-safe progress state
    progress_lock = threading.Lock()

    def _process_one_sample(idx: int, item: Dict[str, Any]) -> str:
        """Process a single sample. Each thread gets its own SDK instance."""
        sid = item.get("sample_id", f"conv-{idx}")
        # Each thread creates its own SDK to keep STM buffers isolated
        worker_sdk = _make_sdk(data_dir, llm_model=llm_model, db_name=db_name)

        conversation = item.get("conversation", {})
        sessions = _get_sorted_sessions(conversation)

        create_opts = CreateOptions(
            user_id=USER_ID,
            task_id=sid,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )

        for sess_idx, turns, date_str in sessions:
            parsed_dt = _parse_session_date(date_str)
            base_ts = parsed_dt if parsed_dt else fallback_base + timedelta(hours=sess_idx)
            events = _session_to_events(turns, sess_idx, date_str, base_ts)
            if not events:
                continue
            worker_sdk.ingest(events)
        worker_sdk.create(create_opts)

        # Update progress atomically
        with progress_lock:
            completed.add(sid)
            progress["completed_ids"] = sorted(completed)
            progress["last_updated"] = datetime.now().isoformat()
            _save_progress(progress_path, progress)

        msg = (f"[{idx + 1}/{total}] Created LTM for sample_id={sid} "
               f"({len(sessions)} sessions, {len(item.get('qa', []))} questions)")
        flog.info(msg)
        print(msg)
        return sid

    # Run in parallel
    if pending:
        workers = min(max_workers, len(pending))
        flog.info("Starting parallel LTM creation: %d pending, %d workers", len(pending), workers)
        print(f"\nProcessing {len(pending)} sample(s) with {workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one_sample, idx, item): item.get("sample_id", f"conv-{idx}")
                for idx, item in pending
            }
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    fut.result()
                except Exception:
                    flog.error("Failed LTM creation for sample_id=%s", sid, exc_info=True)
                    print(f"ERROR: LTM creation failed for sample_id={sid} (see log)")

    flog.info("Done. Processed %d samples into LTM at %s", total, data_dir)
    print(f"\nDone. Processed {total} samples into LTM at {data_dir}")

    # Return a shared SDK instance for subsequent pipeline stages
    sdk = _make_sdk(data_dir, llm_model=llm_model, db_name=db_name)
    return sdk


# ============================================================================
# 2. RETRIEVE & EVALUATE
# ============================================================================

def _trim_memory_pack(
    pack: MemoryPack,
    max_words: int,
    budget: Optional[RetrievalBudget] = None,
) -> MemoryPack:
    """Return a new MemoryPack trimmed so that to_text() stays under *max_words*.

    Removal is distributed **proportionally** across categories according to
    their budget caps so that no single category is drained before others.

    For facts, reflections, and skills the last item is the lowest-ranked
    (LTM fills buckets in descending score order).  For episodes — which are
    reordered chronologically — the item with the **lowest relevance score**
    is removed instead, preserving temporal order of the survivors.

    Scores are read from ``pack.scores`` (populated by LTM retrieval).
    If scores are unavailable the function falls back to removing the last item.
    If *budget* is None the four categories are treated as equal weight.
    """
    categories = ["facts", "reflections", "skills", "episodes"]

    # Mutable copy of item lists
    items: Dict[str, List[str]] = {
        cat: list(getattr(pack, cat)) for cat in categories
    }

    def _word_count() -> int:
        return sum(len(s.split()) for cat in categories for s in items[cat])

    if _word_count() <= max_words:
        return pack

    # Budget weights (proportional share per category)
    budget_map = {
        "facts": budget.max_facts if budget else 1,
        "reflections": budget.max_reflections if budget else 1,
        "skills": budget.max_skills if budget else 1,
        "episodes": budget.max_episodes if budget else 1,
    }
    total_budget = sum(budget_map.values()) or 1

    # Parallel score lists (used for score-aware removal of episodes)
    score_lists: Dict[str, List[float]] = {}
    for cat in categories:
        if pack.scores and cat in pack.scores:
            score_lists[cat] = list(pack.scores[cat])
        else:
            score_lists[cat] = []

    # Compute per-item word counts (cache for efficiency)
    word_counts: Dict[str, List[int]] = {
        cat: [len(s.split()) for s in items[cat]] for cat in categories
    }

    # Iteratively remove items until under budget.
    # Each round, pick the category that is most over-represented relative
    # to its budget share and remove its lowest-ranked item.
    while _word_count() > max_words:
        # Find which categories still have items
        non_empty = [cat for cat in categories if items[cat]]
        if not non_empty:
            break

        # For each non-empty category compute how over-represented it is:
        #   actual_share / target_share  (higher = more over-represented)
        total_words = _word_count() or 1
        worst_cat = None
        worst_ratio = -1.0
        for cat in non_empty:
            cat_words = sum(word_counts[cat])
            target_share = budget_map[cat] / total_budget
            actual_share = cat_words / total_words
            ratio = actual_share / target_share if target_share > 0 else float("inf")
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_cat = cat

        if worst_cat is None:
            break

        # For episodes: remove the item with the lowest relevance score
        # (preserves chronological order of survivors).
        # For other categories: remove the last item (already lowest-ranked).
        if worst_cat == "episodes" and score_lists.get("episodes"):
            min_idx = min(
                range(len(score_lists["episodes"])),
                key=lambda i: score_lists["episodes"][i],
            )
            items["episodes"].pop(min_idx)
            word_counts["episodes"].pop(min_idx)
            score_lists["episodes"].pop(min_idx)
        else:
            items[worst_cat].pop()
            word_counts[worst_cat].pop()
            if score_lists.get(worst_cat):
                score_lists[worst_cat].pop()

    return MemoryPack(
        facts=items["facts"],
        reflections=items["reflections"],
        skills=items["skills"],
        episodes=items["episodes"],
        citations=list(pack.citations),
        scores={cat: score_lists[cat] for cat in categories} if any(score_lists.values()) else None,
    )


def retrieve_and_evaluate(
    data_path: Path,
    data_dir: Path,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    budget: Optional[RetrievalBudget] = None,
    llm_model: str = "gpt-4o-mini",
    sample_ids: Optional[List[str]] = None,
    sdk: Optional[AgenticMemorySDK] = None,
    use_openai_judge: bool = False,  # kept for API compat; ignored (always uses LLM judge on retrieved memory)
    max_questions_per_sample: Optional[int] = None,
    categories: Optional[List[int]] = None,
    db_name: str = "locomo_memory.sqlite",
    ppr_score_weight: float = 1.0,
    sim_score_weight: float = 1.0,
    degree_correction: bool = False,
    expansion_depth: int | None = None,
    ppr_weight_by_category: dict[int, float] | None = None,
    edge_type_weights: dict[str, float] | None = None,
    adaptive_ppr_model: object | None = None,
    evaluator: str = "hypothesis",
    eval_model: str = "gpt-4o-mini",
    max_memory_words: int = 800,
    max_workers: int = 20,
    test_qa_override: Optional[Dict[str, List[dict]]] = None,
) -> List[Dict[str, Any]]:
    """
    For each sample and each question within it:
      1. Retrieve relevant memories from LTM (scoped to sample's task_id).
      2. Generate an answer from memory (hypothesis).
      3. Compute reward by comparing with ground truth.
      4. Store results in a file buffer.

    Questions are evaluated in parallel (up to *max_workers* threads).

    Parameters
    ----------
    data_path   : Path to locomo10.json
    data_dir    : Directory where SDK data lives
    output_path : If set, write results as JSONL
    limit       : Max samples to evaluate
    budget      : Retrieval budget
    llm_model   : Model for answer generation and reward
    sample_ids  : If provided, only evaluate these sample IDs
    sdk         : Reuse an existing SDK instance
    use_openai_judge : If True, use LLM judge; else rule-based
    max_questions_per_sample : Cap questions evaluated per sample (None = all)
    categories  : If provided, only evaluate these categories (default: [1,2,3,4])
    ppr_score_weight : Weight for PPR score in retrieval ranking (default: 1.0)
    sim_score_weight : Weight for similarity score in retrieval ranking (default: 1.0)
    evaluator   : "retrieval" = judge retrieved memory directly,
                  "hypothesis" = generate answer from memory then judge the answer
    eval_model  : LLM model used for evaluation (default: gpt-4o-mini)
    max_memory_words : Trim retrieved memory to at most this many words (-1 = no trim)
    max_workers : Max parallel threads for evaluation (default: 20)
    test_qa_override : If provided, a dict mapping sample_id → list of QA dicts
                  to evaluate instead of loading from the dataset. When set,
                  categories and max_questions_per_sample filters are not applied
                  (the caller is responsible for pre-filtering).

    Returns
    -------
    results : List of dicts per question
    """
    # Default to cat1-cat4 (exclude cat5 adversarial)
    if categories is None:
        categories = [1, 2, 3, 4]
    if budget is None:
        budget = DEFAULT_BUDGET

    dataset = _load_dataset(data_path)
    if sample_ids is not None:
        sid_set = set(sample_ids)
        dataset = [item for item in dataset if item.get("sample_id") in sid_set]
    if limit is not None:
        dataset = dataset[:limit]

    if sdk is None:
        sdk = _make_sdk(data_dir, llm_model=llm_model, db_name=db_name)

    llm = _get_llm(sdk)
    total_samples = len(dataset)

    # --- Progress & logging setup ---
    flog = _setup_file_logger("retrieve_evaluate")
    progress_path = _FILE_DIR / "retrieve_evaluate_progress.json"
    progress = _load_progress(progress_path)
    completed_results: List[Dict[str, Any]] = progress.get("results", [])
    # Build set of (sample_id, question_idx) already done
    completed_keys: Set[str] = {
        f"{r['sample_id']}:{r['question_idx']}" for r in completed_results
    }
    completed_samples: Set[str] = set(progress.get("completed_samples", []))
    if completed_keys:
        flog.info("Resuming retrieve & evaluate – %d questions already completed", len(completed_keys))
        print(f"Resuming: {len(completed_keys)} question(s) already completed, skipping them.")

    results: List[Dict[str, Any]] = list(completed_results)

    # --- Build flat list of work items across all samples ---
    work_items: List[Tuple[str, int, int, Dict[str, Any]]] = []  # (sid, q_idx, total_q, qa)
    sample_order: List[Tuple[int, str, int]] = []  # (s_idx, sid, num_questions)

    for s_idx, item in enumerate(dataset):
        sid = item.get("sample_id", f"conv-{s_idx}")

        if sid in completed_samples:
            flog.info("[Sample %d/%d] SKIP (already done) sample_id=%s", s_idx + 1, total_samples, sid)
            print(f"\n[Sample {s_idx + 1}/{total_samples}] SKIP {sid} (already done)")
            continue

        # Use override QA if provided, otherwise load from dataset
        if test_qa_override is not None and sid in test_qa_override:
            qa_list = test_qa_override[sid]
        else:
            qa_list = item.get("qa", [])
            if categories is not None:
                qa_list = [q for q in qa_list if q.get("category") in categories]
            if max_questions_per_sample is not None:
                qa_list = qa_list[:max_questions_per_sample]

        print(f"\n[Sample {s_idx + 1}/{total_samples}] {sid}: "
              f"evaluating {len(qa_list)} questions")
        sample_order.append((s_idx, sid, len(qa_list)))

        for q_idx, qa in enumerate(qa_list):
            q_key = f"{sid}:{q_idx}"
            if q_key in completed_keys:
                continue
            work_items.append((sid, q_idx, len(qa_list), qa))

    if not work_items:
        print("\nNo pending questions to evaluate.")
        return results

    # --- Worker function for a single question ---
    results_lock = threading.Lock()
    q_counter = len(completed_results)

    def _evaluate_one(sid: str, q_idx: int, total_q: int, qa: Dict[str, Any]) -> Dict[str, Any]:
        question = qa.get("question", "")
        category = qa.get("category", 0)
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}_unknown")
        ground_truth = str(qa.get("answer", ""))
        adversarial_answer = qa.get("adversarial_answer", "")
        evidence = qa.get("evidence", [])

        if not question:
            return {}

        # Per-category PPR weight override
        effective_ppr_w = ppr_score_weight
        if ppr_weight_by_category and category in ppr_weight_by_category:
            effective_ppr_w = ppr_weight_by_category[category]

        # Retrieve
        pack = sdk.retrieve(
            question,
            filters=QueryFilters(agent_id=AGENT_ID, user_id=USER_ID, task_id=sid),
            budget=budget,
            sources="ltm",
            ppr_score_weight=effective_ppr_w,
            sim_score_weight=sim_score_weight,
            degree_correction=degree_correction,
            expansion_depth=expansion_depth,
            edge_type_weights=edge_type_weights,
            adaptive_ppr_model=adaptive_ppr_model,
        )

        # Trim retrieved memory if requested
        if max_memory_words >= 0:
            pack = _trim_memory_pack(pack, max_memory_words, budget=budget)

        retrieved_text = pack.to_text(include_citations=False).strip()

        # Compute reward using selected evaluator
        if evaluator == "hypothesis":
            hypothesis = answer_from_memory(question, pack, llm=llm, model=eval_model, temperature=0)
            if not hypothesis:
                hypothesis = "(No answer generated)"
            reward_result = _compute_hypothesis_reward(
                question, ground_truth, hypothesis, llm, llm_model=eval_model, temperature=0,
            )
        elif evaluator == "bleu":
            hypothesis = "(BLEU evaluation — no LLM)"
            reward_result = _compute_bleu_reward(
                question, ground_truth, retrieved_text,
            )
        else:
            hypothesis = "(evaluation based on retrieved memory)"
            reward_result = _compute_retrieval_reward(
                question, ground_truth, retrieved_text, llm, llm_model=eval_model,
            )
        reward = reward_result["reward"]
        justification = reward_result["justification"]

        entry = {
            "sample_id": sid,
            "question_idx": q_idx,
            "category": category,
            "question_type": cat_name,
            "question": question,
            "answer": ground_truth,
            "adversarial_answer": adversarial_answer,
            "hypothesis": hypothesis,
            "reward": reward,
            "justification": justification,
            "evidence": evidence,
            "retrieved_memory": retrieved_text,
        }

        marker = "+" if reward > 0.5 else "~" if reward > 0 else "-"
        msg = (f"  [{q_idx + 1}/{total_q}] {marker} {sid} cat{category}: "
               f"reward={reward:.2f}  retrieved={len(retrieved_text)} chars")
        flog.info("%s | %s:%d | %s | justification: %s", sid, sid, q_idx, msg.strip(), justification)
        print(msg)
        return entry

    # --- Run evaluation in parallel ---
    workers = min(max_workers, len(work_items))
    print(f"\nEvaluating {len(work_items)} question(s) with {workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_evaluate_one, sid, q_idx, total_q, qa): (sid, q_idx)
            for sid, q_idx, total_q, qa in work_items
        }
        for fut in as_completed(futures):
            sid, q_idx = futures[fut]
            try:
                entry = fut.result()
                if entry:
                    with results_lock:
                        results.append(entry)
                        q_counter += 1
            except Exception:
                flog.error("Failed evaluation for %s:%d", sid, q_idx, exc_info=True)
                print(f"ERROR: evaluation failed for {sid}:{q_idx} (see log)")

    # Mark all processed samples as completed
    for _, sid, _ in sample_order:
        completed_samples.add(sid)
    progress["results"] = results
    progress["completed_samples"] = sorted(completed_samples)
    progress["last_updated"] = datetime.now().isoformat()
    _save_progress(progress_path, progress)

    flog.info("Evaluated %d questions across %d samples.", q_counter, total_samples)
    print(f"\nEvaluated {q_counter} questions across {total_samples} samples.")

    # Write JSONL results
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        flog.info("Results written to %s", output_path)
        print(f"Results written to {output_path}")

    # --- Write per-question evaluation log and summary CSV ---
    _write_evaluation_log_and_summary(results, data_dir)

    return results


def _write_evaluation_log_and_summary(
    results: List[Dict[str, Any]], data_dir: Path,
) -> None:
    """Write detailed evaluation log (JSONL) and summary CSV to *data_dir*."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Detailed log per question
    log_path = data_dir / "evaluation_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Evaluation log written to {log_path}")

    # 2. Summary CSV – one row per (sample_id, category)
    # Accumulate stats
    stats: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    sample_stats: Dict[str, List[float]] = defaultdict(list)
    category_stats: Dict[str, List[float]] = defaultdict(list)

    for r in results:
        sid = r["sample_id"]
        cat_name = r.get("question_type", f"cat{r['category']}_unknown")
        reward = r["reward"]
        stats[(sid, cat_name)].append(reward)
        sample_stats[sid].append(reward)
        category_stats[cat_name].append(reward)

    csv_path = data_dir / "evaluation_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "category", "num_questions", "mean_reward", "num_correct", "accuracy"])

        # Per sample+category rows
        for (sid, cat_name) in sorted(stats.keys()):
            rewards = stats[(sid, cat_name)]
            n = len(rewards)
            correct = sum(1 for r in rewards if r > 0.5)
            mean_r = sum(rewards) / n if n else 0.0
            writer.writerow([sid, cat_name, n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])

        # Per category totals
        writer.writerow([])
        writer.writerow(["TOTAL_BY_CATEGORY", "category", "num_questions", "mean_reward", "num_correct", "accuracy"])
        for cat_name in sorted(category_stats.keys()):
            rewards = category_stats[cat_name]
            n = len(rewards)
            correct = sum(1 for r in rewards if r > 0.5)
            mean_r = sum(rewards) / n if n else 0.0
            writer.writerow(["ALL", cat_name, n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])

        # Per sample totals
        writer.writerow([])
        writer.writerow(["TOTAL_BY_SAMPLE", "category", "num_questions", "mean_reward", "num_correct", "accuracy"])
        for sid in sorted(sample_stats.keys()):
            rewards = sample_stats[sid]
            n = len(rewards)
            correct = sum(1 for r in rewards if r > 0.5)
            mean_r = sum(rewards) / n if n else 0.0
            writer.writerow([sid, "ALL", n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])

        # Overall
        all_rewards = [r["reward"] for r in results]
        n = len(all_rewards)
        correct = sum(1 for r in all_rewards if r > 0.5)
        mean_r = sum(all_rewards) / n if n else 0.0
        writer.writerow([])
        writer.writerow(["OVERALL", "ALL", n, f"{mean_r:.4f}", correct, f"{correct/n:.4f}" if n else "0.0000"])

    print(f"Evaluation summary CSV written to {csv_path}")


def _normalize(s: str) -> str:
    return " ".join(re.split(r"\s+", s.lower().strip()))


def _compute_local_reward(
    question_type: str, question: str, answer: str, hypothesis: str,
) -> int:
    """Rule-based reward: 1 if answer appears in hypothesis, else 0."""
    ans_norm = _normalize(answer)
    hyp_norm = _normalize(hypothesis)
    if not ans_norm:
        return 0
    if len(ans_norm) <= 80:
        return 1 if ans_norm in hyp_norm else 0
    words = set(ans_norm.split())
    hyp_words = set(hyp_norm.split())
    overlap = len(words & hyp_words) / max(len(words), 1)
    return 1 if overlap >= 0.5 else 0


def _compute_openai_reward(
    question_type: str, question: str, answer: str, hypothesis: str,
    llm, llm_model: str,
) -> int:
    """Use LLM to judge correctness."""
    prompt = (
        f"I will give you a question, a correct answer, and a response from a model. "
        f"Answer yes if the response contains the correct answer or equivalent; otherwise no.\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\n"
        f"Model Response: {hypothesis}\n\n"
        f"Is the model response correct? Answer yes or no only."
    )
    try:
        raw = llm.complete(prompt, max_tokens=10, model=llm_model)
        return 1 if "yes" in (raw or "").lower() else 0
    except Exception as e:
        logger.warning("OpenAI judge failed: %s", e)
        return _compute_local_reward(question_type, question, answer, hypothesis)


def _compute_bleu_reward(
    question: str, answer: str, retrieved_text: str,
) -> Dict[str, Any]:
    """ROUGE-L recall between reference answer and retrieved memory (no LLM).

    Uses the ``rouge`` package to compute ROUGE-1, ROUGE-2 and ROUGE-L recall.
    The reward is the ROUGE-L recall score — the fraction of the answer's
    longest common subsequence found in the retrieved text.

    Returns a dict with:
      - reward: float between 0.0 and 1.0 (ROUGE-L recall)
      - justification: description with all three ROUGE recall scores
    """
    from rouge import Rouge

    if not answer.strip() or not retrieved_text.strip():
        return {"reward": 0.0, "justification": "empty reference or retrieved text"}

    rouge = Rouge()
    scores = rouge.get_scores(retrieved_text, answer)[0]

    r1_r = scores["rouge-1"]["r"]
    r2_r = scores["rouge-2"]["r"]
    rl_r = scores["rouge-l"]["r"]

    return {
        "reward": round(rl_r, 4),
        "justification": f"ROUGE-L_recall={rl_r:.4f} (R1_r={r1_r:.3f} R2_r={r2_r:.3f} RL_r={rl_r:.3f})",
    }


def _compute_retrieval_reward(
    question: str, answer: str, retrieved_text: str,
    llm, llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Use GPT-5.1 to judge how much of the reference answer can be produced
    from the retrieved memories.

    Returns a dict with:
      - reward: float between 0.0 and 1.0
          1.0 = all information needed for the complete reference answer is present
          0.0 = none of the required information is present
          intermediate values = partial coverage
      - justification: explanation of the evaluation decision
    """
    if not retrieved_text or not answer:
        return {"reward": 0.0, "justification": "No retrieved text or no reference answer provided."}
    prompt = (
        "You are an evaluator. Given a question, a correct reference answer, and a set of "
        "retrieved memory elements, determine what fraction of the reference answer can be "
        "generated from the retrieved memories.\n\n"
        "Scoring guidelines:\n"
        "- 1.0: All key facts and details needed to produce the complete reference answer "
        "are present in the retrieved memories (even if indirectly without explicit mention).\n"
        "- 0.0: None of the information required to answer the question is found in the "
        "retrieved memories.\n"
        "- Between 0.0 and 1.0: Some relevant information is present (even if indirectly without explicit mention). The score should "
        "reflect the proportion of the reference answer that can be correctly generated.  "
        "For example, if the answer requires 3 key facts and only 2 are present, the score "
        "should be around 0.67.\n\n"
        "You MUST respond in the following JSON format (no markdown, no extra text):\n"
        '{"reward": <float between 0.0 and 1.0>, "justification": "<brief explanation '
        'of what information is present, what is missing, and why you assigned this score>"}\n\n'
        f"Question: {question}\n\n"
        f"Correct Reference Answer: {answer}\n\n"
        f"Retrieved Memories:\n{retrieved_text}\n\n"
        "Evaluate the coverage and respond with the JSON only."
    )
    eval_model = llm_model
    try:
        raw = llm.complete(prompt, max_tokens=400, model=eval_model)
        raw = (raw or "").strip()
        try:
            parsed = json.loads(raw)
            reward = float(parsed.get("reward", 0.0))
            justification = parsed.get("justification", "")
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Fallback: try to extract a number from the raw text
            justification = raw
            import re as _re
            m = _re.search(r"(\d+\.?\d*)", raw)
            reward = float(m.group(1)) if m else 0.0
        # Clamp to [0, 1]
        reward = max(0.0, min(1.0, reward))
        return {"reward": round(reward, 2), "justification": justification}
    except Exception as e:
        logger.warning("Retrieval reward LLM judge failed (model=%s): %s", eval_model, e)
        return {"reward": 0.0, "justification": f"LLM judge failed: {e}"}


def _compute_hypothesis_reward(
    question: str, answer: str, hypothesis: str,
    llm, llm_model: str = "gpt-4o-mini", temperature: float | None = None,
) -> Dict[str, Any]:
    """Generate-then-evaluate: judge how much of the reference answer is
    captured in the *generated hypothesis* (not the raw retrieved memory).

    Returns a dict with:
      - reward: float between 0.0 and 1.0
      - justification: explanation of the evaluation decision
    """
    if not hypothesis or not answer:
        return {"reward": 0.0, "justification": "No hypothesis or no reference answer provided."}
    prompt = (
        "You are an evaluator. Given a question, a correct reference answer, and a "
        "generated response (hypothesis), determine what fraction of the reference answer "
        "is present in the generated response.\n\n"
        "IMPORTANT: Only check whether the key facts from the reference answer appear in "
        "the generated response. Do NOT penalize the response for containing extra "
        "information, additional details, or tangential content beyond the reference answer. "
        "The ONLY thing that matters is whether the reference answer's key facts are covered.\n\n"
        "Scoring guidelines:\n"
        "- 1.0: All key facts and details from the reference answer are present in the "
        "generated response (even if the response also contains extra information).\n"
        "- 0.0: None of the key facts from the reference answer appear in the generated "
        "response.\n"
        "- Between 0.0 and 1.0: Some key facts from the reference answer are present. "
        "Score = (number of reference answer key facts found) / (total key facts in "
        "reference answer). For example, if the answer has 3 key facts and 2 are found "
        "in the response, score = 0.67.\n\n"
        "You MUST respond in the following JSON format (no markdown, no extra text):\n"
        '{"reward": <float between 0.0 and 1.0>, "justification": "<brief explanation '
        'of which reference answer facts are present and which are missing>"}\n\n'
        f"Question: {question}\n\n"
        f"Correct Reference Answer: {answer}\n\n"
        f"Generated Response: {hypothesis}\n\n"
        "Evaluate ONLY the coverage of the reference answer's facts. Do NOT reduce the "
        "score for extra information. Respond with the JSON only."
    )
    eval_model = llm_model
    try:
        raw = llm.complete(prompt, max_tokens=400, model=eval_model, temperature=temperature)
        raw = (raw or "").strip()
        try:
            parsed = json.loads(raw)
            reward = float(parsed.get("reward", 0.0))
            justification = parsed.get("justification", "")
        except (json.JSONDecodeError, ValueError, AttributeError):
            justification = raw
            import re as _re
            m = _re.search(r"(\d+\.?\d*)", raw)
            reward = float(m.group(1)) if m else 0.0
        reward = max(0.0, min(1.0, reward))
        return {"reward": round(reward, 2), "justification": justification}
    except Exception as e:
        logger.warning("Hypothesis reward LLM judge failed (model=%s): %s", eval_model, e)
        return {"reward": 0.0, "justification": f"LLM judge failed: {e}"}


def _compute_adversarial_reward(
    question: str, adversarial_answer: str, ground_truth: str,
    hypothesis: str, llm=None, llm_model: str = "gpt-4o-mini",
) -> int:
    """
    Category 5 adversarial: reward=1 if the model does NOT produce the
    adversarial (wrong) answer.  If ground_truth is available, also check
    the model produced the correct one.
    """
    hyp_norm = _normalize(hypothesis)
    adv_norm = _normalize(adversarial_answer) if adversarial_answer else ""

    # Fail if hypothesis contains the adversarial answer
    if adv_norm and adv_norm in hyp_norm:
        return 0

    # If ground truth available, check it's present
    if ground_truth:
        gt_norm = _normalize(ground_truth)
        if gt_norm and gt_norm in hyp_norm:
            return 1

    # If LLM judge available and ground truth exists
    if llm and ground_truth:
        return _compute_openai_reward("cat5_adversarial", question, ground_truth, hypothesis,
                                      llm, llm_model)

    # No adversarial answer found → pass (benefit of the doubt)
    return 1 if adv_norm and adv_norm not in hyp_norm else 0



# ============================================================================
# NEURAL PPR — DATA GENERATION, TRAINING, SUMMARISATION
# ============================================================================

def generate_ppr_training_data(
    data_path: Path,
    data_dir: Path,
    output_dir: Path,
    sample_ids: Optional[List[str]] = None,
    categories: Optional[List[int]] = None,
    ppr_weights: Optional[List[float]] = None,
    budget: Optional[RetrievalBudget] = None,
    llm_model: str = "gpt-4o-mini",
    eval_model: str = "gpt-4o-mini",
    db_name: str = "locomo_memory_with_bl.sqlite",
    max_workers: int = 20,
    max_memory_words: int = 600,
    edge_type_weights: Optional[dict] = None,
    limit: Optional[int] = None,
) -> Path:
    """Run evaluations at each ppr_weight and store results in CSV + embeddings.

    For every (sample, question, ppr_weight) triple, runs retrieve_and_evaluate
    and records the reward. Produces two files:
      - ``output_dir/ppr_training_data.csv``   — sample_id, category, question, ppr_weight, reward
      - ``output_dir/ppr_training_embeddings.json`` — {question: embedding} (query text → embedding)

    Parameters
    ----------
    data_path    : Path to locomo10.json
    data_dir     : Directory where SDK data lives
    output_dir   : Where to write CSV + embedding files
    sample_ids   : Restrict to these samples (None = all)
    categories   : Categories to include (default [1,2,3,4])
    ppr_weights  : PPR weight values to evaluate (default [0.01, 0.5, 1.0])
    budget       : Retrieval budget
    llm_model    : Model for answer generation
    eval_model   : Model for judging
    db_name      : SQLite DB filename
    max_workers  : Parallel workers per evaluation run
    max_memory_words : Trim retrieved memory to at most this many words
    edge_type_weights : Override edge-type weights for PPR transitions
    limit        : Max samples

    Returns
    -------
    csv_path : Path to the generated CSV file
    """
    if categories is None:
        categories = [1, 2, 3, 4]
    if ppr_weights is None:
        ppr_weights = [0.01, 0.5, 1.0]
    if budget is None:
        budget = DEFAULT_BUDGET

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ppr_training_data.csv"
    emb_path = output_dir / "ppr_training_embeddings.json"

    sdk = _make_sdk(data_dir, llm_model=llm_model, db_name=db_name)

    # Collect embeddings for all questions
    embedder = sdk._orchestrator._ltm_retriever._vector_store._embedder

    # Run evaluation at each ppr_weight
    all_rows: List[Dict[str, Any]] = []
    for pw in ppr_weights:
        print(f"\n{'='*60}")
        print(f"Evaluating ppr_weight={pw}, sim_weight=1.0")
        print(f"{'='*60}")

        # Clear progress for fresh run
        progress_path = _FILE_DIR / "retrieve_evaluate_progress.json"
        if progress_path.exists():
            progress_path.unlink()

        results = retrieve_and_evaluate(
            data_path=data_path,
            data_dir=data_dir,
            limit=limit,
            budget=budget,
            llm_model=llm_model,
            sample_ids=sample_ids,
            sdk=sdk,
            categories=categories,
            db_name=db_name,
            ppr_score_weight=pw,
            sim_score_weight=1.0,
            evaluator="hypothesis",
            eval_model=eval_model,
            max_workers=max_workers,
            max_memory_words=max_memory_words,
            edge_type_weights=edge_type_weights,
        )

        for r in results:
            all_rows.append({
                "sample_id": r.get("sample_id", ""),
                "category": r.get("category", 0),
                "question": r.get("question", ""),
                "ppr_weight": pw,
                "reward": r.get("reward", 0.0),
                "hypothesis": r.get("hypothesis", ""),
                "answer": r.get("answer", ""),
            })

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "category", "question", "ppr_weight", "reward", "hypothesis", "answer",
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")

    # Generate embeddings for unique questions
    unique_questions = list({r["question"] for r in all_rows if r["question"]})
    print(f"\nEmbedding {len(unique_questions)} unique questions...")
    embeddings_dict: Dict[str, List[float]] = {}
    batch_size = 100
    for i in range(0, len(unique_questions), batch_size):
        batch = unique_questions[i:i + batch_size]
        batch_embs = embedder.embed_batch(batch)
        for q, emb in zip(batch, batch_embs):
            embeddings_dict[q] = emb
        print(f"  Embedded {min(i + batch_size, len(unique_questions))}/{len(unique_questions)}")

    emb_path.write_text(json.dumps(embeddings_dict), encoding="utf-8")
    print(f"Wrote embeddings to {emb_path}")

    return csv_path


def summarize_ppr_eval(csv_path: Path) -> Dict[str, Any]:
    """Summarise a ppr_training_data.csv file.

    Prints and returns per-category and per-ppr_weight reward averages.
    """
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data in CSV.")
        return {}

    # Group by (ppr_weight, category)
    from collections import defaultdict
    by_pw_cat: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_pw: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        pw = row["ppr_weight"]
        cat = CATEGORY_NAMES.get(int(row["category"]), f"cat{row['category']}")
        reward = float(row["reward"])
        by_pw_cat[pw][cat].append(reward)
        by_pw[pw].append(reward)

    print("\n" + "=" * 80)
    print("PPR TRAINING DATA SUMMARY")
    print("=" * 80)

    # Header
    all_cats = sorted({cat for pw_cats in by_pw_cat.values() for cat in pw_cats})
    header = f"{'ppr_weight':>12s}"
    for cat in all_cats:
        header += f"  {cat:>16s}"
    header += f"  {'OVERALL':>12s}  {'N':>6s}"
    print(header)
    print("-" * len(header))

    summary = {}
    for pw in sorted(by_pw.keys(), key=float):
        line = f"{pw:>12s}"
        pw_summary = {}
        for cat in all_cats:
            vals = by_pw_cat[pw].get(cat, [])
            avg = sum(vals) / len(vals) if vals else 0.0
            line += f"  {avg:>15.1%}"
            pw_summary[cat] = {"avg": avg, "n": len(vals)}
        overall = by_pw[pw]
        avg_overall = sum(overall) / len(overall) if overall else 0.0
        line += f"  {avg_overall:>11.1%}  {len(overall):>6d}"
        pw_summary["overall"] = {"avg": avg_overall, "n": len(overall)}
        summary[pw] = pw_summary
        print(line)

    print("=" * 80)
    return summary


def train_neural_ppr(
    csv_path: Path,
    embeddings_path: Path,
    model_save_path: Path,
    train_ratio: float = 0.7,
    hidden_dim: int = 16,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 16,
    seed: int = 42,
    log: bool = True,
) -> Tuple[NeuralPPRModel, Dict[str, Any]]:
    """Train a NeuralPPRModel from CSV + embeddings files.

    Splits data into train/test by questions (not rows — all ppr_weight
    rows for a question go to the same split).

    Parameters
    ----------
    csv_path        : Path to ppr_training_data.csv
    embeddings_path : Path to ppr_training_embeddings.json
    model_save_path : Where to save the trained model
    train_ratio     : Fraction of questions for training (default 0.7)
    hidden_dim      : MLP hidden dimension
    epochs          : Training epochs
    lr              : Learning rate
    batch_size      : Mini-batch size
    seed            : Random seed
    log             : Print progress

    Returns
    -------
    (model, split_info) where split_info contains train/test question sets
    and by_question/embeddings for evaluation.
    """
    # Load data
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    embeddings = json.loads(Path(embeddings_path).read_text(encoding="utf-8"))

    # Group rows by question
    from collections import defaultdict
    by_question: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_question[row["question"]].append(row)

    # Split by question
    questions = sorted(by_question.keys())
    rng = random.Random(seed)
    rng.shuffle(questions)
    k = max(1, int(len(questions) * train_ratio))
    train_questions = set(questions[:k])
    test_questions = set(questions[k:])
    all_questions = set(questions)

    # Build (embedding, gap) pairs where gap = r_ppr1 - r_ppr0
    def _build_gap_data(q_set):
        pairs = []
        for q in q_set:
            emb = embeddings.get(q)
            if not emb:
                continue
            rewards_by_w = {}
            for row in by_question[q]:
                pw = float(row["ppr_weight"])
                rewards_by_w[pw] = float(row["reward"])
            r_low = rewards_by_w.get(0.01, 0)
            r_high = rewards_by_w.get(1.0, 0)
            gap = r_high - r_low
            pairs.append((emb, gap))
        return pairs

    train_data = _build_gap_data(train_questions)
    test_data = _build_gap_data(test_questions)

    n_diff = sum(1 for _, g in train_data if g != 0)
    if log:
        print(f"\nNeural PPR Training (Gap Regression)")
        print(f"  Train questions: {len(train_questions)} ({n_diff} with signal)")
        print(f"  Test questions: {len(test_questions)}")

    embed_dim = len(train_data[0][0]) if train_data else 1536

    random.seed(seed)
    model = NeuralPPRModel(embed_dim=embed_dim, hidden_dim=hidden_dim)
    trainer = NeuralPPRTrainer(model, lr=lr, weight_decay=0.01)

    log_every = max(1, epochs // 4)
    stats = trainer.train(
        data=train_data,
        epochs=epochs,
        batch_size=batch_size,
        log_every=log_every,
        val_data=test_data if test_data else None,
        early_stopping_patience=50,
        diff_weight=10.0,
        tie_weight=0.5,
    )

    if log:
        print(f"\n  Final train loss: {stats['final_loss']:.6f}")
        if stats.get("final_val_loss") is not None:
            print(f"  Final val loss:   {stats['final_val_loss']:.6f}")

    # Evaluate on test set (30%)
    if test_data and log:
        print("\n  --- Evaluation on test set (30%) ---")
        _evaluate_neural_model(model, test_questions, by_question, embeddings)

    # Evaluate on full dataset
    if log:
        print("\n  --- Evaluation on full dataset ---")
        _evaluate_neural_model(model, all_questions, by_question, embeddings)

    # Save model
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    if log:
        print(f"\n  Model saved to {model_save_path}")

    # Return split info for downstream evaluation
    split_info = {
        "train_questions": train_questions,
        "test_questions": test_questions,
        "all_questions": all_questions,
        "by_question": by_question,
        "embeddings": embeddings,
    }

    return model, split_info


def _evaluate_neural_model(
    model: NeuralPPRModel,
    test_questions,
    by_question: Dict[str, List[dict]],
    embeddings: Dict[str, List[float]],
):
    """Evaluate predicted vs actual best ppr_weight on test set."""
    from collections import defaultdict

    cat_stats: Dict[str, Dict] = defaultdict(lambda: {
        "pred_0": 0, "pred_1": 0, "correct": 0, "total": 0,
        "pred_reward": [], "best_reward": [],
    })

    for q in test_questions:
        emb = embeddings.get(q)
        if not emb:
            continue
        rows = by_question[q]
        cat = CATEGORY_NAMES.get(int(rows[0]["category"]), f"cat{rows[0]['category']}")

        rewards_by_w = {}
        for row in rows:
            pw = float(row["ppr_weight"])
            rewards_by_w[pw] = float(row["reward"])

        r_low = rewards_by_w.get(0.01, 0)
        r_high = rewards_by_w.get(1.0, 0)
        actual_gap = r_high - r_low

        predicted_w = model.predict(emb)
        predicted_r = rewards_by_w.get(predicted_w, 0.0)
        best_r = max(rewards_by_w.values()) if rewards_by_w else 0.0

        d = cat_stats[cat]
        d["total"] += 1
        if predicted_w >= 0.5:
            d["pred_1"] += 1
        else:
            d["pred_0"] += 1
        # Correct if: predicted high and gap>0, predicted low and gap<0, or tie
        if actual_gap == 0 or (predicted_w >= 0.5 and actual_gap > 0) or (predicted_w < 0.5 and actual_gap <= 0):
            d["correct"] += 1
        d["pred_reward"].append(predicted_r)
        d["best_reward"].append(best_r)

    print(f"\n  {'Category':<18s} {'Acc':>7s} {'Pred=0':>8s} {'Pred=1':>8s} {'Pred_R':>8s} {'Best_R':>8s} {'N':>6s}")
    print(f"  {'-'*58}")
    total_correct = total_n = 0
    total_pred_r = []
    total_best_r = []
    for cat in sorted(cat_stats.keys()):
        d = cat_stats[cat]
        acc = d["correct"] / d["total"] if d["total"] else 0
        avg_pred_r = sum(d["pred_reward"]) / len(d["pred_reward"]) if d["pred_reward"] else 0
        avg_best_r = sum(d["best_reward"]) / len(d["best_reward"]) if d["best_reward"] else 0
        print(f"  {cat:<18s} {acc:>6.1%} {d['pred_0']:>8} {d['pred_1']:>8} "
              f"{avg_pred_r:>7.1%} {avg_best_r:>7.1%} {d['total']:>6}")
        total_correct += d["correct"]
        total_n += d["total"]
        total_pred_r.extend(d["pred_reward"])
        total_best_r.extend(d["best_reward"])
    if total_n:
        acc = total_correct / total_n
        avg_pr = sum(total_pred_r) / len(total_pred_r)
        avg_br = sum(total_best_r) / len(total_best_r)
        print(f"  {'OVERALL':<18s} {acc:>6.1%} {'':>8s} {'':>8s} "
              f"{avg_pr:>7.1%} {avg_br:>7.1%} {total_n:>6}")


# ============================================================================
# COMPUTE REWARD SUMMARY
# ============================================================================

def compute_reward_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute overall and per-category reward percentages.
    Also breaks down by sample_id.

    Parameters
    ----------
    results : Output from retrieve_and_evaluate()

    Returns
    -------
    summary : Dict with 'overall', 'by_category', and 'by_sample' breakdowns
    """
    if not results:
        return {"overall": 0.0, "total": 0, "by_category": {}, "by_sample": {}}

    cat_rewards: Dict[str, List[float]] = defaultdict(list)
    sample_rewards: Dict[str, List[float]] = defaultdict(list)
    all_rewards: List[float] = []

    for r in results:
        reward = float(r.get("reward", 0))
        cat_name = r.get("question_type", "unknown")
        sid = r.get("sample_id", "unknown")
        cat_rewards[cat_name].append(reward)
        sample_rewards[sid].append(reward)
        all_rewards.append(reward)

    overall = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    by_category = {}
    for cat in sorted(cat_rewards.keys()):
        vals = cat_rewards[cat]
        avg = sum(vals) / len(vals) if vals else 0.0
        by_category[cat] = {"avg_reward": avg, "sum_reward": sum(vals), "total": len(vals)}

    by_sample = {}
    for sid in sorted(sample_rewards.keys()):
        vals = sample_rewards[sid]
        avg = sum(vals) / len(vals) if vals else 0.0
        by_sample[sid] = {"avg_reward": avg, "sum_reward": sum(vals), "total": len(vals)}

    summary = {
        "overall": overall,
        "total": len(all_rewards),
        "sum_reward": sum(all_rewards),
        "by_category": by_category,
        "by_sample": by_sample,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("REWARD SUMMARY — BY CATEGORY")
    print("=" * 60)
    for cat, info in by_category.items():
        print(f"  {cat:30s}: {info['avg_reward']:.4f} "
              f"({info['sum_reward']:.2f}/{info['total']})")
    print(f"\n  {'OVERALL':30s}: {overall:.4f} "
          f"({sum(all_rewards):.2f}/{len(all_rewards)})")

    print("\n" + "=" * 60)
    print("REWARD SUMMARY — BY SAMPLE")
    print("=" * 60)
    for sid, info in by_sample.items():
        print(f"  {sid:30s}: {info['avg_reward']:.4f} "
              f"({info['sum_reward']:.2f}/{info['total']})")
    print("=" * 60)

    return summary


DB_WITH_BL = "locomo_memory_with_bl.sqlite"
DB_WITHOUT_BL = "locomo_memory_without_bl.sqlite"


# ============================================================================
# 5. PRINT ALL MEMORY FOR A SAMPLE
# ============================================================================

def print_all_memory(
    data_dir: Path,
    sample_id: str = "conv-26",
    output_path: Optional[Path] = None,
    db_name: str = "locomo_memory.sqlite",
    llm_model: str = "gpt-4o-mini",
) -> str:
    """Retrieve ALL memory nodes for a sample and write them as MemoryPack text.

    Queries the node store for every node scoped to *sample_id*, groups them
    by kind (fact / reflection / skill / episode), builds a MemoryPack, and
    writes ``pack.to_text()`` to *output_path* (overwriting any existing file).

    Parameters
    ----------
    data_dir     : Directory where the SDK database lives.
    sample_id    : The task_id / sample_id to dump (default ``"conv-26"``).
    output_path  : File to write. Defaults to ``<data_dir>/<sample_id>_memory.txt``.
    db_name      : SQLite database filename.
    llm_model    : Model name (only used for SDK init).

    Returns
    -------
    text : The full ``pack.to_text()`` string that was written.
    """
    sdk = _make_sdk(data_dir, llm_model=llm_model, db_name=db_name)
    node_store = sdk._orchestrator._node_store

    filters = QueryFilters(agent_id=AGENT_ID, user_id=USER_ID, task_id=sample_id)
    all_nodes = node_store.query(filters, limit=100_000)

    # Bucket by kind
    facts: List[str] = []
    reflections: List[str] = []
    skills: List[str] = []
    episodes: List[str] = []

    for node in all_nodes:
        kind = (node.kind or "").lower()
        if kind == "fact":
            facts.append(node.fact_text)
        elif kind == "reflection":
            reflections.append(node.reflection_text)
        elif kind == "skill":
            skills.append(node.skill_description)
        elif kind == "episode":
            episodes.append(node.summary)

    pack = MemoryPack(
        facts=facts,
        reflections=reflections,
        skills=skills,
        episodes=episodes,
    )
    text = pack.to_text(include_citations=False)

    if output_path is None:
        output_path = Path(data_dir) / f"{sample_id}_memory.txt"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {len(all_nodes)} nodes ({len(facts)} facts, {len(reflections)} reflections, "
          f"{len(skills)} skills, {len(episodes)} episodes) to {output_path}")
    return text
