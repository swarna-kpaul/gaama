"""
MemoryBench Group Travel Planner Evaluation Module
===================================================
Two evaluation classes:
  - GroupTravelSDKEval      GAAMA memory SDK retrieval-augmented evaluation
  - GroupTravelBaselineEval Full chat-history baseline (perfect memory upper bound)

Each entry has a base_person (seed trip plan) and 5-8 sequential questions
where new travelers join the group with constraints.  Q0 needs base context;
Q1+ needs memory of prior travelers' plans for constraint satisfaction.

Shared utilities:
  - load_dataset(path)       Load group_travel_planner.jsonl
  - compute_summary(results) Per-step and overall reward summary
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# SDK path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gaama.api import AgenticMemorySDK, create_default_sdk  # noqa: E402
from gaama.config.settings import (  # noqa: E402
    SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
)
from gaama.core import (  # noqa: E402
    TraceEvent, QueryFilters, RetrievalBudget,
)
from gaama.services.interfaces import CreateOptions  # noqa: E402

sys.path.insert(0, str(_FILE_DIR))
from config import (  # noqa: E402
    AGENT_ID, USER_ID, BUDGET, PPR_WEIGHT, SIM_WEIGHT,
    MAX_MEMORY_WORDS, LLM_MODEL, EVAL_MODEL, EMBEDDING_MODEL,
    DB_NAME, DATA_DIR,
)

logger = logging.getLogger(__name__)

# Itinerary fields to compare (skip "days" and "current_city" as structural)
_PLAN_FIELDS = ["transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[dict]:
    """Load group_travel_planner.jsonl."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_daily_plans(daily_plans: List[dict]) -> str:
    """Format a list of day-plan dicts into readable text."""
    lines = []
    for day in daily_plans:
        lines.append(f"Day {day['days']}: {day['current_city']}")
        for field in _PLAN_FIELDS:
            val = day.get(field, "-")
            if val and val != "-":
                lines.append(f"  {field}: {val}")
    return "\n".join(lines)


def _format_base_context(entry: dict) -> str:
    """Build the base context from the base_person's query and daily plans."""
    bp = entry["base_person"]
    plan_text = _format_daily_plans(bp["daily_plans"])
    return (
        f"{bp['query']}\n\n"
        f"## Base Itinerary\n{plan_text}"
    )


def _extract_person_name(question: str) -> str:
    """Extract person name from 'I am Eric.' pattern."""
    m = re.match(r"I am (\w+)\.", question)
    return m.group(1) if m else "Unknown"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_itinerary(
    ground_truth: List[dict],
    hypothesis: str,
    llm,
    eval_model: str = EVAL_MODEL,
) -> Dict[str, Any]:
    """Score a hypothesis itinerary against the ground truth daily plans.

    Uses LLM judge to determine what fraction of the ground truth plan
    details are present in the hypothesis.
    """
    if not hypothesis:
        return {"reward": 0.0, "field_match": 0.0, "justification": "No hypothesis provided."}

    gt_text = _format_daily_plans(ground_truth)

    # Count non-empty GT fields for reference
    total_fields = 0
    gt_details = []
    for day in ground_truth:
        for field in _PLAN_FIELDS:
            val = day.get(field, "-")
            if val and val != "-":
                total_fields += 1
                gt_details.append(f"Day {day['days']} {field}: {val}")

    gt_details_text = "\n".join(gt_details)

    prompt = (
        "You are evaluating a travel itinerary. Compare the ground truth plan details "
        "against a model's generated itinerary.\n\n"
        f"Ground truth plan details ({total_fields} items):\n{gt_details_text}\n\n"
        f"Model's itinerary:\n{hypothesis}\n\n"
        "For each ground truth item, check if the same choice (restaurant name, "
        "accommodation name, attraction, transportation) appears in the model's output. "
        "Score = fraction of ground truth items correctly present.\n\n"
        "Score 1.0 if all items match, 0.0 if none match. Score proportionally "
        "for partial matches.\n\n"
        "IMPORTANT: Match on the key identifying details (e.g. restaurant name, "
        "hotel name). Minor formatting differences are OK.\n\n"
        'Respond with JSON only: {"reward": <float 0-1>, "justification": "<brief>"}'
    )
    return _parse_llm_json_reward(prompt, llm, eval_model)


def _parse_llm_json_reward(
    prompt: str, llm, eval_model: str,
) -> Dict[str, Any]:
    """Call LLM, parse JSON reward response with retries."""
    last_raw = ""
    last_error = ""

    for attempt in range(3):
        try:
            if attempt == 0:
                raw = llm.complete(prompt, max_tokens=400, model=eval_model, temperature=0)
            else:
                retry_prompt = (
                    f"Your previous response could not be parsed as valid JSON.\n\n"
                    f"Error: {last_error}\n\n"
                    f"Your previous output:\n{last_raw[:500]}\n\n"
                    f"Please return ONLY valid JSON with no markdown fences, no extra text.\n"
                    f'Required format: {{"reward": <float 0.0-1.0>, "justification": "<brief>"}}'
                )
                raw = llm.complete(retry_prompt, max_tokens=400, model=eval_model, temperature=0)
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
                raw = raw.strip()
            parsed = json.loads(raw)
            reward = max(0.0, min(1.0, float(parsed.get("reward", 0.0))))
            return {"reward": round(reward, 2), "justification": parsed.get("justification", "")}
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as e:
            last_raw = raw if raw else ""
            last_error = str(e)
            logger.warning("JSON parse failed (attempt %d/3): %s", attempt + 1, e)
        except Exception as e:
            logger.warning("LLM judge failed (model=%s): %s", eval_model, e)
            return {"reward": 0.0, "justification": f"LLM judge failed: {e}"}

    return {"reward": 0.0, "justification": f"JSON parse failed after 3 attempts: {last_error}"}


# ---------------------------------------------------------------------------
# SDK factory
# ---------------------------------------------------------------------------

def _make_sdk(
    data_dir: Path = DATA_DIR,
    llm_model: str = LLM_MODEL,
    db_name: str = DB_NAME,
) -> AgenticMemorySDK:
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
            model=EMBEDDING_MODEL,
            api_key_env="OPENAI_API_KEY",
        ),
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    return create_default_sdk(settings, agent_id=AGENT_ID)


def _get_llm(sdk: AgenticMemorySDK):
    llm = getattr(sdk._orchestrator._extractor, "_llm", None)
    if llm is None:
        raise RuntimeError("No LLM adapter on extractor. Ensure OPENAI_API_KEY is set.")
    return llm


# ---------------------------------------------------------------------------
# SDK Evaluation
# ---------------------------------------------------------------------------

class GroupTravelSDKEval:
    """Evaluate Group Travel Planner entries using GAAMA memory SDK retrieval.

    Flow per entry:
      Q0: base_context + person_request → LLM → hypothesis itinerary
      Q1+: retrieve memory from person_request,
           base_context + memory + person_request → LLM → hypothesis
      After every question: ingest person_request + reference itinerary
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        llm_model: str = LLM_MODEL,
        eval_model: str = EVAL_MODEL,
        db_name: str = DB_NAME,
        budget: RetrievalBudget | None = None,
        ppr_weight: float = PPR_WEIGHT,
        sim_weight: float = SIM_WEIGHT,
        max_memory_words: int = MAX_MEMORY_WORDS,
        semantic_only: bool = False,
        max_workers: int = 10,
        output_dir: Path | None = None,
    ):
        self.data_dir = data_dir
        self.llm_model = llm_model
        self.eval_model = eval_model
        self.db_name = db_name
        self.budget = budget or BUDGET
        self.ppr_weight = ppr_weight
        self.sim_weight = sim_weight
        self.max_memory_words = max_memory_words
        self.semantic_only = semantic_only
        self.max_workers = max_workers
        self.output_dir = output_dir or (_FILE_DIR / "data" / "results" / "sdk")

    def cleanup(self):
        """Delete all per-entry LTM databases, blobs, and output files before a fresh run."""
        if self.data_dir.exists():
            for db_file in self.data_dir.glob("*.sqlite*"):
                db_file.unlink()
            print(f"Cleaned DBs in {self.data_dir}", flush=True)

        blobs_dir = self.data_dir / "blobs"
        if blobs_dir.exists():
            shutil.rmtree(blobs_dir)
            print(f"Deleted {blobs_dir}", flush=True)

        if self.output_dir.exists():
            for f in self.output_dir.glob("*"):
                if f.is_file():
                    f.unlink()
            print(f"Cleaned {self.output_dir}", flush=True)

    def evaluate_entry(self, entry: dict) -> List[Dict[str, Any]]:
        """Evaluate one entry (5-8 sequential traveler questions) using SDK retrieval."""
        entry_id = str(entry["id"])
        questions = entry["questions"]
        answers = entry["answers"]
        base_context = _format_base_context(entry)

        entry_db = f"entry_{entry_id}_{self.db_name}"
        sdk = _make_sdk(self.data_dir, llm_model=self.llm_model, db_name=entry_db)
        llm = _get_llm(sdk)

        results = []
        base_ts = datetime(2025, 1, 1, 12, 0, 0)

        for step_idx in range(len(questions)):
            question = questions[step_idx]
            answer = answers[step_idx]
            person_name = _extract_person_name(question)

            # --- Retrieve memory and build prompt ---
            retrieved_text = ""

            if step_idx == 0:
                prompt = f"{base_context}\n\n{question}"
            else:
                pack = sdk.retrieve(
                    question,
                    filters=QueryFilters(agent_id=AGENT_ID, user_id=USER_ID, task_id=entry_id),
                    budget=self.budget,
                    sources="ltm",
                    ppr_score_weight=self.ppr_weight,
                    sim_score_weight=self.sim_weight,
                    semantic_only=self.semantic_only,
                    max_memory_words=self.max_memory_words,
                )
                retrieved_text = pack.to_text(include_citations=False).strip()

                if retrieved_text:
                    prompt = (
                        f"{base_context}\n\n"
                        f"## Retrieved Memory\n{retrieved_text}\n\n"
                        f"Question:\n{question}"
                    )
                else:
                    prompt = f"{base_context}\n\n{question}"

            # --- Generate hypothesis ---
            hypothesis = (
                llm.complete(prompt, max_tokens=1024, model=self.eval_model, temperature=0) or ""
            ).strip()

            # --- Score ---
            scoring = _score_itinerary(answer, hypothesis, llm, self.eval_model)

            results.append({
                "entry_id": entry["id"],
                "step_idx": step_idx,
                "person_name": person_name,
                "ground_truth": answer,
                "hypothesis": hypothesis,
                "reward": scoring["reward"],
                "justification": scoring["justification"],
                "retrieved_memory": retrieved_text,
                "memory_needed": step_idx > 0,
            })

            marker = "+" if scoring["reward"] > 0.5 else "~" if scoring["reward"] > 0 else "-"
            print(
                f"  {marker} entry_{entry_id} step{step_idx} ({person_name}): "
                f"reward={scoring['reward']:.2f}",
                flush=True,
            )

            # --- Ingest question + reference itinerary into memory ---
            answer_text = _format_daily_plans(answer)
            event = TraceEvent(
                event_id=f"entry{entry_id}_step{step_idx}",
                event_type="message",
                actor="travel_agent",
                content=(
                    f"{question}\n\n"
                    f"Planned itinerary:\n{answer_text}"
                ),
                ts=base_ts + timedelta(minutes=step_idx),
                metadata={},
            )
            sdk.ingest([event])
            sdk.create(CreateOptions(user_id=USER_ID, task_id=entry_id))

        return results

    def run(self, dataset: List[dict]) -> List[Dict[str, Any]]:
        """Cleanup, then evaluate all entries in parallel."""
        self.cleanup()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        workers = min(self.max_workers, len(dataset))
        print(f"\nSDK Eval: {len(dataset)} entries, {workers} workers\n", flush=True)

        all_results: List[Dict[str, Any]] = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.evaluate_entry, entry): str(entry["id"])
                for entry in dataset
            }
            for fut in as_completed(futures):
                eid = futures[fut]
                try:
                    entry_results = fut.result()
                    all_results.extend(entry_results)
                    avg_r = sum(r["reward"] for r in entry_results) / len(entry_results)
                    print(f"[done] entry_{eid}: avg_reward={avg_r:.3f}", flush=True)
                except Exception:
                    logger.error("Failed: entry_%s", eid, exc_info=True)
                    print(f"ERROR: entry_{eid} failed (see log)", flush=True)

        elapsed = time.time() - t0
        print(f"\nSDK Eval done. {len(all_results)} results in {elapsed:.1f}s", flush=True)

        if all_results:
            save_results_jsonl(all_results, self.output_dir / "evaluation_log.jsonl")

        return all_results


# ---------------------------------------------------------------------------
# Baseline Evaluation
# ---------------------------------------------------------------------------

class GroupTravelBaselineEval:
    """Evaluate Group Travel Planner entries using full chat history.

    Flow per entry:
      Q0: base_context + person_request → LLM → hypothesis
      Q1+: base_context + chat history (all prior Q&A) + person_request → LLM → hypothesis
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        llm_model: str = LLM_MODEL,
        eval_model: str = EVAL_MODEL,
        db_name: str = DB_NAME,
        max_workers: int = 10,
        output_dir: Path | None = None,
    ):
        self.data_dir = data_dir
        self.llm_model = llm_model
        self.eval_model = eval_model
        self.db_name = db_name
        self.max_workers = max_workers
        self.output_dir = output_dir or (_FILE_DIR / "data" / "results" / "baseline")

    def evaluate_entry(self, entry: dict) -> List[Dict[str, Any]]:
        """Evaluate one entry using full chat-history context stuffing."""
        entry_id = str(entry["id"])
        questions = entry["questions"]
        answers = entry["answers"]
        base_context = _format_base_context(entry)

        sdk = _make_sdk(self.data_dir, llm_model=self.llm_model, db_name=self.db_name)
        llm = _get_llm(sdk)

        results = []
        history: List[Dict[str, str]] = []

        for step_idx in range(len(questions)):
            question = questions[step_idx]
            answer = answers[step_idx]
            person_name = _extract_person_name(question)

            # --- Build prompt ---
            if step_idx == 0:
                prompt = f"{base_context}\n\n{question}"
            else:
                history_lines = []
                for h in history:
                    history_lines.append(f"Q: {h['question']}\nA: {h['answer']}")
                history_block = "\n\n".join(history_lines)

                prompt = (
                    f"{base_context}\n\n"
                    f"## Chat History\n{history_block}\n\n"
                    f"Question:\n{question}"
                )

            # --- Generate hypothesis ---
            hypothesis = (
                llm.complete(prompt, max_tokens=1024, model=self.eval_model, temperature=0) or ""
            ).strip()

            # --- Score ---
            scoring = _score_itinerary(answer, hypothesis, llm, self.eval_model)

            results.append({
                "entry_id": entry["id"],
                "step_idx": step_idx,
                "person_name": person_name,
                "ground_truth": answer,
                "hypothesis": hypothesis,
                "reward": scoring["reward"],
                "justification": scoring["justification"],
                "retrieved_memory": "",
                "memory_needed": step_idx > 0,
            })

            marker = "+" if scoring["reward"] > 0.5 else "~" if scoring["reward"] > 0 else "-"
            print(
                f"  {marker} entry_{entry_id} step{step_idx} ({person_name}): "
                f"reward={scoring['reward']:.2f}",
                flush=True,
            )

            # --- Add to history ---
            answer_text = _format_daily_plans(answer)
            history.append({
                "question": question,
                "answer": f"Planned itinerary:\n{answer_text}",
            })

        return results

    def run(self, dataset: List[dict]) -> List[Dict[str, Any]]:
        """Evaluate all entries in parallel."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        workers = min(self.max_workers, len(dataset))
        print(f"\nBaseline Eval: {len(dataset)} entries, {workers} workers\n", flush=True)

        all_results: List[Dict[str, Any]] = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.evaluate_entry, entry): str(entry["id"])
                for entry in dataset
            }
            for fut in as_completed(futures):
                eid = futures[fut]
                try:
                    entry_results = fut.result()
                    all_results.extend(entry_results)
                    avg_r = sum(r["reward"] for r in entry_results) / len(entry_results)
                    print(f"[done] entry_{eid}: avg_reward={avg_r:.3f}", flush=True)
                except Exception:
                    logger.error("Failed: entry_%s", eid, exc_info=True)
                    print(f"ERROR: entry_{eid} failed (see log)", flush=True)

        elapsed = time.time() - t0
        print(f"\nBaseline Eval done. {len(all_results)} results in {elapsed:.1f}s", flush=True)

        if all_results:
            save_results_jsonl(all_results, self.output_dir / "evaluation_log.jsonl")

        return all_results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results_jsonl(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to {path}", flush=True)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-step and overall reward summary."""
    if not results:
        print("No results to summarize.", flush=True)
        return {"overall": 0.0, "total": 0}

    all_rewards = [r["reward"] for r in results]
    overall = sum(all_rewards) / len(all_rewards)

    step_rewards: Dict[int, List[float]] = defaultdict(list)
    memory_rewards: Dict[bool, List[float]] = defaultdict(list)

    for r in results:
        step_rewards[r["step_idx"]].append(r["reward"])
        memory_rewards[r["memory_needed"]].append(r["reward"])

    n = len(all_rewards)
    correct = sum(1 for r in all_rewards if r > 0.5)

    print("\n" + "=" * 70, flush=True)
    print("REWARD SUMMARY — Group Travel Planner", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  OVERALL: n={n}  reward={overall:.4f}  acc={correct/n:.4f}", flush=True)

    print(f"\n{'--- By Step ---':^70}", flush=True)
    by_step = {}
    for step in sorted(step_rewards):
        rw = step_rewards[step]
        avg = sum(rw) / len(rw)
        c = sum(1 for r in rw if r > 0.5)
        mem = " (no memory)" if step == 0 else " (memory)"
        print(f"  step_{step}{mem:15s}  n={len(rw):4d}  reward={avg:.4f}  acc={c/len(rw):.4f}", flush=True)
        by_step[step] = {"avg_reward": avg, "total": len(rw), "correct": c}

    print(f"\n{'--- Memory Impact ---':^70}", flush=True)
    q0 = memory_rewards.get(False, [])
    q1plus = memory_rewards.get(True, [])
    if q0:
        print(f"  Q0 (no memory):  n={len(q0):4d}  reward={sum(q0)/len(q0):.4f}", flush=True)
    if q1plus:
        print(f"  Q1+ (memory):    n={len(q1plus):4d}  reward={sum(q1plus)/len(q1plus):.4f}", flush=True)
    if q0 and q1plus:
        delta = sum(q1plus) / len(q1plus) - sum(q0) / len(q0)
        print(f"  delta (Q1+ - Q0): {delta:+.4f}", flush=True)

    print("=" * 70, flush=True)

    return {
        "overall": overall,
        "total": n,
        "sum_reward": sum(all_rewards),
        "by_step": by_step,
    }
