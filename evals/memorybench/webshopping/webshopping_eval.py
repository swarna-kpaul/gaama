"""
MemoryBench WebShopping Evaluation Module
==========================================
Two evaluation classes:
  - WebShoppingSDKEval      GAAMA memory SDK retrieval-augmented evaluation
  - WebShoppingBaselineEval  Full chat-history baseline (perfect memory upper bound)

Shared utilities:
  - load_dataset(path)       Load data.jsonl
  - compute_summary(results) Per-domain, per-step, overall reward summary
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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[dict]:
    """Load data.jsonl — list of dicts with keys: id, questions, answers, category."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_domain(category: str) -> str:
    """'baking_item_0' -> 'baking'."""
    return category.rsplit("_item_", 1)[0]


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

def _extract_global_section(question: str) -> str:
    """Extract the global rules preamble (everything before the --- separator)."""
    parts = re.split(r"-{10,}", question)
    return parts[0].strip() if len(parts) >= 2 else ""


def _extract_question_body(question: str) -> str:
    """Extract the local product-selection section (everything after the --- separator)."""
    parts = re.split(r"-{10,}", question)
    return parts[-1].strip() if len(parts) >= 2 else question.strip()


def _extract_product_title(question: str) -> str:
    """Extract title from '### Select Cake Base' header."""
    m = re.search(r"### (.+)", question)
    return m.group(1).strip() if m else "Unknown Product"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_attribute_reward(
    ground_truth: dict,
    hypothesis: str,
    llm,
    eval_model: str = EVAL_MODEL,
) -> Dict[str, Any]:
    """LLM judge: what fraction of GT attributes are covered by the hypothesis?"""
    attrs_text = ", ".join(ground_truth["attributes"])
    if not hypothesis:
        return {"reward": 0.0, "justification": "No hypothesis provided."}

    prompt = (
        "You are evaluating a product selection. Given ground truth product attributes "
        "and a model's selected product description, determine what fraction of the "
        "ground truth attributes are present or implied in the selection.\n\n"
        f"Ground truth attributes: {attrs_text}\n\n"
        f"Model's selection: {hypothesis}\n\n"
        "Score 1.0 if all attributes match, 0.0 if none match. Score proportionally "
        "for partial matches (e.g. 2 of 4 attributes found = 0.5).\n\n"
        "IMPORTANT: The model only needs to have selected the correct product. "
        "Check if the product described in the selection matches the ground truth "
        "attributes. Do NOT penalize for extra information.\n\n"
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


def _heuristic_option_match(ground_truth: dict, hypothesis: str) -> bool:
    """Check if >= 50% of GT attributes appear in the hypothesis text."""
    if not hypothesis:
        return False
    hyp_lower = hypothesis.lower()
    matched = sum(1 for a in ground_truth["attributes"] if a.lower() in hyp_lower)
    return matched >= len(ground_truth["attributes"]) * 0.5


def _score_product_selection(
    ground_truth: dict,
    hypothesis: str,
    llm,
    eval_model: str = EVAL_MODEL,
) -> Dict[str, Any]:
    """Score a product selection: LLM attribute coverage + heuristic option match."""
    reward_result = _compute_attribute_reward(ground_truth, hypothesis, llm, eval_model)
    option_match = _heuristic_option_match(ground_truth, hypothesis)
    return {
        "reward": reward_result["reward"],
        "attribute_coverage": reward_result["reward"],
        "option_match": option_match,
        "justification": reward_result["justification"],
    }


# ---------------------------------------------------------------------------
# SDK factory (shared by both classes)
# ---------------------------------------------------------------------------

def _make_sdk(
    data_dir: Path = DATA_DIR,
    llm_model: str = LLM_MODEL,
    db_name: str = DB_NAME,
) -> AgenticMemorySDK:
    """Create a GAAMA SDK instance."""
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
    """Extract the LLM adapter from the SDK instance."""
    llm = getattr(sdk._orchestrator._extractor, "_llm", None)
    if llm is None:
        raise RuntimeError("No LLM adapter on extractor. Ensure OPENAI_API_KEY is set.")
    return llm


# ---------------------------------------------------------------------------
# SDK Evaluation
# ---------------------------------------------------------------------------

class WebShoppingSDKEval:
    """Evaluate WebShopping entries using GAAMA memory SDK retrieval.

    Flow per entry (6 sequential questions):
      Q0: global + local -> LLM -> hypothesis
      Q1+: retrieve memory from local question, global + memory + local -> LLM -> hypothesis
      After every question: ingest local question + reference answer into LTM
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
        """Evaluate one entry (6 sequential product selections) using SDK retrieval."""
        category = entry["category"]
        questions = entry["questions"]
        answers = entry["answers"]

        entry_db = f"entry_{category}_{self.db_name}"
        sdk = _make_sdk(self.data_dir, llm_model=self.llm_model, db_name=entry_db)
        llm = _get_llm(sdk)

        results = []
        base_ts = datetime(2025, 1, 1, 12, 0, 0)

        for step_idx in range(len(questions)):
            question = questions[step_idx]
            answer = answers[step_idx]

            global_section = _extract_global_section(question)
            local_section = _extract_question_body(question)
            title = _extract_product_title(question)

            # --- Retrieve memory and build prompt ---
            retrieved_text = ""

            if step_idx == 0:
                prompt = f"{global_section}\n\n{local_section}"
            else:
                pack = sdk.retrieve(
                    local_section,
                    filters=QueryFilters(agent_id=AGENT_ID, user_id=USER_ID, task_id=category),
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
                        f"{global_section}\n\n"
                        f"## Retrieved Memory\n{retrieved_text}\n\n"
                        f"Question:\n"
                        f"{local_section}"
                    )
                else:
                    prompt = f"{global_section}\n\n{local_section}"

            # --- Generate hypothesis ---
            hypothesis = (
                llm.complete(prompt, max_tokens=512, model=self.eval_model, temperature=0) or ""
            ).strip()

            # --- Score ---
            scoring = _score_product_selection(answer, hypothesis, llm, self.eval_model)

            results.append({
                "entry_id": entry["id"],
                "category": category,
                "domain": extract_domain(category),
                "step_idx": step_idx,
                "product_title": title,
                "ground_truth_asin": answer["target_asin"],
                "ground_truth_attributes": answer["attributes"],
                "hypothesis": hypothesis,
                "reward": scoring["reward"],
                "attribute_coverage": scoring["attribute_coverage"],
                "option_match": scoring["option_match"],
                "justification": scoring["justification"],
                "retrieved_memory": retrieved_text,
                "memory_needed": step_idx > 0,
            })

            marker = "+" if scoring["reward"] > 0.5 else "~" if scoring["reward"] > 0 else "-"
            print(
                f"  {marker} {category} step{step_idx} ({title}): "
                f"reward={scoring['reward']:.2f}  match={scoring['option_match']}",
                flush=True,
            )

            # --- Ingest local question + reference answer into memory ---
            attrs_text = ", ".join(answer["attributes"])
            event = TraceEvent(
                event_id=f"{category}_step{step_idx}",
                event_type="message",
                actor="shopping_agent",
                content=(
                    f"{local_section}\n\n"
                    f"Selected product Attributes: \n"
                    f"{attrs_text}"
                ),
                ts=base_ts + timedelta(minutes=step_idx),
                metadata={},
            )
            sdk.ingest([event])
            sdk.create(CreateOptions(user_id=USER_ID, task_id=category))

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
                pool.submit(self.evaluate_entry, entry): entry["category"]
                for entry in dataset
            }
            for fut in as_completed(futures):
                cat = futures[fut]
                try:
                    entry_results = fut.result()
                    all_results.extend(entry_results)
                    avg_r = sum(r["reward"] for r in entry_results) / len(entry_results)
                    print(f"[done] {cat}: avg_reward={avg_r:.3f}", flush=True)
                except Exception:
                    logger.error("Failed: %s", cat, exc_info=True)
                    print(f"ERROR: {cat} failed (see log)", flush=True)

        elapsed = time.time() - t0
        print(f"\nSDK Eval done. {len(all_results)} results in {elapsed:.1f}s", flush=True)

        if all_results:
            save_results_jsonl(all_results, self.output_dir / "evaluation_log.jsonl")

        return all_results


# ---------------------------------------------------------------------------
# Baseline Evaluation
# ---------------------------------------------------------------------------

class WebShoppingBaselineEval:
    """Evaluate WebShopping entries using full chat history (perfect memory).

    Flow per entry (6 sequential questions):
      Q0: global + local -> LLM -> hypothesis
      Q1+: global + chat history (all prior local Q&A pairs) + local -> LLM -> hypothesis
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
        category = entry["category"]
        questions = entry["questions"]
        answers = entry["answers"]

        sdk = _make_sdk(self.data_dir, llm_model=self.llm_model, db_name=self.db_name)
        llm = _get_llm(sdk)

        results = []
        history: List[Dict[str, str]] = []

        for step_idx in range(len(questions)):
            question = questions[step_idx]
            answer = answers[step_idx]

            global_section = _extract_global_section(question)
            local_section = _extract_question_body(question)
            title = _extract_product_title(question)

            # --- Build prompt ---
            if step_idx == 0:
                prompt = f"{global_section}\n\n{local_section}"
            else:
                history_lines = []
                for h in history:
                    history_lines.append(f"Q: {h['question']}\nA: {h['answer']}")
                history_block = "\n\n".join(history_lines)

                prompt = (
                    f"{global_section}\n\n"
                    f"## Chat History\n{history_block}\n\n"
                    f"Question:\n"
                    f"{local_section}"
                )

            # --- Generate hypothesis ---
            hypothesis = (
                llm.complete(prompt, max_tokens=512, model=self.eval_model, temperature=0) or ""
            ).strip()

            # --- Score ---
            scoring = _score_product_selection(answer, hypothesis, llm, self.eval_model)

            results.append({
                "entry_id": entry["id"],
                "category": category,
                "domain": extract_domain(category),
                "step_idx": step_idx,
                "product_title": title,
                "ground_truth_asin": answer["target_asin"],
                "ground_truth_attributes": answer["attributes"],
                "hypothesis": hypothesis,
                "reward": scoring["reward"],
                "attribute_coverage": scoring["attribute_coverage"],
                "option_match": scoring["option_match"],
                "justification": scoring["justification"],
                "retrieved_memory": "",
                "memory_needed": step_idx > 0,
            })

            marker = "+" if scoring["reward"] > 0.5 else "~" if scoring["reward"] > 0 else "-"
            print(
                f"  {marker} {category} step{step_idx} ({title}): "
                f"reward={scoring['reward']:.2f}  match={scoring['option_match']}",
                flush=True,
            )

            # --- Add to history for subsequent steps ---
            attrs_text = ", ".join(answer["attributes"])
            history.append({
                "question": local_section,
                "answer": f"Selected Attributes: {attrs_text}",
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
                pool.submit(self.evaluate_entry, entry): entry["category"]
                for entry in dataset
            }
            for fut in as_completed(futures):
                cat = futures[fut]
                try:
                    entry_results = fut.result()
                    all_results.extend(entry_results)
                    avg_r = sum(r["reward"] for r in entry_results) / len(entry_results)
                    print(f"[done] {cat}: avg_reward={avg_r:.3f}", flush=True)
                except Exception:
                    logger.error("Failed: %s", cat, exc_info=True)
                    print(f"ERROR: {cat} failed (see log)", flush=True)

        elapsed = time.time() - t0
        print(f"\nBaseline Eval done. {len(all_results)} results in {elapsed:.1f}s", flush=True)

        if all_results:
            save_results_jsonl(all_results, self.output_dir / "evaluation_log.jsonl")

        return all_results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results_jsonl(results: List[Dict[str, Any]], path: Path) -> None:
    """Save results to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved to {path}", flush=True)


# ---------------------------------------------------------------------------
# Summary computation (works for both SDK and baseline results)
# ---------------------------------------------------------------------------

def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-domain, per-step, and overall reward/accuracy summary."""
    if not results:
        print("No results to summarize.", flush=True)
        return {"overall": 0.0, "total": 0}

    all_rewards = [r["reward"] for r in results]
    overall = sum(all_rewards) / len(all_rewards)

    domain_rewards: Dict[str, List[float]] = defaultdict(list)
    step_rewards: Dict[int, List[float]] = defaultdict(list)
    memory_rewards: Dict[bool, List[float]] = defaultdict(list)

    for r in results:
        domain_rewards[r["domain"]].append(r["reward"])
        step_rewards[r["step_idx"]].append(r["reward"])
        memory_rewards[r["memory_needed"]].append(r["reward"])

    n = len(all_rewards)
    correct = sum(1 for r in all_rewards if r > 0.5)
    match_rate = sum(1 for r in results if r["option_match"]) / n

    print("\n" + "=" * 70, flush=True)
    print("REWARD SUMMARY — WebShopping", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  OVERALL: n={n}  reward={overall:.4f}  acc={correct/n:.4f}  option_match={match_rate:.4f}", flush=True)

    print(f"\n{'--- By Domain ---':^70}", flush=True)
    by_domain = {}
    for domain in sorted(domain_rewards):
        rw = domain_rewards[domain]
        avg = sum(rw) / len(rw)
        c = sum(1 for r in rw if r > 0.5)
        print(f"  {domain:15s}  n={len(rw):4d}  reward={avg:.4f}  acc={c/len(rw):.4f}", flush=True)
        by_domain[domain] = {"avg_reward": avg, "total": len(rw), "correct": c}

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
        "by_domain": by_domain,
        "by_step": by_step,
    }
