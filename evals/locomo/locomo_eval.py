"""
LoCoMo Evaluation Module
=========================
Core evaluation functions for the LoCoMo benchmark. All functions import from
gaama.* and are designed to be called from the run_*.py CLI scripts.

Functions:
  - load_dataset(path)              Load locomo10.json
  - answer_and_judge(...)           Generate hypothesis from memory, then judge
  - evaluate_sample(...)            Evaluate all questions for one sample
  - compute_summary(results)        Per-category and per-sample reward summary
  - save_results_jsonl(results, p)  Save results to JSONL
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# SDK path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent  # gaama/evals/locomo -> gaama/evals -> gaama -> agentic-memory
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gaama.api import AgenticMemorySDK  # noqa: E402
from gaama.core import (  # noqa: E402
    MemoryPack,
    QueryFilters,
    RetrievalBudget,
)
from gaama.services import answer_from_memory  # noqa: E402

from config import CATEGORY_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
USER_ID = "locomo"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> List[dict]:
    """Load locomo10.json and return the list of sample dicts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return data


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_llm(sdk: AgenticMemorySDK):
    """Extract the LLM adapter from the SDK instance."""
    orch = sdk._orchestrator
    llm = getattr(orch._extractor, "_llm", None)
    if llm is None:
        raise RuntimeError("No LLM adapter on extractor. Ensure OPENAI_API_KEY is set.")
    return llm


# ---------------------------------------------------------------------------
# Memory trimming
# ---------------------------------------------------------------------------

def _trim_memory_pack(
    pack: MemoryPack,
    max_words: int,
    budget: Optional[RetrievalBudget] = None,
) -> MemoryPack:
    """Return a new MemoryPack trimmed so that to_text() stays under *max_words*.

    Removal is distributed proportionally across categories according to their
    budget caps so that no single category is drained before others.
    """
    categories = ["facts", "reflections", "skills", "episodes"]

    items: Dict[str, List[str]] = {
        cat: list(getattr(pack, cat)) for cat in categories
    }

    def _word_count() -> int:
        return sum(len(s.split()) for cat in categories for s in items[cat])

    if _word_count() <= max_words:
        return pack

    budget_map = {
        "facts": budget.max_facts if budget else 1,
        "reflections": budget.max_reflections if budget else 1,
        "skills": budget.max_skills if budget else 1,
        "episodes": budget.max_episodes if budget else 1,
    }
    total_budget = sum(budget_map.values()) or 1

    score_lists: Dict[str, List[float]] = {}
    for cat in categories:
        if pack.scores and cat in pack.scores:
            score_lists[cat] = list(pack.scores[cat])
        else:
            score_lists[cat] = []

    word_counts: Dict[str, List[int]] = {
        cat: [len(s.split()) for s in items[cat]] for cat in categories
    }

    while _word_count() > max_words:
        non_empty = [cat for cat in categories if items[cat]]
        if not non_empty:
            break

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


# ---------------------------------------------------------------------------
# Answer generation and judging
# ---------------------------------------------------------------------------

def answer_and_judge(
    question: str,
    answer: str,
    retrieved_memory: MemoryPack,
    llm,
    eval_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Generate a hypothesis from retrieved memory, then judge with fractional evaluator.

    Parameters
    ----------
    question : The question being evaluated.
    answer   : Ground-truth reference answer.
    retrieved_memory : MemoryPack from sdk.retrieve().
    llm      : LLM adapter (from _get_llm).
    eval_model : Model name for both answer generation and judging.

    Returns
    -------
    Dict with keys: hypothesis, reward, justification.
    """
    # Step 1: Generate hypothesis from memory
    hypothesis = answer_from_memory(
        question, retrieved_memory, llm=llm, model=eval_model, temperature=0,
    )
    if not hypothesis:
        hypothesis = "(No answer generated)"

    # Step 2: Judge with fractional evaluator
    reward_result = _compute_hypothesis_reward(
        question, answer, hypothesis, llm, llm_model=eval_model, temperature=0,
    )

    return {
        "hypothesis": hypothesis,
        "reward": reward_result["reward"],
        "justification": reward_result["justification"],
    }


def _compute_hypothesis_reward(
    question: str,
    answer: str,
    hypothesis: str,
    llm,
    llm_model: str = "gpt-4o-mini",
    temperature: float | None = None,
) -> Dict[str, Any]:
    """Fractional coverage evaluator (0.0-1.0).

    Scores based on what fraction of reference answer key facts are present
    in the generated response.
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

    _MAX_RETRIES = 3
    last_raw = ""
    last_error = ""

    for attempt in range(_MAX_RETRIES):
        try:
            if attempt == 0:
                raw = llm.complete(prompt, max_tokens=400, model=llm_model, temperature=temperature)
            else:
                retry_prompt = (
                    f"Your previous response could not be parsed as valid JSON.\n\n"
                    f"Error: {last_error}\n\n"
                    f"Your previous output:\n{last_raw[:500]}\n\n"
                    f"Please return ONLY valid JSON with no markdown fences, no extra text.\n"
                    f'Required format: {{"reward": <float 0.0-1.0>, "justification": "<brief explanation>"}}'
                )
                raw = llm.complete(retry_prompt, max_tokens=400, model=llm_model, temperature=temperature)
            raw = (raw or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
                raw = raw.strip()
            parsed = json.loads(raw)
            reward = float(parsed.get("reward", 0.0))
            justification = parsed.get("justification", "")
            reward = max(0.0, min(1.0, reward))
            return {"reward": round(reward, 2), "justification": justification}
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as e:
            last_raw = raw if raw else ""
            last_error = str(e)
            logger.warning(
                "Hypothesis reward JSON parse failed (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES, e,
            )
        except Exception as e:
            logger.warning("Hypothesis reward LLM judge failed (model=%s): %s", llm_model, e)
            return {"reward": 0.0, "justification": f"LLM judge failed: {e}"}

    logger.warning("All %d hypothesis reward attempts failed. Returning 0.", _MAX_RETRIES)
    return {"reward": 0.0, "justification": f"JSON parse failed after {_MAX_RETRIES} attempts: {last_error}"}


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    sdk: AgenticMemorySDK,
    data_path: Path,
    sample_id: str,
    categories: Optional[List[int]] = None,
    budget: Optional[RetrievalBudget] = None,
    max_memory_words: int = 1000,
    ppr_weight: float = 0.0,
    sim_weight: float = 1.0,
    semantic_only: bool = True,
    expansion_depth: Optional[int] = None,
    eval_model: str = "gpt-4o-mini",
    max_workers: int = 20,
) -> List[Dict[str, Any]]:
    """Evaluate all questions for one sample, returns list of result dicts.

    Parameters
    ----------
    sdk             : GAAMA SDK instance (with LTM already created).
    data_path       : Path to locomo10.json.
    sample_id       : The sample to evaluate (e.g. "conv-26").
    categories      : Question categories to include (default: [1,2,3,4]).
    budget          : Retrieval budget.
    max_memory_words: Trim retrieved memory to at most this many words.
    ppr_weight      : Weight for PPR score in retrieval.
    sim_weight      : Weight for similarity score in retrieval.
    semantic_only   : If True, use pure semantic (KNN) retrieval.
    expansion_depth : PPR expansion depth (None = default).
    eval_model      : LLM model for answer generation and judging.
    max_workers     : Max parallel threads for question evaluation.

    Returns
    -------
    List of result dicts, one per question.
    """
    if categories is None:
        categories = [1, 2, 3, 4]
    if budget is None:
        budget = RetrievalBudget()

    # Load dataset and find the sample
    dataset = load_dataset(data_path)
    sample = None
    for item in dataset:
        if item.get("sample_id") == sample_id:
            sample = item
            break
    if sample is None:
        print(f"  [{sample_id}] Sample not found in dataset, skipping.", flush=True)
        return []

    # Filter questions
    qa_list = sample.get("qa", [])
    qa_list = [q for q in qa_list if q.get("category") in categories and q.get("answer")]
    if not qa_list:
        print(f"  [{sample_id}] No matching questions found.", flush=True)
        return []

    llm = _get_llm(sdk)
    total_q = len(qa_list)
    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    t0 = time.time()

    def _evaluate_one(q_idx: int, qa: Dict[str, Any]) -> Dict[str, Any]:
        question = qa.get("question", "")
        ground_truth = str(qa.get("answer", ""))
        category = qa.get("category", 0)
        cat_name = CATEGORY_NAMES.get(category, f"cat{category}_unknown")

        if not question:
            return {}

        # Retrieve
        pack = sdk.retrieve(
            question,
            filters=QueryFilters(agent_id=AGENT_ID, user_id=USER_ID, task_id=sample_id),
            budget=budget,
            sources="ltm",
            ppr_score_weight=ppr_weight,
            sim_score_weight=sim_weight,
            semantic_only=semantic_only,
            expansion_depth=expansion_depth,
            max_memory_words=max_memory_words,
        )

        # Trim
        if max_memory_words >= 0:
            pack = _trim_memory_pack(pack, max_memory_words, budget=budget)

        retrieved_text = pack.to_text(include_citations=False).strip()

        # Answer and judge
        result = answer_and_judge(question, ground_truth, pack, llm, eval_model=eval_model)

        entry = {
            "sample_id": sample_id,
            "question_idx": q_idx,
            "category": category,
            "question_type": cat_name,
            "question": question,
            "answer": ground_truth,
            "hypothesis": result["hypothesis"],
            "reward": result["reward"],
            "justification": result["justification"],
            "retrieved_memory": retrieved_text,
        }

        marker = "+" if result["reward"] > 0.5 else "~" if result["reward"] > 0 else "-"
        print(
            f"  [{q_idx + 1}/{total_q}] {marker} {sample_id} {cat_name}: "
            f"reward={result['reward']:.2f}  retrieved={len(retrieved_text)} chars",
            flush=True,
        )
        return entry

    # Parallel evaluation
    workers = min(max_workers, total_q)
    print(f"  [{sample_id}] Evaluating {total_q} questions with {workers} workers...", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_evaluate_one, q_idx, qa): q_idx
            for q_idx, qa in enumerate(qa_list)
        }
        for fut in as_completed(futures):
            q_idx = futures[fut]
            try:
                entry = fut.result()
                if entry:
                    with results_lock:
                        results.append(entry)
            except Exception as e:
                print(f"  [{sample_id}:{q_idx}] ERROR: {e}", flush=True)
                logger.error("Failed evaluation for %s:%d", sample_id, q_idx, exc_info=True)

    elapsed = time.time() - t0
    n = len(results)
    reward = sum(r["reward"] for r in results) / n if n else 0
    correct = sum(1 for r in results if r.get("reward", 0) >= 1.0)
    acc = correct / n * 100 if n else 0
    print(
        f"  [{sample_id}] Done: {n} questions, reward={reward:.4f}, "
        f"acc={acc:.1f}%, elapsed={elapsed:.1f}s",
        flush=True,
    )
    return results


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-category and per-sample reward/accuracy summary.

    Returns a summary dict and prints formatted tables.
    """
    if not results:
        print("No results to summarize.", flush=True)
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
        correct = sum(1 for v in vals if v >= 1.0)
        by_category[cat] = {
            "avg_reward": avg,
            "sum_reward": sum(vals),
            "total": len(vals),
            "correct": correct,
            "accuracy": correct / len(vals) if vals else 0.0,
        }

    by_sample = {}
    for sid in sorted(sample_rewards.keys()):
        vals = sample_rewards[sid]
        avg = sum(vals) / len(vals) if vals else 0.0
        correct = sum(1 for v in vals if v >= 1.0)
        by_sample[sid] = {
            "avg_reward": avg,
            "sum_reward": sum(vals),
            "total": len(vals),
            "correct": correct,
            "accuracy": correct / len(vals) if vals else 0.0,
        }

    summary = {
        "overall": overall,
        "total": len(all_rewards),
        "sum_reward": sum(all_rewards),
        "by_category": by_category,
        "by_sample": by_sample,
    }

    # Print reward summary by category
    print("\n" + "=" * 70, flush=True)
    print("REWARD SUMMARY -- BY CATEGORY", flush=True)
    print("=" * 70, flush=True)
    for cat, info in by_category.items():
        print(
            f"  {cat:30s}: reward={info['avg_reward']:.4f} "
            f"({info['sum_reward']:.2f}/{info['total']})  "
            f"acc={info['accuracy']:.1%} ({info['correct']}/{info['total']})",
            flush=True,
        )
    total_correct = sum(info["correct"] for info in by_category.values())
    total_n = sum(info["total"] for info in by_category.values())
    print(
        f"\n  {'OVERALL':30s}: reward={overall:.4f} "
        f"({sum(all_rewards):.2f}/{len(all_rewards)})  "
        f"acc={total_correct/total_n:.1%} ({total_correct}/{total_n})" if total_n else "",
        flush=True,
    )

    # Print reward summary by sample
    print("\n" + "=" * 70, flush=True)
    print("REWARD SUMMARY -- BY SAMPLE", flush=True)
    print("=" * 70, flush=True)
    for sid, info in by_sample.items():
        print(
            f"  {sid:30s}: reward={info['avg_reward']:.4f} "
            f"({info['sum_reward']:.2f}/{info['total']})  "
            f"acc={info['accuracy']:.1%} ({info['correct']}/{info['total']})",
            flush=True,
        )
    print("=" * 70, flush=True)

    return summary


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results_jsonl(results: List[Dict[str, Any]], path: Path) -> None:
    """Save results to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "sample_id": r["sample_id"],
                "question_idx": r.get("question_idx", -1),
                "category": r["category"],
                "question_type": r["question_type"],
                "question": r["question"],
                "answer": r["answer"],
                "hypothesis": r["hypothesis"],
                "reward": r["reward"],
                "justification": r.get("justification", ""),
                "retrieved_memory": r.get("retrieved_memory", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Results saved to {path}", flush=True)
