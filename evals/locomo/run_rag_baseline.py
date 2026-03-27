"""
RAG Baseline Evaluation for LoCoMo Dataset.
=============================================
A self-contained RAG baseline that uses OpenAI embeddings + cosine similarity
retrieval to answer LoCoMo questions. Two-step process: ingest+index, then
retrieve+evaluate.

Usage:
    python run_rag_baseline.py --step 1                          # ingest & index
    python run_rag_baseline.py --step 2                          # retrieve & evaluate
    python run_rag_baseline.py --step all                        # both steps
    python run_rag_baseline.py --step all --sample-ids conv-26   # specific sample
    python run_rag_baseline.py --step 2 --max-words 800          # custom word limit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai

# ---------------------------------------------------------------------------
# SDK path setup (for config only, RAG baseline is self-contained)
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent  # gaama/evals/locomo -> gaama/evals -> gaama -> agentic-memory
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_FILE_DIR))

from config import (  # noqa: E402
    SAMPLE_IDS, MAX_MEMORY_WORDS, LLM_MODEL, EVAL_MODEL,
    EMBEDDING_MODEL, MAX_WORKERS, CATEGORY_NAMES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = _FILE_DIR / "locomo10.json"
INDEX_DIR = _FILE_DIR / "data" / "rag_index"
OUTPUT_DIR = _FILE_DIR / "data" / "results"

# ---------------------------------------------------------------------------
# Prompts (same as locomo_pipeline.py / rag_baseline.py)
# ---------------------------------------------------------------------------

HYPOTHESIS_PROMPT = """\
# Answer from memory

You are a precise answer assistant. Given a **query** and the **retrieved memory** below, answer the query using the provided memory.

## Query

{question}

## Retrieved memory

{retrieved_text}

## Instructions

- Answer the query in one or two short paragraphs. Be direct and specific.
- Extract concrete answers from the memory even if the information is scattered across multiple items. Synthesize and combine partial evidence.
- When counting occurrences (e.g., "how many times"), carefully scan ALL memory items and count each distinct instance.
- When listing items (e.g., "which cities"), exhaustively list EVERY item mentioned across all memory entries.
- Prefer giving a direct answer over saying "the memory does not specify." If the memory contains relevant clues, use them to form a best-effort answer.
- Do not repeat the query. Do not cite section headers; use the memory content naturally."""


FRACTIONAL_JUDGE_PROMPT = """\
You are an evaluator. Given a question, a correct reference answer, and a \
generated response (hypothesis), determine what fraction of the reference answer \
is present in the generated response.

IMPORTANT: Only check whether the key facts from the reference answer appear in \
the generated response. Do NOT penalize the response for containing extra \
information, additional details, or tangential content beyond the reference answer. \
The ONLY thing that matters is whether the reference answer's key facts are covered.

Scoring guidelines:
- 1.0: All key facts and details from the reference answer are present in the \
generated response (even if the response also contains extra information).
- 0.0: None of the key facts from the reference answer appear in the generated \
response.
- Between 0.0 and 1.0: Some key facts from the reference answer are present. \
Score = (number of reference answer key facts found) / (total key facts in \
reference answer). For example, if the answer has 3 key facts and 2 are found \
in the response, score = 0.67.

You MUST respond in the following JSON format (no markdown, no extra text):
{{"reward": <float between 0.0 and 1.0>, "justification": "<brief explanation \
of which reference answer facts are present and which are missing>"}}

Question: {question}

Correct Reference Answer: {answer}

Generated Response: {hypothesis}

Evaluate ONLY the coverage of the reference answer's facts. Do NOT reduce the \
score for extra information. Respond with the JSON only."""


# ---------------------------------------------------------------------------
# OpenAI client helpers
# ---------------------------------------------------------------------------

def _get_client() -> openai.OpenAI:
    """Create an OpenAI client from the OPENAI_API_KEY env var."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr, flush=True)
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def _embed_texts(
    client: openai.OpenAI, texts: List[str], batch_size: int = 200,
) -> List[List[float]]:
    """Compute embeddings for a list of texts, batched."""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embs = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_embs)
    return all_embeddings


def _embed_single(client: openai.OpenAI, text: str) -> List[float]:
    """Compute embedding for a single text."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def _chat_completion(client: openai.OpenAI, prompt: str, model: str = LLM_MODEL) -> str:
    """Run a chat completion at temperature=0."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load locomo10.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_sorted_sessions(conversation: dict) -> List[Tuple[int, list, str]]:
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


def _extract_documents(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract all conversation turns from a sample into indexable documents."""
    conversation = sample.get("conversation", {})
    sessions = _get_sorted_sessions(conversation)
    documents = []

    for _idx, turns, date_str in sessions:
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            if not text:
                continue

            doc_text = f"{speaker}: {text}"
            if date_str:
                doc_text += f" [date: {date_str}]"

            blip_caption = turn.get("blip_caption")
            if blip_caption:
                doc_text += f" [image: {blip_caption}]"

            documents.append({
                "text": doc_text,
                "session_date": date_str,
            })

    return documents


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Step 1: Ingest & Index
# ---------------------------------------------------------------------------

def ingest_and_index(
    data: List[Dict[str, Any]],
    sample_ids: Optional[List[str]] = None,
) -> None:
    """Build embedding index for each sample. Skips if index file already exists."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    for sample in data:
        sid = sample["sample_id"]
        if sample_ids and sid not in sample_ids:
            continue

        index_path = INDEX_DIR / f"{sid}.json"
        if index_path.exists():
            print(f"  [{sid}] Index already exists, skipping.", flush=True)
            continue

        docs = _extract_documents(sample)
        if not docs:
            print(f"  [{sid}] No documents found, skipping.", flush=True)
            continue

        print(f"  [{sid}] Indexing {len(docs)} documents...", flush=True)

        texts = [d["text"] for d in docs]
        embeddings = _embed_texts(client, texts)

        index_entries = []
        for doc, emb in zip(docs, embeddings):
            index_entries.append({
                "text": doc["text"],
                "embedding": emb,
                "session_date": doc["session_date"],
            })

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_entries, f, ensure_ascii=False)

        print(f"  [{sid}] Saved {len(index_entries)} entries to {index_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Step 2: Retrieve & Evaluate
# ---------------------------------------------------------------------------

def _retrieve(
    query_embedding: List[float],
    index: List[Dict[str, Any]],
    max_words: int,
) -> Tuple[str, int]:
    """Retrieve top documents by cosine similarity until the word budget is filled."""
    scored = []
    for entry in index:
        sim = _cosine_similarity(query_embedding, entry["embedding"])
        scored.append((sim, entry["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    total_words = 0
    for _sim, text in scored:
        word_count = len(text.split())
        if total_words + word_count > max_words and selected:
            break
        selected.append(text)
        total_words += word_count

    formatted = "## Retrieved Conversations\n" + "\n".join(f"- {t}" for t in selected)
    return formatted, total_words


def _evaluate_single(
    client: openai.OpenAI,
    sample_id: str,
    qa: Dict[str, Any],
    index: List[Dict[str, Any]],
    max_words: int,
    eval_model: str = EVAL_MODEL,
) -> Dict[str, Any]:
    """Evaluate a single question: retrieve, generate hypothesis, judge."""
    question = qa["question"]
    answer = str(qa["answer"])
    category = qa.get("category", 0)
    cat_name = CATEGORY_NAMES.get(category, f"cat{category}_unknown")

    # Embed query
    query_emb = _embed_single(client, question)

    # Retrieve
    retrieved_text, retrieved_words = _retrieve(query_emb, index, max_words)

    # Generate hypothesis
    hyp_prompt = HYPOTHESIS_PROMPT.format(question=question, retrieved_text=retrieved_text)
    hypothesis = _chat_completion(client, hyp_prompt, model=eval_model)

    # Judge (fractional evaluator)
    judge_prompt = FRACTIONAL_JUDGE_PROMPT.format(
        question=question, answer=answer, hypothesis=hypothesis,
    )
    judge_resp = client.chat.completions.create(
        model=eval_model, temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )
    judge_response = judge_resp.choices[0].message.content.strip()
    reward = 0.0
    reason = ""
    try:
        cleaned = judge_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        reward = float(parsed.get("reward", parsed.get("score", 0.0)))
        reward = max(0.0, min(1.0, reward))
        reason = parsed.get("justification", judge_response[:200])
    except (json.JSONDecodeError, ValueError, TypeError):
        reward = 0.0
        reason = f"parse_error: {judge_response[:200]}"

    return {
        "sample_id": sample_id,
        "category": category,
        "question_type": cat_name,
        "question": question,
        "answer": answer,
        "hypothesis": hypothesis,
        "retrieved_text": retrieved_text,
        "reward": reward,
        "justification": reason,
        "retrieved_words": retrieved_words,
    }


def retrieve_and_evaluate(
    data: List[Dict[str, Any]],
    sample_ids: Optional[List[str]] = None,
    max_words: int = MAX_MEMORY_WORDS,
    max_workers: int = MAX_WORKERS,
    eval_model: str = EVAL_MODEL,
) -> List[Dict[str, Any]]:
    """Run retrieval + evaluation for all questions across selected samples."""
    client = _get_client()
    all_results: List[Dict[str, Any]] = []

    for sample in data:
        sid = sample["sample_id"]
        if sample_ids and sid not in sample_ids:
            continue

        index_path = INDEX_DIR / f"{sid}.json"
        if not index_path.exists():
            print(f"  [{sid}] No index file found, skipping. Run --step 1 first.", flush=True)
            continue

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        qa_list = sample.get("qa", [])
        qa_list = [q for q in qa_list if q.get("category") in CATEGORY_NAMES and q.get("answer")]

        if not qa_list:
            print(f"  [{sid}] No matching questions.", flush=True)
            continue

        print(f"\n  [{sid}] Evaluating {len(qa_list)} questions...", flush=True)

        # Build work items for this sample
        results: List[Dict[str, Any]] = []
        completed = 0
        total = len(qa_list)
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for qa in qa_list:
                future = executor.submit(
                    _evaluate_single, client, sid, qa, index, max_words, eval_model,
                )
                futures[future] = qa["question"][:60]

            for future in as_completed(futures):
                completed += 1
                q_preview = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    marker = "+" if result["reward"] > 0.5 else "~" if result["reward"] > 0 else "-"
                    print(
                        f"  [{completed}/{total}] {marker} {sid} | "
                        f"reward={result['reward']:.2f} | "
                        f"{result['question_type']} | "
                        f"{q_preview}... ({rate:.1f} q/s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [{completed}/{total}] ERROR {sid}: {e}", file=sys.stderr, flush=True)

        # Filter to only this sample (safety)
        results = [r for r in results if r.get("sample_id") == sid]

        # Save per-sample JSONL
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{sid}_rag.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                row = {
                    "sample_id": r["sample_id"],
                    "category": r["category"],
                    "question_type": r["question_type"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "hypothesis": r["hypothesis"],
                    "retrieved_text": r["retrieved_text"],
                    "reward": r["reward"],
                    "justification": r.get("justification", ""),
                    "retrieved_words": r.get("retrieved_words", 0),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{sid}] Saved {len(results)} results to {out_path.name}", flush=True)

        n = len(results)
        reward = sum(r["reward"] for r in results) / n if n else 0
        correct = sum(1 for r in results if r.get("reward", 0) >= 1.0)
        acc = correct / n * 100 if n else 0
        print(f"  [{sid}] reward={reward:.4f}, acc={acc:.1f}%", flush=True)

        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print per-category and per-sample reward summary."""
    if not results:
        print("No results to summarize.", flush=True)
        return

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

    print("\n" + "=" * 70, flush=True)
    print("REWARD SUMMARY -- BY CATEGORY", flush=True)
    print("=" * 70, flush=True)
    for cat in sorted(cat_rewards.keys()):
        vals = cat_rewards[cat]
        avg = sum(vals) / len(vals) if vals else 0.0
        correct = sum(1 for v in vals if v >= 1.0)
        print(
            f"  {cat:30s}: reward={avg:.4f} ({sum(vals):.2f}/{len(vals)})  "
            f"acc={correct/len(vals)*100:.1f}% ({correct}/{len(vals)})",
            flush=True,
        )

    total_correct = sum(1 for v in all_rewards if v >= 1.0)
    print(
        f"\n  {'OVERALL':30s}: reward={overall:.4f} "
        f"({sum(all_rewards):.2f}/{len(all_rewards)})  "
        f"acc={total_correct/len(all_rewards)*100:.1f}% ({total_correct}/{len(all_rewards)})",
        flush=True,
    )

    print("\n" + "=" * 70, flush=True)
    print("REWARD SUMMARY -- BY SAMPLE", flush=True)
    print("=" * 70, flush=True)
    for sid in sorted(sample_rewards.keys()):
        vals = sample_rewards[sid]
        avg = sum(vals) / len(vals) if vals else 0.0
        correct = sum(1 for v in vals if v >= 1.0)
        print(
            f"  {sid:30s}: reward={avg:.4f} ({sum(vals):.2f}/{len(vals)})  "
            f"acc={correct/len(vals)*100:.1f}% ({correct}/{len(vals)})",
            flush=True,
        )
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Baseline Evaluation for LoCoMo Dataset.",
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["1", "2", "all"],
        help="1=ingest/index, 2=retrieve/evaluate, all=both.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Filter to specific sample IDs (e.g. conv-26 conv-30). Default: all.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=MAX_MEMORY_WORDS,
        help=f"Max words for retrieved context (default: {MAX_MEMORY_WORDS}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Max parallel workers (default: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=EVAL_MODEL,
        help=f"LLM model for evaluation (default: {EVAL_MODEL}).",
    )
    args = parser.parse_args()

    sample_ids = args.sample_ids if args.sample_ids else SAMPLE_IDS

    # Load data
    print(f"Loading LoCoMo data from {DATA_PATH}...", flush=True)
    data = _load_dataset(DATA_PATH)
    print(f"Loaded {len(data)} samples.", flush=True)

    if args.step in ("1", "all"):
        print("\n" + "=" * 70, flush=True)
        print("STEP 1: Ingest & Index", flush=True)
        print("=" * 70, flush=True)
        ingest_and_index(data, sample_ids=sample_ids)

    if args.step in ("2", "all"):
        print("\n" + "=" * 70, flush=True)
        print("STEP 2: Retrieve & Evaluate", flush=True)
        print("=" * 70, flush=True)
        results = retrieve_and_evaluate(
            data,
            sample_ids=sample_ids,
            max_words=args.max_words,
            max_workers=args.max_workers,
            eval_model=args.eval_model,
        )
        _print_summary(results)


if __name__ == "__main__":
    main()
