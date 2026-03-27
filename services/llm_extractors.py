"""LLM-based memory extractors using the prompt library."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence

from gaama.adapters.interfaces import LLMAdapter
from gaama.core import MemoryNode, TraceEvent
from gaama.infra.prompt_loader import load_prompt
from gaama.services.interfaces import (
    ExtractResult,
    MemoryExtractor,
)

logger = logging.getLogger(__name__)

_MAX_JSON_RETRIES = 2


def _strip_json_block(text: str) -> str:
    """Remove markdown code fences around JSON if present."""
    text = text.strip()
    for pattern in (r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", r"^```\s*\n?(.*?)\n?```\s*$"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def _retry_llm_for_json(
    llm: LLMAdapter,
    original_response: str,
    error_message: str,
    max_tokens: int,
    expected_keys: list[str] | None = None,
) -> Dict[str, Any]:
    """Retry the LLM with the error message to get a valid JSON response.

    Sends the original (broken) output back to the LLM along with the parse
    error so it can correct the JSON.  Retries up to ``_MAX_JSON_RETRIES``
    times.  Returns the parsed dict, or a dict with empty lists for each
    expected key if all retries fail.
    """
    keys_hint = ""
    if expected_keys:
        keys_hint = (
            f" The JSON must be a dict with these top-level keys: {expected_keys}."
            " Use empty arrays [] for categories with nothing to extract."
        )

    last_error = error_message
    last_raw = original_response

    for attempt in range(1, _MAX_JSON_RETRIES + 1):
        retry_prompt = (
            f"Your previous response could not be parsed as valid JSON.\n\n"
            f"Error: {last_error}\n\n"
            f"Your previous output (first 1000 chars):\n"
            f"{(last_raw or '')[:1000]}\n\n"
            f"Please return ONLY valid JSON with no markdown fences, no extra text.{keys_hint}"
        )
        logger.info("JSON retry attempt %d/%d: %s", attempt, _MAX_JSON_RETRIES, last_error)
        print(f"[JSON retry {attempt}/{_MAX_JSON_RETRIES}] {last_error}")

        try:
            raw = llm.complete(retry_prompt, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("LLM retry call failed: %s", exc)
            last_error = f"LLM call failed: {exc}"
            continue

        stripped = _strip_json_block(raw or "")
        if not stripped:
            last_error = "LLM returned empty response on retry"
            last_raw = raw or ""
            continue

        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and data:
                logger.info("JSON retry succeeded on attempt %d", attempt)
                print(f"[JSON retry {attempt}/{_MAX_JSON_RETRIES}] Success")
                return data
            last_error = "Parsed JSON is empty or not a dict"
            last_raw = raw or ""
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            last_raw = raw or ""

    logger.warning(
        "All %d JSON retries failed. Returning fallback with empty arrays.", _MAX_JSON_RETRIES
    )
    print(f"[JSON retry] All {_MAX_JSON_RETRIES} retries failed, using empty fallback.")
    if expected_keys:
        return {k: [] for k in expected_keys}
    return {}


class LLMFactExtractor:
    """Extract facts from new episodes given context of related episodes and existing facts.

    Used by the new LTM creation pipeline (Step 3).
    """

    MAX_CONTEXT_WORDS = 2000  # upper cap on total context sent to LLM

    def __init__(self, llm: LLMAdapter, max_tokens: int = 8000) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    @staticmethod
    def _budget_truncate(
        sections: List[tuple],  # [(lines, min_share), ...]
        max_words: int,
    ) -> List[List[str]]:
        """Truncate multiple sections to fit within max_words.

        Each section is (lines: List[str], min_share: float) where min_share
        is the minimum fraction of budget reserved for this section (0.0-1.0).
        Fills each section up to its reserved budget first, then distributes
        remaining budget to sections that need more.
        """
        # Calculate word counts per line per section
        section_data = []
        for lines, min_share in sections:
            word_counts = [len(line.split()) for line in lines]
            section_data.append({
                "lines": lines,
                "word_counts": word_counts,
                "min_share": min_share,
                "selected": [],
            })

        # Phase 1: Fill each section up to its reserved budget
        remaining = max_words
        for sd in section_data:
            budget = int(max_words * sd["min_share"])
            words_used = 0
            for i, (line, wc) in enumerate(zip(sd["lines"], sd["word_counts"])):
                if words_used + wc > budget:
                    break
                sd["selected"].append(line)
                words_used += wc
            remaining -= words_used

        # Phase 2: Distribute remaining budget to sections that have more lines
        for sd in section_data:
            already = len(sd["selected"])
            for i in range(already, len(sd["lines"])):
                wc = sd["word_counts"][i]
                if remaining < wc:
                    break
                sd["selected"].append(sd["lines"][i])
                remaining -= wc

        return [sd["selected"] for sd in section_data]

    def extract_facts(
        self,
        new_episodes: List[MemoryNode],
        related_episodes: List[MemoryNode],
        existing_facts: List[MemoryNode],
        existing_concepts: List[MemoryNode] | None = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract new facts and concepts from episodes.

        Returns ``(facts, concepts)`` where each is a list of dicts.
        Facts have keys: fact_text, belief, source_episode_ids, concepts.
        Concepts have keys: concept_label, episode_ids.
        """
        # Format all lines
        new_ep_lines = []
        for ep in new_episodes:
            content = (ep.summary or "").strip()
            session_date = (ep.tags or {}).get("session_date", "")
            new_ep_lines.append(f"[{ep.node_id}] [{session_date}] {content}")

        rel_ep_lines = []
        for ep in related_episodes:
            content = (ep.summary or "").strip()
            session_date = (ep.tags or {}).get("session_date", "")
            rel_ep_lines.append(f"[{ep.node_id}] [{session_date}] {content}")

        fact_lines = []
        for f in existing_facts:
            fact_lines.append(f"[{f.node_id}] {f.fact_text}")

        concept_lines = []
        for c in (existing_concepts or []):
            label = (getattr(c, "concept_label", "") or "").strip()
            if label:
                concept_lines.append(label)

        # Budget: new episodes 35%, related episodes 25%, existing facts 25%, concepts 15%
        truncated = self._budget_truncate(
            [
                (new_ep_lines, 0.35),
                (rel_ep_lines, 0.25),
                (fact_lines, 0.25),
                (concept_lines, 0.15),
            ],
            self.MAX_CONTEXT_WORDS,
        )
        new_episodes_text = "\n".join(truncated[0]) if truncated[0] else "(none)"
        related_episodes_text = "\n".join(truncated[1]) if truncated[1] else "(none)"
        existing_facts_text = "\n".join(truncated[2]) if truncated[2] else "(none)"
        existing_concepts_text = "\n".join(truncated[3]) if truncated[3] else "(none)"

        prompt = load_prompt("fact_generation", {
            "new_episodes": new_episodes_text,
            "related_episodes": related_episodes_text,
            "existing_facts": existing_facts_text,
            "existing_concepts": existing_concepts_text,
        })

        raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
        data = self._parse_response(raw)
        return data.get("facts", []), data.get("concepts", [])

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        stripped = _strip_json_block(raw or "")
        if not stripped:
            return {"facts": [], "concepts": []}
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Retry
        data = _retry_llm_for_json(
            self._llm, raw or "", "Could not parse fact extraction JSON",
            self._max_tokens, expected_keys=["facts", "concepts"],
        )
        return data


class LLMReflectionExtractor:
    """Extract reflections from new facts given context of related facts and existing reflections.

    Used by the new LTM creation pipeline (Step 4).
    """

    def __init__(self, llm: LLMAdapter, max_tokens: int = 4000) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    MAX_CONTEXT_WORDS = 2000  # upper cap on total context sent to LLM

    def extract_reflections(
        self,
        new_facts: List[MemoryNode],
        related_facts: List[MemoryNode],
        existing_reflections: List[MemoryNode],
    ) -> List[Dict[str, Any]]:
        """Extract new reflections from facts. Returns list of dicts with keys:
        reflection_text, belief, source_fact_ids.
        """
        # Format all lines
        new_fact_lines = []
        for f in new_facts:
            new_fact_lines.append(f"[{f.node_id}] {f.fact_text}")

        rel_fact_lines = []
        for f in related_facts:
            rel_fact_lines.append(f"[{f.node_id}] {f.fact_text}")

        refl_lines = []
        for r in existing_reflections:
            refl_lines.append(f"[{r.node_id}] {r.reflection_text}")

        # Budget: new facts 40%, related facts 30%, existing reflections 30%
        truncated = LLMFactExtractor._budget_truncate(
            [(new_fact_lines, 0.40), (rel_fact_lines, 0.30), (refl_lines, 0.30)],
            self.MAX_CONTEXT_WORDS,
        )
        new_facts_text = "\n".join(truncated[0]) if truncated[0] else "(none)"
        related_facts_text = "\n".join(truncated[1]) if truncated[1] else "(none)"
        existing_reflections_text = "\n".join(truncated[2]) if truncated[2] else "(none)"

        prompt = load_prompt("reflection_generation", {
            "new_facts": new_facts_text,
            "related_facts": related_facts_text,
            "existing_reflections": existing_reflections_text,
        })

        raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
        data = self._parse_response(raw)
        return data.get("reflections", [])

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        stripped = _strip_json_block(raw or "")
        if not stripped:
            return {"reflections": []}
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Retry
        data = _retry_llm_for_json(
            self._llm, raw or "", "Could not parse reflection extraction JSON",
            self._max_tokens, expected_keys=["reflections"],
        )
        return data


class NoOpMemoryExtractor(MemoryExtractor):
    """No-op memory extractor: returns empty ExtractResult."""

    def extract(self, events: Sequence[TraceEvent]) -> ExtractResult:
        return ExtractResult(nodes=[], edge_specs=[])
