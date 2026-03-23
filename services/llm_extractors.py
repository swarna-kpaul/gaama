"""LLM-based memory extractors and STM services using the prompt library."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Sequence

from gaama.adapters.interfaces import LLMAdapter
from gaama.core import MemoryNode, ProvenanceRef, TraceEvent
from gaama.services.semantic_canonicalization import canonical_id_entity
from gaama.infra.prompt_loader import load_prompt
from gaama.services.interfaces import (
    EdgeSpec,
    EpisodeSummaryResult,
    EpisodeSummarizer,
    ExtractResult,
    MemoryExtractor,
    STMNoteExtractor,
    STMNoteItem,
)

logger = logging.getLogger(__name__)

_MAX_JSON_RETRIES = 2
_LTM_CONTENT_KEYS = ("facts", "episodes", "reflections", "skills")


def _ltm_has_content(data: Dict[str, Any]) -> bool:
    """Return True if at least one content category has items."""
    return any(bool(data.get(k)) for k in _LTM_CONTENT_KEYS)


def _format_conversation(events: Sequence[TraceEvent]) -> str:
    lines = []
    for e in events:
        line = f"{e.actor}: {e.content}"
        meta = getattr(e, "metadata", None) or {}
        if meta:
            parts = [f"{k}={v}" for k, v in meta.items() if v]
            if parts:
                line += f" [{', '.join(parts)}]"
        lines.append(line)
    return "\n".join(lines)


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


class LLMMemoryExtractor(MemoryExtractor):
    """Extracts entities, facts, episodes, reflections, and skills from conversation."""

    def __init__(self, llm: LLMAdapter, max_tokens: int = 16000) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def extract(self, events: Sequence[TraceEvent]) -> ExtractResult:
        if not events:
            return ExtractResult(nodes=[], edge_specs=[])
        conversation = _format_conversation(events)
        prompt = load_prompt("ltm_extraction", {"conversation": conversation})

        for attempt in range(_MAX_JSON_RETRIES + 1):
            if attempt == 0:
                raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
                if not (raw and raw.strip()):
                    print("############################################")
                    print("LLM returned EMPTY response. Check: OPENAI_API_KEY, model name (e.g. gpt-4o-mini), and network.")
                    print("############################################")
                data = _parse_ltm_response(raw, llm=self._llm, max_tokens=self._max_tokens)
            else:
                # Retry: re-call the original prompt with feedback
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous attempt produced 0 memory nodes. "
                    f"The conversation above clearly contains information. "
                    f"You MUST extract at least some facts, episodes, or reflections. "
                    f"Return valid JSON with non-empty arrays."
                )
                raw = self._llm.complete(retry_prompt, max_tokens=self._max_tokens)
                data = _parse_ltm_response(raw, llm=self._llm, max_tokens=self._max_tokens)

            try:
                print("############################################")
                print(f"LLM Response (attempt {attempt + 1}):")
                print(data)
                print("############################################")
            except UnicodeEncodeError:
                pass

            nodes, edge_specs = _build_memory_nodes_and_edges(data, events)
            if nodes:
                return ExtractResult(nodes=nodes, edge_specs=edge_specs)

            # No nodes built — retry if we have attempts left
            if attempt < _MAX_JSON_RETRIES:
                logger.warning(
                    "Extraction attempt %d produced 0 nodes from %d events. Retrying.",
                    attempt + 1, len(events),
                )
                print(f"[Extraction retry] Attempt {attempt + 1} produced 0 nodes. Retrying...")

        logger.warning("All extraction attempts produced 0 nodes for %d events.", len(events))
        print(f"[Extraction] All {_MAX_JSON_RETRIES + 1} attempts produced 0 nodes.")
        return ExtractResult(nodes=[], edge_specs=[])


def _parse_ltm_response(
    raw: str,
    llm: LLMAdapter | None = None,
    max_tokens: int = 16000,
) -> Dict[str, Any]:
    """Parse the LLM extraction response as JSON. On failure, retry the LLM
    with the error message so it can correct the output. Never returns empty."""
    stripped = _strip_json_block(raw)
    if not stripped or not stripped.strip():
        if llm is not None:
            return _retry_llm_for_json(
                llm, raw, "LLM returned empty response", max_tokens,
                expected_keys=["entities", "facts", "episodes", "reflections", "skills"],
            )
        return {}
    expected_keys = ["entities", "facts", "episodes", "reflections", "skills"]
    try:
        data = json.loads(stripped)
        if not isinstance(data, dict) or not data:
            if llm is not None:
                return _retry_llm_for_json(
                    llm, raw, "Parsed JSON is empty or not a dict", max_tokens,
                    expected_keys=expected_keys,
                )
            return {}
        # Valid dict but all content arrays empty — retry
        if not _ltm_has_content(data) and llm is not None:
            return _retry_llm_for_json(
                llm, raw,
                "All content arrays (facts, episodes, reflections, skills) are empty. "
                "The conversation clearly contains information to extract. "
                "Re-read the conversation carefully and extract facts, episodes, reflections, and skills.",
                max_tokens,
                expected_keys=expected_keys,
            )
        return data
    except json.JSONDecodeError as exc:
        if llm is not None:
            return _retry_llm_for_json(
                llm, raw, str(exc), max_tokens,
                expected_keys=expected_keys,
            )
        return {}


def _event_provenance(events: Sequence[TraceEvent]) -> List[ProvenanceRef]:
    return [ProvenanceRef(ref_type="trace_event", ref_id=e.event_id) for e in events]


def _event_session_date(events: Sequence[TraceEvent]) -> str:
    """Return the session_date from the first event's metadata, or empty string."""
    for e in events:
        meta = getattr(e, "metadata", None) or {}
        sd = meta.get("session_date", "")
        if sd:
            return sd
    return ""


def _clamp_01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, v))


def _safe_int(val: Any) -> int | None:
    """Return int if val is a non-negative integer, else None."""
    if val is None:
        return None
    try:
        i = int(val)
        return i if i >= 0 else None
    except (TypeError, ValueError):
        return None


def _build_memory_nodes_and_edges(
    data: Dict[str, Any], events: Sequence[TraceEvent]
) -> tuple[List[MemoryNode], List[EdgeSpec]]:
    """Build typed MemoryNodes and EdgeSpecs from the LLM JSON response.

    Uses a two-pass approach so that cross-type edges (e.g. Fact -> Episode)
    can reference nodes that appear later in the flat list.

    Pass 1: build all nodes, recording (flat_index, raw_item) per section.
    Pass 2: emit EdgeSpecs using known section offsets.
    """
    nodes: List[MemoryNode] = []
    now = datetime.utcnow()
    provenance = _event_provenance(events)
    session_date = _event_session_date(events)
    tags = {"session_date": session_date} if session_date else {}

    # ── Pass 1: build nodes ──────────────────────────────────────────────

    # 1. Entities
    for item in data.get("entities") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        aliases = item.get("aliases") or []
        nodes.append(MemoryNode(
            node_id=canonical_id_entity(name, aliases),
            created_at=now, updated_at=now,
            kind="entity", name=name, aliases=aliases,
            provenance=provenance, tags=dict(tags),
        ))
    entity_count = len(nodes)

    # 2. Facts
    fact_items: List[tuple[int, Dict]] = []
    for item in data.get("facts") or []:
        if isinstance(item, str):
            item = {"fact_text": item}
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("fact_text", "")).strip()
        if not fact_text:
            continue
        flat_idx = len(nodes)
        nodes.append(MemoryNode(
            node_id=canonical_id_entity(fact_text, []),
            created_at=now, updated_at=now,
            kind="fact", fact_text=fact_text,
            belief=_clamp_01(item.get("belief", 1.0)),
            polarity=bool(item.get("polarity", True)),
            provenance=provenance, tags=dict(tags),
        ))
        fact_items.append((flat_idx, item))
    fact_count = len(fact_items)

    # 3. Episodes
    episode_items: List[tuple[int, Dict]] = []
    for item in data.get("episodes") or []:
        if isinstance(item, str):
            item = {"summary": item}
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        raw_seq = _safe_int(item.get("sequence"))
        sequence = raw_seq if (raw_seq is not None and raw_seq >= 1) else None
        flat_idx = len(nodes)
        nodes.append(MemoryNode(
            node_id=canonical_id_entity(summary, []),
            created_at=now, updated_at=now,
            kind="episode", summary=summary,
            outcome=str(item.get("outcome", "")),
            belief=_clamp_01(item.get("belief", 1)),
            provenance=provenance, tags=dict(tags),
            sequence=sequence,
        ))
        episode_items.append((flat_idx, item))
    episode_count = len(episode_items)

    # 4. Reflections
    reflection_items: List[tuple[int, Dict]] = []
    for item in data.get("reflections") or []:
        if isinstance(item, str):
            item = {"reflection_text": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("reflection_text", "")).strip()
        if not text:
            continue
        flat_idx = len(nodes)
        nodes.append(MemoryNode(
            node_id=canonical_id_entity(text, []),
            created_at=now, updated_at=now,
            kind="reflection", reflection_text=text,
            belief=_clamp_01(item.get("belief", 1.0)),
            provenance=provenance, tags=dict(tags),
        ))
        reflection_items.append((flat_idx, item))
    reflection_count = len(reflection_items)

    # 5. Skills
    skill_items: List[tuple[int, Dict]] = []
    for item in data.get("skills") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        desc = str(item.get("skill_description", "")).strip()
        if not name and not desc:
            continue
        flat_idx = len(nodes)
        nodes.append(MemoryNode(
            node_id=canonical_id_entity(name or desc, []),
            created_at=now, updated_at=now,
            kind="skill", name=name, skill_description=desc,
            belief=_clamp_01(item.get("belief", 1.0)),
            provenance=provenance, tags=dict(tags),
        ))
        skill_items.append((flat_idx, item))

    # ── Section offsets (for resolving per-type indices to flat indices) ──
    fact_offset = entity_count
    episode_offset = entity_count + fact_count
    reflection_offset = entity_count + fact_count + episode_count
    skill_offset = entity_count + fact_count + episode_count + reflection_count
    total = len(nodes)

    def _entity_idx(val: Any) -> int | None:
        i = _safe_int(val)
        return i if i is not None and i < entity_count else None

    def _in_section(val: Any, offset: int, count: int) -> int | None:
        """Convert a per-type local index to a flat index, or None."""
        i = _safe_int(val)
        if i is None or i >= count:
            return None
        flat = offset + i
        return flat if flat < total else None

    # ── Pass 2: build edge specs ─────────────────────────────────────────
    edge_specs: List[EdgeSpec] = []

    def _add(src: int, tgt: int | None, etype: str) -> None:
        if tgt is not None and src != tgt:
            edge_specs.append(EdgeSpec(source_index=src, target_index=tgt, edge_type=etype))

    def _add_list(src: int, field: Any, etype: str, offset: int, count: int) -> None:
        for v in field or []:
            _add(src, _in_section(v, offset, count), etype)

    # 2. Fact edges
    for flat_idx, item in fact_items:
        _add(flat_idx, _entity_idx(item.get("subject")), "SUBJECT")
        _add(flat_idx, _entity_idx(item.get("object")), "OBJECT")
        _add_list(flat_idx, item.get("supported_by_episodes"), "SUPPORTED_BY", episode_offset, episode_count)
        _add_list(flat_idx, item.get("contradicts_facts"), "CONTRADICTS", fact_offset, fact_count)
        _add_list(flat_idx, item.get("refines_facts"), "REFINES", fact_offset, fact_count)

    # 3. Episode edges
    for flat_idx, item in episode_items:
        _add_list(flat_idx, item.get("involves"), "INVOLVES", 0, entity_count)
        _add_list(flat_idx, item.get("mentions"), "MENTIONS", 0, entity_count)
        _add_list(flat_idx, item.get("produced_facts"), "PRODUCED", fact_offset, fact_count)
        _add(flat_idx, _entity_idx(item.get("triggered_by")), "TRIGGERED_BY")
        _add(flat_idx, _in_section(item.get("next_episode"), episode_offset, episode_count), "NEXT")

    # 4. Reflection edges
    for flat_idx, item in reflection_items:
        _add_list(flat_idx, item.get("about"), "ABOUT", 0, entity_count)
        _add_list(flat_idx, item.get("supported_by_episodes"), "SUPPORTED_BY", episode_offset, episode_count)
        _add_list(flat_idx, item.get("derived_from_facts"), "DERIVED_FROM", fact_offset, fact_count)
        _add_list(flat_idx, item.get("contradicts_reflections"), "CONTRADICTS", reflection_offset, reflection_count)
        _add_list(flat_idx, item.get("refines_reflections"), "REFINES", reflection_offset, reflection_count)

    # 5. Skill edges
    for flat_idx, item in skill_items:
        _add_list(flat_idx, item.get("uses_tool"), "USES_TOOL", 0, entity_count)
        _add_list(flat_idx, item.get("applies_to"), "APPLIES_TO", 0, entity_count)
        _add_list(flat_idx, item.get("learned_from_episodes"), "LEARNED_FROM", episode_offset, episode_count)
        _add_list(flat_idx, item.get("refines_skills"), "REFINES", skill_offset, len(skill_items))
        _add_list(flat_idx, item.get("contradicts_skills"), "CONTRADICTS", skill_offset, len(skill_items))
        # Backward compat: also handle old "related_to" field
        _add_list(flat_idx, item.get("related_to"), "RELATED_TO", 0, entity_count)

    return nodes, edge_specs


# ---------------------------------------------------------------------------
# STM: Episode summarizer
# ---------------------------------------------------------------------------

class LLMEpisodeSummarizer(EpisodeSummarizer):
    """Summarizes conversation episodes using an LLM."""

    def __init__(self, llm: LLMAdapter, max_tokens: int = 512) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def summarize(self, events: Sequence[TraceEvent]) -> EpisodeSummaryResult:
        if not events:
            return EpisodeSummaryResult(summary="", outcomes=[], unresolved_items=[])
        conversation = _format_conversation(events)
        prompt = load_prompt("episode_summary", {"conversation": conversation})
        raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
        data = _parse_episode_summary_response(raw, llm=self._llm, max_tokens=self._max_tokens)
        return EpisodeSummaryResult(
            summary=data.get("summary", ""),
            outcomes=data.get("outcomes", []),
            unresolved_items=data.get("unresolved_items", []),
        )

    def summarize_update(
        self, previous_summary: str, new_events: Sequence[TraceEvent]
    ) -> EpisodeSummaryResult:
        if not new_events:
            return EpisodeSummaryResult(summary=previous_summary or "", outcomes=[], unresolved_items=[])
        new_conversation = _format_conversation(new_events)
        prompt = load_prompt(
            "episode_summary_update",
            {"previous_summary": previous_summary or "(none yet)", "new_conversation": new_conversation},
        )
        raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
        data = _parse_episode_summary_response(raw, llm=self._llm, max_tokens=self._max_tokens)
        return EpisodeSummaryResult(
            summary=data.get("summary", ""),
            outcomes=data.get("outcomes", []),
            unresolved_items=data.get("unresolved_items", []),
        )


def _parse_episode_summary_response(
    raw: str,
    llm: LLMAdapter | None = None,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    stripped = _strip_json_block(raw)
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and data:
            return {
                "summary": data.get("summary", ""),
                "outcomes": data.get("outcomes") or [],
                "unresolved_items": data.get("unresolved_items") or [],
            }
    except (json.JSONDecodeError, TypeError):
        pass
    # Retry if we have an LLM
    if llm is not None:
        error = "Could not parse as JSON" if stripped else "Empty response"
        data = _retry_llm_for_json(
            llm, raw, error, max_tokens,
            expected_keys=["summary", "outcomes", "unresolved_items"],
        )
        return {
            "summary": data.get("summary", ""),
            "outcomes": data.get("outcomes") or [],
            "unresolved_items": data.get("unresolved_items") or [],
        }
    return {"summary": "", "outcomes": [], "unresolved_items": []}


# ---------------------------------------------------------------------------
# STM: Working notes extractor
# ---------------------------------------------------------------------------

class LLMSTMNoteExtractor(STMNoteExtractor):
    """Extracts working notes (key-value) from conversation using an LLM."""

    def __init__(self, llm: LLMAdapter, max_tokens: int = 1024) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def extract_notes(self, events: Sequence[TraceEvent]) -> Sequence[STMNoteItem]:
        if not events:
            return []
        conversation = _format_conversation(events)
        prompt = load_prompt("stm_working_notes", {"conversation": conversation})
        raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
        data = _parse_stm_notes_response(raw, llm=self._llm, max_tokens=self._max_tokens)
        return [
            STMNoteItem(
                key=str(n.get("key", "")),
                value=n.get("value"),
                confidence=float(n.get("confidence", 0.8)),
            )
            for n in data
        ]


def _parse_stm_notes_response(
    raw: str,
    llm: LLMAdapter | None = None,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    stripped = _strip_json_block(raw)
    try:
        data = json.loads(stripped)
        notes = data.get("notes")
        if isinstance(notes, list) and notes:
            return notes
    except (json.JSONDecodeError, TypeError):
        pass
    if llm is not None:
        error = "Could not parse as JSON" if stripped else "Empty response"
        data = _retry_llm_for_json(
            llm, raw, error, max_tokens,
            expected_keys=["notes"],
        )
        notes = data.get("notes")
        return notes if isinstance(notes, list) else []
    return []


class NoOpMemoryExtractor(MemoryExtractor):
    """No-op memory extractor: returns empty ExtractResult."""

    def extract(self, events: Sequence[TraceEvent]) -> ExtractResult:
        return ExtractResult(nodes=[], edge_specs=[])
