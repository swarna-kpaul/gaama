from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Default similarity threshold for connecting related nodes
DEFAULT_SIMILARITY_THRESHOLD = 0.75

from gaama.adapters import BlobStoreAdapter, GraphStoreAdapter, NodeStoreAdapter, VectorStoreAdapter
from gaama.adapters.interfaces import EmbeddingAdapter, LLMAdapter, VectorStoreAdapter
from gaama.core import (
    Edge,
    EvalReport,
    ForgetReport,
    IntegrationBundle,
    MemoryNode,
    MemoryPack,
    QueryFilters,
    RetrievalBudget,
    Scope,
    TraceEvent,
)
from gaama.core.policies import ExtractionPolicy
from gaama.infra.prompt_loader import load_prompt
from gaama.services.interfaces import (
    CreateOptions,
    Evaluator,
    ForgettingEngine,
    IntegrationComposer,
    MemoryExtractor,
    RetrieveOptions,
    RetrievalEngine,
    TraceNormalizer,
)
from gaama.services.ltm_creator import LTMCreator


class AgenticMemoryOrchestrator:
    def __init__(
        self,
        normalizer: TraceNormalizer,
        extractor: MemoryExtractor,
        ltm_retriever: RetrievalEngine,
        forgetter: ForgettingEngine,
        integrator: IntegrationComposer,
        evaluator: Evaluator,
        node_store: NodeStoreAdapter,
        graph_store: GraphStoreAdapter,
        vector_store: VectorStoreAdapter,
        blob_store: BlobStoreAdapter,
        extraction_policy: ExtractionPolicy | None = None,
        embedder: EmbeddingAdapter | None = None,
        budget_llm_adapter: LLMAdapter | None = None,
        budget_llm_model_name: str | None = None,
        budget_llm_max_tokens: int = 512,
        trace_buffer_max_events: int = 200,
        context_max_items: int = 50,
        context_max_tokens: int = 1200,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._normalizer = normalizer
        self._extractor = extractor
        self._ltm_retriever = ltm_retriever
        self._forgetter = forgetter
        self._integrator = integrator
        self._evaluator = evaluator
        self._node_store = node_store
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._blob_store = blob_store
        self._extraction_policy = extraction_policy or ExtractionPolicy()
        self._context_max_items = context_max_items
        self._context_max_tokens = context_max_tokens
        self._embedder = embedder
        self._budget_llm_adapter = budget_llm_adapter
        self._budget_llm_model_name = budget_llm_model_name
        self._budget_llm_max_tokens = budget_llm_max_tokens
        self._trace_buffer_max_events = trace_buffer_max_events
        self._similarity_threshold = similarity_threshold
        self._trace_buffer: Dict[str, List[TraceEvent]] = {}
        self._last_flushed_index: Dict[str, int] = {}
        self._last_created_index: Dict[str, int] = {}

        # LLM adapter is extracted from the extractor (may be None)
        _llm = getattr(self._extractor, "_llm", None)
        self._ltm_creator = LTMCreator(
            node_store=node_store,
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
            llm=_llm,
            similarity_threshold=similarity_threshold,
        )

    def _buffer_for(self, agent_id: str) -> List[TraceEvent]:
        if agent_id not in self._trace_buffer:
            self._trace_buffer[agent_id] = []
        return self._trace_buffer[agent_id]

    def _last_flushed(self, agent_id: str) -> int:
        return self._last_flushed_index.get(agent_id, -1)

    def _last_created(self, agent_id: str) -> int:
        return self._last_created_index.get(agent_id, -1)

    def ingest(self, trace_events, agent_id: str) -> Sequence[str]:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id is required for ingest.")
        normalized = list(self._normalizer.normalize(trace_events))
        buf = self._buffer_for(agent_id)
        lf = self._last_flushed(agent_id)
        lc = self._last_created(agent_id)
        for event in normalized:
            buf.append(event)
            while len(buf) > self._trace_buffer_max_events and lf >= 0 and lc >= 0:
                buf.pop(0)
                lf -= 1
                lc -= 1
        self._last_flushed_index[agent_id] = lf
        self._last_created_index[agent_id] = lc
        raw_key = f"traces/{uuid4().hex}.jsonl"
        payload = "\n".join([event.content for event in normalized]).encode("utf-8")
        self._blob_store.put_blob(raw_key, payload)
        return [event.event_id for event in normalized]

    @staticmethod
    def _estimate_event_tokens(event: TraceEvent) -> int:
        """Rough token count for one trace event (actor + content). Uses word count as proxy."""
        text = (getattr(event, "actor", "") or "") + " " + (getattr(event, "content", "") or "")
        return max(1, len(text.split()))

    def _events_to_token_bounded_chunks(
        self,
        events: List[TraceEvent],
        max_tokens_per_chunk: int,
    ) -> List[List[TraceEvent]]:
        """Divide events into chunks such that each chunk's total token count does not exceed max_tokens_per_chunk.
        Events are never split; each event is assigned entirely to one bucket. Fills a bucket until adding the
        next event would exceed the limit, then starts a new bucket."""
        if max_tokens_per_chunk <= 0 or not events:
            return [events] if events else []
        chunks: List[List[TraceEvent]] = []
        current: List[TraceEvent] = []
        current_tokens = 0
        for event in events:
            tok = self._estimate_event_tokens(event)
            if current_tokens + tok > max_tokens_per_chunk and current:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(event)
            current_tokens += tok
        if current:
            chunks.append(current)
        return chunks

    def create(self, options: CreateOptions) -> Sequence[str]:
        """Create LTM nodes from the trace-event buffer.

        Handles buffer management and chunking, then delegates each chunk
        to ``LTMCreator.create_from_events()`` for the four-step pipeline
        (episodes -> related-episode edges -> facts -> reflections).

        Chunking: if options.max_tokens_per_chunk is set, events are divided into
        token-bounded buckets; else if options.chunk_size > 0, events are processed
        in fixed-size chunks; otherwise all events are processed in one pass.
        """
        _t0 = time.perf_counter()

        if not options.agent_id or not options.agent_id.strip():
            raise ValueError("agent_id is required for memory creation.")
        agent_id = options.agent_id
        buf = self._buffer_for(agent_id)
        lc = self._last_created(agent_id)
        events = list(buf[lc + 1:])
        if not events:
            return []

        # Chunk events
        if options.max_tokens_per_chunk is not None and options.max_tokens_per_chunk > 0:
            chunks = self._events_to_token_bounded_chunks(events, options.max_tokens_per_chunk)
            print(f"\n--- create() token-bounded chunks (max_tokens_per_chunk={options.max_tokens_per_chunk}) ---")
            print(f"Total events: {len(events)} -> {len(chunks)} chunks to process:")
            for i, ch in enumerate(chunks):
                tok = sum(self._estimate_event_tokens(e) for e in ch)
                print(f"  Chunk {i+1}/{len(chunks)}: {len(ch)} events, ~{tok} tokens")
            print("----------------------------------------\n")
        else:
            chunk_size = options.chunk_size or 0
            if chunk_size <= 0:
                chunks = [events]
            else:
                overlap = min(max(0, options.chunk_overlap or 0), chunk_size - 1)
                step = max(1, chunk_size - overlap)
                chunks = [
                    events[i : i + chunk_size]
                    for i in range(0, len(events), step)
                ]

        scope = Scope(
            agent_id=agent_id,
            user_id=options.user_id or None,
            task_id=options.task_id or None,
        )
        all_node_ids: List[str] = []

        sequence_offset = 0
        if hasattr(self._node_store, "get_max_sequence"):
            sequence_offset = self._node_store.get_max_sequence(agent_id)
        elif hasattr(self._node_store, "get_max_episode_sequence"):
            sequence_offset = self._node_store.get_max_episode_sequence(agent_id)

        filters = QueryFilters(
            agent_id=agent_id,
            task_id=options.task_id or None,
        )

        for chunk_idx, chunk_events in enumerate(chunks):
            if not chunk_events:
                continue
            print(f"[create] processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_events)} events)")
            chunk_ids, total_new = self._ltm_creator.create_from_events(
                chunk_events,
                scope,
                filters=filters,
                sequence_offset=sequence_offset,
            )
            all_node_ids.extend(chunk_ids)
            sequence_offset += total_new

        # Update last-created index
        if buf:
            self._last_created_index[agent_id] = len(buf) - 1

        _total = time.perf_counter() - _t0
        print(f"\n--- create() total: {_total:.3f}s, {len(all_node_ids)} nodes ---\n")
        return all_node_ids

    def _derive_budget_from_llm(
        self,
        query: str,
        base: RetrievalBudget,
        llm: LLMAdapter,
        max_tokens: int,
        max_memory_items: int,
        model_name: str | None = None,
    ) -> RetrievalBudget:
        """Call LLM to derive retrieval budget from query; LLM distributes max_memory_items across categories."""
        prompt = load_prompt(
            "retrieval_budget",
            {"query": query, "max_memory_items": str(max_memory_items)},
        )
        raw = llm.complete(prompt, max_tokens=max_tokens, model=model_name)
        raw = raw.strip()
        for pattern in (r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", r"^```\s*\n?(.*?)\n?```\s*$"):
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
                break
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            from gaama.services.llm_extractors import _retry_llm_for_json
            data = _retry_llm_for_json(
                llm, raw, "Invalid JSON in budget response", max_tokens,
                expected_keys=["max_facts", "max_reflections", "max_skills", "max_episodes"],
            )
        if not isinstance(data, dict):
            return base
        # Parse category caps; clamp each to [0, max_memory_items]
        def _int(key: str, default: int) -> int:
            val = data.get(key)
            if val is None:
                return default
            try:
                return max(0, min(max_memory_items, int(val)))
            except (TypeError, ValueError):
                return default

        max_f = _int("max_facts", base.max_facts)
        max_r = _int("max_reflections", base.max_reflections)
        max_s = _int("max_skills", base.max_skills)
        max_e = _int("max_episodes", base.max_episodes)
        total = max_f + max_r + max_s + max_e
        # Enforce sum <= max_memory_items (scale down proportionally if over)
        if total > max_memory_items and total > 0:
            scale = max_memory_items / total
            max_f = max(0, int(round(max_f * scale)))
            max_r = max(0, int(round(max_r * scale)))
            max_s = max(0, int(round(max_s * scale)))
            max_e = max(0, int(round(max_e * scale)))
            # Ensure sum exactly max_memory_items (adjust largest category for rounding)
            current = max_f + max_r + max_s + max_e
            if current < max_memory_items and max_e > 0:
                max_e += max_memory_items - current
            elif current < max_memory_items and max_f > 0:
                max_f += max_memory_items - current
        return RetrievalBudget(
            max_tokens=base.max_tokens,
            max_items=max_memory_items,
            max_facts=max_f,
            max_reflections=max_r,
            max_skills=max_s,
            max_episodes=max_e,
        )

    def _apply_max_memory_items(self, budget: RetrievalBudget, max_memory_items: int) -> RetrievalBudget:
        """Scale the four category caps so they sum to max_memory_items (LTM uses these, not max_items)."""
        total = budget.max_facts + budget.max_reflections + budget.max_skills + budget.max_episodes
        if total <= 0:
            # Equal split
            q, r = divmod(max_memory_items, 4)
            return RetrievalBudget(
                max_tokens=budget.max_tokens,
                max_items=max_memory_items,
                max_facts=q + (1 if r > 0 else 0),
                max_reflections=q + (1 if r > 1 else 0),
                max_skills=q + (1 if r > 2 else 0),
                max_episodes=q + (1 if r > 3 else 0),
            )
        if total == max_memory_items:
            return budget
        scale = max_memory_items / total
        max_f = max(0, int(round(budget.max_facts * scale)))
        max_r = max(0, int(round(budget.max_reflections * scale)))
        max_s = max(0, int(round(budget.max_skills * scale)))
        max_e = max(0, int(round(budget.max_episodes * scale)))
        current = max_f + max_r + max_s + max_e
        if current < max_memory_items and max_e > 0:
            max_e += max_memory_items - current
        elif current < max_memory_items and max_f > 0:
            max_f += max_memory_items - current
        return RetrievalBudget(
            max_tokens=budget.max_tokens,
            max_items=max_memory_items,
            max_facts=max_f,
            max_reflections=max_r,
            max_skills=max_s,
            max_episodes=max_e,
        )

    def retrieve(self, query: str, options: RetrieveOptions) -> MemoryPack:
        if not options.filters.agent_id or not options.filters.agent_id.strip():
            raise ValueError("agent_id is required for memory retrieval.")
        agent_id = options.filters.agent_id
        sources = options.sources

        # Optional: derive budget from query via LLM (internal adapter + model name)
        effective_budget = options.budget
        if options.use_llm_budget and self._budget_llm_adapter is not None:
            model_name = options.budget_llm_model_name or self._budget_llm_model_name
            if model_name:
                max_memory_items = (
                    options.max_memory_items
                    if options.max_memory_items is not None
                    else options.budget.max_items
                )
                max_memory_items = max(1, max_memory_items)
                effective_budget = self._derive_budget_from_llm(
                    query,
                    options.budget,
                    self._budget_llm_adapter,
                    self._budget_llm_max_tokens,
                    max_memory_items=max_memory_items,
                    model_name=model_name,
                )
        # When max_memory_items is set, ensure the four category caps sum to it (LTM uses these, not max_items)
        if options.max_memory_items is not None:
            effective_budget = self._apply_max_memory_items(
                effective_budget, max(1, options.max_memory_items)
            )
        options_with_budget = replace(options, budget=effective_budget)
        _fmt = options_with_budget
        _b = _fmt.budget
        print(
            "RetrieveOptions:\n"
            f"  filters: agent_id={getattr(_fmt.filters, 'agent_id', None)!r} user_id={getattr(_fmt.filters, 'user_id', None)!r} task_id={getattr(_fmt.filters, 'task_id', None)!r}\n"
            f"  budget: max_facts={_b.max_facts} max_reflections={_b.max_reflections} max_skills={_b.max_skills} max_episodes={_b.max_episodes} max_items={_b.max_items} max_tokens={_b.max_tokens}\n"
            f"  sources={_fmt.sources!r} use_llm_budget={_fmt.use_llm_budget}"
        )

        # GAAMA only supports LTM retrieval (no STM)
        if sources in ("ltm", "both"):
            pack, _ = self._ltm_retriever.retrieve(query, options_with_budget)
            return pack
        return MemoryPack()

    def flush_stm_episode(self, agent_id: str) -> None:
        """No-op: GAAMA does not include STM."""
        return None

    def forget(self, selector: QueryFilters) -> ForgetReport:
        deleted_ids = self._forgetter.forget(selector)
        return ForgetReport(deleted_node_ids=deleted_ids, policy_applied="custom")

    def clear_ltm(self) -> None:
        """Delete all long-term memory from backing store."""
        if hasattr(self._node_store, "clear_ltm"):
            self._node_store.clear_ltm()
        if hasattr(self._vector_store, "clear_ltm"):
            self._vector_store.clear_ltm()

    def clear_trace_buffer(self, agent_id: str) -> None:
        if agent_id in self._trace_buffer:
            self._trace_buffer[agent_id] = []
        self._last_flushed_index[agent_id] = -1
        self._last_created_index[agent_id] = -1

    def get_trace_events(
        self,
        agent_id: str,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Sequence[TraceEvent]:
        """Return trace events for the agent, optionally filtered by user_id and/or task_id."""
        buf = list(self._buffer_for(agent_id))
        if user_id is not None:
            buf = [e for e in buf if getattr(e, "user_id", None) == user_id]
        if task_id is not None:
            buf = [e for e in buf if getattr(e, "task_id", None) == task_id]
        return buf

    def get_trace_events_since(
        self,
        agent_id: str,
        since_index: int,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[Sequence[TraceEvent], int]:
        """Return (events_since_index, current_len) for the filtered trace buffer."""
        buf = list(self._buffer_for(agent_id))
        if user_id is not None:
            buf = [e for e in buf if getattr(e, "user_id", None) == user_id]
        if task_id is not None:
            buf = [e for e in buf if getattr(e, "task_id", None) == task_id]
        current_len = len(buf)
        slice_ = buf[since_index:] if since_index < current_len else []
        return (slice_, current_len)

    def integrate(self, query: str, memory_pack: MemoryPack, mode: str, agent_id: str | None = None) -> IntegrationBundle:
        bundle = self._integrator.compose(query, memory_pack, mode)
        prompt_pack = bundle.get("prompt_pack")
        context = self._render_context_from_buffer(agent_id)
        if context:
            prompt_pack = f"{context}\n\n{prompt_pack}" if prompt_pack else context
        return IntegrationBundle(
            prompt_pack=prompt_pack,
            structured_state=bundle.get("structured_state", {}),
            tool_hints=bundle.get("tool_hints", []),
        )

    def evaluate(self, dataset_id: str) -> EvalReport:
        return self._evaluator.evaluate(dataset_id)

    def _render_context_from_buffer(self, agent_id: str | None = None) -> str:
        if not agent_id:
            return ""
        buf = self._trace_buffer.get(agent_id)
        if not buf:
            return ""
        items = list(buf[-self._context_max_items:])
        lines: List[str] = []
        total_tokens = 0
        for event in reversed(items):
            est = max(1, len(event.content.split()))
            if total_tokens + est > self._context_max_tokens:
                break
            lines.append(event.content)
            total_tokens += est
        return "\n".join(reversed(lines))
