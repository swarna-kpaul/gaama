from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from gaama.adapters import BlobStoreAdapter, GraphStoreAdapter, NodeStoreAdapter, VectorStoreAdapter
from gaama.adapters.interfaces import EmbeddingAdapter, LLMAdapter, VectorStoreAdapter
from gaama.adapters.sqlite_memory import get_first_episode_in_batch
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
from gaama.infra.serialization import node_to_embed_text
from gaama.services.graph_edges import build_edges_from_nodes
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
from gaama.services.semantic_canonicalization import EdgeCanonicalizer, NodeCanonicalizer


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
        canonicalizer: NodeCanonicalizer | None = None,
        edge_canonicalizer: EdgeCanonicalizer | None = None,
        budget_llm_adapter: LLMAdapter | None = None,
        budget_llm_model_name: str | None = None,
        budget_llm_max_tokens: int = 512,
        trace_buffer_max_events: int = 200,
        context_max_items: int = 50,
        context_max_tokens: int = 1200,
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
        self._canonicalizer = canonicalizer
        self._edge_canonicalizer = edge_canonicalizer
        self._budget_llm_adapter = budget_llm_adapter
        self._budget_llm_model_name = budget_llm_model_name
        self._budget_llm_max_tokens = budget_llm_max_tokens
        self._trace_buffer_max_events = trace_buffer_max_events
        self._trace_buffer: Dict[str, List[TraceEvent]] = {}
        self._last_flushed_index: Dict[str, int] = {}
        self._last_created_index: Dict[str, int] = {}

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
        """Divide events into chunks such that each chunk's total token count does not exceed max_tokens_per_chunk."""
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
        """Create LTM nodes from the trace-event buffer (events since last create)."""
        _t0 = time.perf_counter()
        _profile: List[Tuple[str, float]] = []

        if not options.agent_id or not options.agent_id.strip():
            raise ValueError("agent_id is required for memory creation.")
        agent_id = options.agent_id
        buf = self._buffer_for(agent_id)
        lc = self._last_created(agent_id)
        events = list(buf[lc + 1:])
        if not events:
            return []
        _profile.append(("buffer_slice", time.perf_counter() - _t0))

        # Chunk: by max_tokens_per_chunk (bucket fill), or by chunk_size/overlap, or single chunk.
        _t = time.perf_counter()
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
        _profile.append(("chunking", time.perf_counter() - _t))

        scope = Scope(
            agent_id=agent_id,
            user_id=options.user_id or None,
            task_id=options.task_id or None,
        )
        all_node_ids: List[str] = []
        _t = time.perf_counter()
        sequence_offset = 0
        if hasattr(self._node_store, "get_max_sequence"):
            sequence_offset = self._node_store.get_max_sequence(agent_id)
        elif hasattr(self._node_store, "get_max_episode_sequence"):
            sequence_offset = self._node_store.get_max_episode_sequence(agent_id)
        _profile.append(("get_sequence_offset", time.perf_counter() - _t))

        for chunk_idx, chunk_events in enumerate(chunks):
            if not chunk_events:
                continue
            _chunk_t0 = time.perf_counter()

            _t = time.perf_counter()
            result = self._extractor.extract(chunk_events)
            valid_nodes = list(result.nodes)
            _profile.append(("extract", time.perf_counter() - _t))
            if not valid_nodes:
                continue

            for node in valid_nodes:
                node.scopes = [scope]
            for i, node in enumerate(valid_nodes):
                node.sequence = sequence_offset + (i + 1)
            sequence_offset += len(valid_nodes)

            new_episodes = [n for n in valid_nodes if (n.kind or "").strip().lower() == "episode"]

            _t = time.perf_counter()
            edges = build_edges_from_nodes(valid_nodes, result.edge_specs)
            node_id_map = self._canonicalize_nodes(valid_nodes)
            _profile.append(("build_edges+canonicalize_nodes", time.perf_counter() - _t))

            _t = time.perf_counter()
            remapped: List[Edge] = []
            for e in edges:
                new_src = node_id_map.get(e.source_id, e.source_id)
                new_tgt = node_id_map.get(e.target_id, e.target_id)
                if new_src != new_tgt:
                    remapped.append(Edge(
                        edge_id=e.edge_id,
                        edge_type=e.edge_type,
                        source_id=new_src,
                        target_id=new_tgt,
                        created_at=e.created_at,
                        label=getattr(e, "label", "") or "",
                        weight=getattr(e, "weight", 1.0),
                    ))
            edges = remapped
            _profile.append(("remap_edges", time.perf_counter() - _t))

            _t = time.perf_counter()
            now = datetime.utcnow()
            if new_episodes and hasattr(self._node_store, "get_last_episode_node"):
                last_existing = self._node_store.get_last_episode_node(agent_id)
                first_new = get_first_episode_in_batch(new_episodes, edges) or new_episodes[0]
                if last_existing and last_existing.node_id != first_new.node_id:
                    edges.append(Edge(
                        edge_id=f"edge-{uuid4().hex}",
                        edge_type="NEXT",
                        source_id=last_existing.node_id,
                        target_id=first_new.node_id,
                        created_at=now,
                    ))
            _profile.append(("get_last_episode+NEXT_edge", time.perf_counter() - _t))

            _t = time.perf_counter()
            node_ids = self._node_store.upsert_nodes(valid_nodes)
            all_node_ids.extend(node_ids)
            _profile.append(("upsert_nodes", time.perf_counter() - _t))

            _t = time.perf_counter()
            if edges:
                self._graph_store.upsert_edges(edges)
            _profile.append(("upsert_edges", time.perf_counter() - _t))

            _t = time.perf_counter()
            self._upsert_node_embeddings(valid_nodes)
            _profile.append(("upsert_node_embeddings", time.perf_counter() - _t))

            _t = time.perf_counter()
            self._index_nodes_in_canonicalizer(valid_nodes)
            _profile.append(("index_nodes_in_canonicalizer", time.perf_counter() - _t))

            _chunk_elapsed = time.perf_counter() - _chunk_t0
            print(f"[create] chunk {chunk_idx + 1}/{len(chunks)}: {len(valid_nodes)} nodes, {len(edges)} edges — {_chunk_elapsed:.2f}s")

        _t = time.perf_counter()
        if buf:
            self._last_created_index[agent_id] = len(buf) - 1
        _profile.append(("update_last_created", time.perf_counter() - _t))

        _total = time.perf_counter() - _t0
        # Aggregate by step name and print summary
        _by_step: Dict[str, float] = {}
        for name, sec in _profile:
            _by_step[name] = _by_step.get(name, 0.0) + sec
        print("\n--- create() time profile summary ---")
        for name in [
            "buffer_slice", "chunking", "get_sequence_offset",
            "extract", "build_edges+canonicalize_nodes", "remap_edges",
            "get_last_episode+NEXT_edge",
            "upsert_nodes", "upsert_edges", "upsert_node_embeddings",
            "index_nodes_in_canonicalizer", "update_last_created",
        ]:
            sec = _by_step.get(name, 0.0)
            pct = (sec / _total * 100) if _total > 0 else 0
            print(f"  {name}: {sec:.3f}s ({pct:.1f}%)")
        print(f"  TOTAL: {_total:.3f}s")
        print("--------------------------------------\n")
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
        """Call LLM to derive retrieval budget from query."""
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
        if total > max_memory_items and total > 0:
            scale = max_memory_items / total
            max_f = max(0, int(round(max_f * scale)))
            max_r = max(0, int(round(max_r * scale)))
            max_s = max(0, int(round(max_s * scale)))
            max_e = max(0, int(round(max_e * scale)))
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
        """Scale the four category caps so they sum to max_memory_items."""
        total = budget.max_facts + budget.max_reflections + budget.max_skills + budget.max_episodes
        if total <= 0:
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

        # Optional: derive budget from query via LLM
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

    # -- Semantic canonicalization ---------------------------------------------

    def _canonicalize_nodes(self, nodes: list[MemoryNode]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not self._canonicalizer:
            for node in nodes:
                out[node.node_id] = node.node_id
            return out

        if self._embedder and hasattr(self._embedder, "embed_batch"):
            texts = [node_to_embed_text(n) or "" for n in nodes]
            self._embedder.embed_batch(texts)

        def _resolve_one(node: MemoryNode):
            old_id = node.node_id
            match = self._canonicalizer.resolve_node(node)
            return old_id, node, match

        results: list[tuple] = []
        with ThreadPoolExecutor(max_workers=min(len(nodes), 8)) as pool:
            futures = {pool.submit(_resolve_one, n): n for n in nodes}
            for fut in as_completed(futures):
                results.append(fut.result())

        for old_id, node, match in results:
            if match.matched_existing:
                node.node_id = match.node_id
                existing = self._node_store.get_nodes([match.node_id])
                if existing:
                    merged = list(existing[0].scopes)
                    for s in node.scopes:
                        if s not in merged:
                            merged.append(s)
                    node.scopes = merged
            out[old_id] = node.node_id
        return out

    def _index_nodes_in_canonicalizer(self, nodes: list[MemoryNode]) -> None:
        if not self._canonicalizer:
            return
        self._canonicalizer.index_nodes(nodes)

    def _canonicalize_edges(
        self, edges: list[Edge],
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
    ) -> list[Edge]:
        if not self._edge_canonicalizer or not edges:
            return edges
        resolved: list[Edge] = []
        seen_ids: set[str] = set()
        for edge in edges:
            match = self._edge_canonicalizer.resolve_edge(edge, agent_id=agent_id, user_id=user_id, task_id=task_id)
            if match.edge_id in seen_ids:
                continue
            seen_ids.add(match.edge_id)
            resolved.append(Edge(
                edge_id=match.edge_id,
                edge_type=edge.edge_type,
                source_id=edge.source_id,
                target_id=edge.target_id,
                created_at=edge.created_at,
                label=edge.label,
                weight=edge.weight,
            ))
        return resolved

    def _index_edges_in_canonicalizer(
        self, edges: list[Edge],
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
    ) -> None:
        if not self._edge_canonicalizer:
            return
        self._edge_canonicalizer.index_edges(edges, agent_id=agent_id, user_id=user_id, task_id=task_id)

    # -- Node embedding --------------------------------------------------------

    def _upsert_node_embeddings(self, nodes: list[MemoryNode]) -> int:
        if not self._embedder or not nodes:
            return 0
        if not hasattr(self._vector_store, "upsert_embeddings"):
            return 0

        texts: list[str] = []
        text_indices: list[int] = []
        for i, node in enumerate(nodes):
            text = node_to_embed_text(node)
            if text:
                texts.append(text)
                text_indices.append(i)

        if not texts:
            return 0

        if hasattr(self._embedder, "embed_batch"):
            embeddings = self._embedder.embed_batch(texts)
            for j, emb in enumerate(embeddings):
                if emb:
                    nodes[text_indices[j]].embedding = list(emb)
        else:
            for j, text in enumerate(texts):
                emb = self._embedder.embed(text)
                if emb:
                    nodes[text_indices[j]].embedding = list(emb)

        to_upsert = [n for n in nodes if getattr(n, "embedding", None)]
        if not to_upsert:
            return 0
        ids = self._vector_store.upsert_embeddings(to_upsert)
        return len(ids) if ids is not None else len(to_upsert)
