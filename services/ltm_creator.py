"""LTM creation pipeline: episodes -> facts + concepts -> reflections.

Extracted from ``AgenticMemoryOrchestrator.create()`` so the orchestrator
remains a thin routing layer.

Graph structure (concept-based):
    Episode -[NEXT]-> Episode              (temporal chain)
    Fact    -[DERIVED_FROM]-> Episode      (provenance)
    Episode -[HAS_CONCEPT]-> Concept       (topical grouping)
    Fact    -[ABOUT_CONCEPT]-> Concept     (topical grouping)
    Reflection -[DERIVED_FROM_FACT]-> Fact (provenance)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from gaama.adapters import GraphStoreAdapter, NodeStoreAdapter, VectorStoreAdapter
from gaama.adapters.interfaces import EmbeddingAdapter, LLMAdapter
from gaama.core import (
    Edge,
    MemoryNode,
    QueryFilters,
    Scope,
    TraceEvent,
)
from gaama.infra.serialization import node_to_embed_text
from gaama.infra.id_helpers import canonical_id_entity
from gaama.services.graph_edges import make_edge

logger = logging.getLogger(__name__)

# Default similarity threshold for connecting related nodes
DEFAULT_SIMILARITY_THRESHOLD = 0.75


class LTMCreator:
    """Runs the three-step LTM creation pipeline for a single chunk of events.

    Steps:
        1. Convert raw conversation turns to episode nodes (no LLM).
        2. Use LLM to generate facts AND concepts from episodes + context.
           Create concept nodes, HAS_CONCEPT and ABOUT_CONCEPT edges.
        3. Use LLM to generate reflections from facts + context.
    """

    def __init__(
        self,
        node_store: NodeStoreAdapter,
        graph_store: GraphStoreAdapter,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter | None = None,
        llm: LLMAdapter | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._node_store = node_store
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm
        self._similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def create_from_events(
        self,
        events: List[TraceEvent],
        scope: Scope,
        *,
        filters: QueryFilters,
        sequence_offset: int = 0,
    ) -> Tuple[List[str], int]:
        """Run the full LTM creation pipeline on *events* (one chunk).

        Returns ``(all_node_ids, total_nodes_created)`` where
        *total_nodes_created* is the count of episodes + facts + concepts +
        reflections so the caller can advance its sequence offset.
        """
        _chunk_t0 = time.perf_counter()

        if not events:
            return [], 0

        sim_threshold = self._similarity_threshold

        # =============================================================
        # Step 1: Create episode nodes directly from conversation turns
        # =============================================================
        episode_nodes: List[MemoryNode] = []
        now = datetime.utcnow()

        for i, event in enumerate(events):
            text = getattr(event, "content", "") or ""
            if not text.strip():
                continue

            episode_content = text
            meta = getattr(event, "metadata", None) or {}
            session_date = meta.get("session_date", "")
            tags = {"session_date": session_date} if session_date else {}

            seq = sequence_offset + len(episode_nodes) + 1
            node_id = f"ep-{uuid4().hex[:16]}"

            episode_node = MemoryNode(
                node_id=node_id,
                created_at=now,
                updated_at=now,
                kind="episode",
                summary=episode_content,
                tags=tags,
                scopes=[scope],
                sequence=seq,
                relevance_score=1.0,
                provenance=[],
            )
            episode_nodes.append(episode_node)

        if not episode_nodes:
            return [], 0

        # Get last existing episode BEFORE inserting new ones (for NEXT edge linking)
        last_existing_episode = None
        if hasattr(self._node_store, "get_last_episode_node"):
            last_existing_episode = self._node_store.get_last_episode_node(scope.agent_id)

        # Embed episodes and upsert to node + vector store
        self._upsert_node_embeddings(episode_nodes)

        all_node_ids: List[str] = []
        ep_ids = self._node_store.upsert_nodes(episode_nodes)
        all_node_ids.extend(ep_ids)

        # NEXT edges between consecutive episodes in this chunk
        edges: List[Edge] = []

        # Link last existing episode to first new episode
        if last_existing_episode and last_existing_episode.node_id != episode_nodes[0].node_id:
            edges.append(make_edge(
                last_existing_episode.node_id, episode_nodes[0].node_id, "NEXT",
            ))

        # NEXT edges within this chunk
        for i in range(len(episode_nodes) - 1):
            edges.append(make_edge(
                episode_nodes[i].node_id, episode_nodes[i + 1].node_id, "NEXT",
            ))

        # =============================================================
        # Step 2: Fact + Concept generation via LLM
        # =============================================================
        new_fact_nodes: List[MemoryNode] = []
        new_concept_nodes: List[MemoryNode] = []
        if self._llm is not None:
            new_fact_nodes, new_concept_nodes = self._step2_generate_facts_and_concepts(
                self._llm, episode_nodes, scope,
                filters, sim_threshold, edges,
                sequence_offset + len(episode_nodes),
            )
            if new_fact_nodes:
                fact_ids = self._node_store.upsert_nodes(new_fact_nodes)
                all_node_ids.extend(fact_ids)
            if new_concept_nodes:
                concept_ids = self._node_store.upsert_nodes(new_concept_nodes)
                all_node_ids.extend(concept_ids)

        # =============================================================
        # Step 3: Reflection generation via LLM
        # =============================================================
        new_reflection_nodes: List[MemoryNode] = []
        if self._llm is not None and new_fact_nodes:
            new_reflection_nodes = self._step3_generate_reflections(
                self._llm, new_fact_nodes, scope, filters, edges,
                sequence_offset + len(episode_nodes) + len(new_fact_nodes) + len(new_concept_nodes),
            )
            if new_reflection_nodes:
                refl_ids = self._node_store.upsert_nodes(new_reflection_nodes)
                all_node_ids.extend(refl_ids)

        total_new = (
            len(episode_nodes) + len(new_fact_nodes)
            + len(new_concept_nodes) + len(new_reflection_nodes)
        )

        # Upsert all edges
        if edges:
            self._graph_store.upsert_edges(edges)

        _chunk_elapsed = time.perf_counter() - _chunk_t0
        print(
            f"[LTMCreator] "
            f"{len(episode_nodes)} episodes, {len(new_fact_nodes)} facts, "
            f"{len(new_concept_nodes)} concepts, "
            f"{len(new_reflection_nodes)} reflections, {len(edges)} edges "
            f"— {_chunk_elapsed:.2f}s"
        )

        return all_node_ids, total_new

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _step2_generate_facts_and_concepts(
        self,
        llm: LLMAdapter,
        episode_nodes: List[MemoryNode],
        scope: Scope,
        filters: QueryFilters,
        sim_threshold: float,
        edges: List[Edge],
        seq_start: int,
    ) -> Tuple[List[MemoryNode], List[MemoryNode]]:
        """Step 2: Generate facts AND concepts from episodes using a single LLM call.

        2a. Context retrieval:
            - Find related older episodes via vector search.
            - Find existing facts connected to those episodes.
            - Find existing concept nodes via vector search.
        2b. Single LLM call extracts facts + concepts.
        2c. Create nodes and edges:
            - Fact nodes + DERIVED_FROM edges to source episodes.
            - Concept nodes + HAS_CONCEPT (episode->concept) and
              ABOUT_CONCEPT (fact->concept) edges.

        Returns (new_fact_nodes, new_concept_nodes).
        """
        from gaama.services.llm_extractors import LLMFactExtractor

        # --- 2a. Context retrieval ---

        # Find related older episodes via vector search
        related_older_episodes: List[MemoryNode] = []
        seen_ids: set = {ep.node_id for ep in episode_nodes}
        for ep_node in episode_nodes:
            ep_text = node_to_embed_text(ep_node)
            if not ep_text:
                continue
            try:
                results = self._vector_store.search(ep_text, filters, top_k=10, kind="node")
            except Exception:
                results = []
            for result_node, sim_score in results:
                if result_node.node_id in seen_ids:
                    continue
                if (result_node.kind or "").strip().lower() != "episode":
                    continue
                if sim_score >= sim_threshold:
                    if result_node.node_id not in seen_ids:
                        related_older_episodes.append(result_node)
                        seen_ids.add(result_node.node_id)

        # Collect existing facts connected to those older episodes via DERIVED_FROM
        existing_facts: List[MemoryNode] = []
        if related_older_episodes:
            related_ep_ids = [ep.node_id for ep in related_older_episodes]
            try:
                connected_edges = self._graph_store.get_edges_for_nodes(related_ep_ids)
                fact_ids = set()
                for e in connected_edges:
                    etype = (getattr(e, "edge_type", "") or "").upper()
                    if etype == "DERIVED_FROM":
                        fact_ids.add(e.source_id)
                        fact_ids.add(e.target_id)
                if fact_ids:
                    all_nodes = self._node_store.get_nodes(list(fact_ids))
                    existing_facts = [
                        n for n in all_nodes
                        if (n.kind or "").strip().lower() == "fact"
                    ]
            except Exception as exc:
                logger.warning("Failed to retrieve existing facts: %s", exc)

        # Find existing concept nodes via vector search (for reuse by LLM)
        existing_concepts: List[MemoryNode] = []
        concept_seen: set = set()
        for ep_node in episode_nodes:
            ep_text = node_to_embed_text(ep_node)
            if not ep_text:
                continue
            try:
                results = self._vector_store.search(ep_text, filters, top_k=5, kind="node")
            except Exception:
                results = []
            for result_node, sim_score in results:
                if result_node.node_id in concept_seen:
                    continue
                if (result_node.kind or "").strip().lower() != "concept":
                    continue
                if sim_score >= sim_threshold:
                    existing_concepts.append(result_node)
                    concept_seen.add(result_node.node_id)

        # --- 2b. Single LLM call for facts + concepts ---
        extractor = LLMFactExtractor(llm)
        raw_facts, raw_concepts = extractor.extract_facts(
            episode_nodes, related_older_episodes, existing_facts, existing_concepts,
        )

        # --- 2c. Build nodes and edges ---

        now = datetime.utcnow()
        ep_id_set = {ep.node_id for ep in episode_nodes}

        # Build concept nodes first (so facts can reference them)
        new_concept_nodes: List[MemoryNode] = []
        concept_label_to_node: Dict[str, MemoryNode] = {}

        # Index existing concepts by label for reuse
        for c in existing_concepts:
            label = (getattr(c, "concept_label", "") or "").strip().lower()
            if label:
                concept_label_to_node[label] = c

        for item in (raw_concepts or []):
            label = str(item.get("concept_label", "")).strip().lower()
            if not label:
                continue
            # Skip if this concept already exists
            if label in concept_label_to_node:
                # Still create edges to existing concept
                concept_node = concept_label_to_node[label]
                episode_ids = item.get("episode_ids", [])
                for ep_id in episode_ids:
                    if ep_id in ep_id_set:
                        edges.append(make_edge(ep_id, concept_node.node_id, "HAS_CONCEPT"))
                continue

            node_id = canonical_id_entity(label, [])
            concept_node = MemoryNode(
                node_id=node_id,
                created_at=now,
                updated_at=now,
                kind="concept",
                concept_label=label,
                scopes=[scope],
                sequence=seq_start + len(new_concept_nodes) + 1,
                relevance_score=1.0,
            )
            new_concept_nodes.append(concept_node)
            concept_label_to_node[label] = concept_node

            # HAS_CONCEPT edges: episode -> concept
            episode_ids = item.get("episode_ids", [])
            for ep_id in episode_ids:
                if ep_id in ep_id_set:
                    edges.append(make_edge(ep_id, concept_node.node_id, "HAS_CONCEPT"))

        # Embed and upsert concept nodes
        if new_concept_nodes:
            self._upsert_node_embeddings(new_concept_nodes)

        # Build fact nodes
        new_fact_nodes: List[MemoryNode] = []
        fact_seq_start = seq_start + len(new_concept_nodes)

        for i, item in enumerate(raw_facts or []):
            fact_text = str(item.get("fact_text", "")).strip()
            if not fact_text:
                continue
            belief = item.get("belief", 1.0)
            try:
                belief = max(0.0, min(1.0, float(belief)))
            except (TypeError, ValueError):
                belief = 1.0

            source_episode_ids = item.get("source_episode_ids", [])

            node_id = canonical_id_entity(fact_text, [])
            fact_node = MemoryNode(
                node_id=node_id,
                created_at=now,
                updated_at=now,
                kind="fact",
                fact_text=fact_text,
                belief=belief,
                scopes=[scope],
                sequence=fact_seq_start + i + 1,
                relevance_score=1.0,
            )
            new_fact_nodes.append(fact_node)

            # DERIVED_FROM edges: fact -> source episodes
            for ep_id in source_episode_ids:
                if ep_id in ep_id_set:
                    edges.append(make_edge(fact_node.node_id, ep_id, "DERIVED_FROM"))

            # ABOUT_CONCEPT edges: fact -> concept
            fact_concepts = item.get("concepts", [])
            for concept_label in fact_concepts:
                label_lower = str(concept_label).strip().lower()
                concept_node = concept_label_to_node.get(label_lower)
                if concept_node:
                    edges.append(make_edge(
                        fact_node.node_id, concept_node.node_id, "ABOUT_CONCEPT",
                    ))

        if not new_fact_nodes:
            return [], new_concept_nodes

        # Embed fact nodes and upsert to vector store
        self._upsert_node_embeddings(new_fact_nodes)

        return new_fact_nodes, new_concept_nodes

    def _step3_generate_reflections(
        self,
        llm: LLMAdapter,
        new_fact_nodes: List[MemoryNode],
        scope: Scope,
        filters: QueryFilters,
        edges: List[Edge],
        seq_start: int,
    ) -> List[MemoryNode]:
        """Step 3: Generate reflections from facts using LLM.

        Retrieves similar existing facts and existing reflections for context,
        then calls the LLM to generate new reflections. Connects reflections
        to source facts via DERIVED_FROM_FACT edges.
        """
        from gaama.services.llm_extractors import LLMReflectionExtractor

        # Find similar existing facts for context
        related_facts: List[MemoryNode] = []
        seen_ids: set = {f.node_id for f in new_fact_nodes}
        for fact_node in new_fact_nodes:
            fact_text = node_to_embed_text(fact_node)
            if not fact_text:
                continue
            try:
                results = self._vector_store.search(fact_text, filters, top_k=5, kind="node")
            except Exception:
                results = []
            for result_node, sim_score in results:
                if result_node.node_id in seen_ids:
                    continue
                if (result_node.kind or "").strip().lower() != "fact":
                    continue
                if sim_score >= self._similarity_threshold:
                    related_facts.append(result_node)
                    seen_ids.add(result_node.node_id)

        # Find existing reflections connected to those related facts
        existing_reflections: List[MemoryNode] = []
        all_fact_ids = [f.node_id for f in new_fact_nodes] + [f.node_id for f in related_facts]
        if all_fact_ids:
            try:
                connected_edges = self._graph_store.get_edges_for_nodes(all_fact_ids)
                refl_ids = set()
                for e in connected_edges:
                    etype = (getattr(e, "edge_type", "") or "").upper()
                    if etype == "DERIVED_FROM_FACT":
                        refl_ids.add(e.source_id)
                        refl_ids.add(e.target_id)
                if refl_ids:
                    all_nodes = self._node_store.get_nodes(list(refl_ids))
                    existing_reflections = [
                        n for n in all_nodes
                        if (n.kind or "").strip().lower() == "reflection"
                    ]
            except Exception as exc:
                logger.warning("Failed to retrieve existing reflections: %s", exc)

        # Call LLM to generate reflections
        extractor = LLMReflectionExtractor(llm)
        raw_reflections = extractor.extract_reflections(new_fact_nodes, related_facts, existing_reflections)

        if not raw_reflections:
            return []

        # Build reflection MemoryNodes
        now = datetime.utcnow()
        new_reflection_nodes: List[MemoryNode] = []
        fact_id_set = {f.node_id for f in new_fact_nodes}

        for i, item in enumerate(raw_reflections):
            refl_text = str(item.get("reflection_text", "")).strip()
            if not refl_text:
                continue
            belief = item.get("belief", 1.0)
            try:
                belief = max(0.0, min(1.0, float(belief)))
            except (TypeError, ValueError):
                belief = 1.0

            source_fact_ids = item.get("source_fact_ids", [])

            node_id = canonical_id_entity(refl_text, [])
            refl_node = MemoryNode(
                node_id=node_id,
                created_at=now,
                updated_at=now,
                kind="reflection",
                reflection_text=refl_text,
                belief=belief,
                scopes=[scope],
                sequence=seq_start + i + 1,
                relevance_score=1.0,
            )
            new_reflection_nodes.append(refl_node)

            # DERIVED_FROM_FACT edges: reflection -> source facts
            for f_id in source_fact_ids:
                if f_id in fact_id_set:
                    edges.append(make_edge(refl_node.node_id, f_id, "DERIVED_FROM_FACT"))

        if not new_reflection_nodes:
            return []

        # Embed reflection nodes and upsert to vector store
        self._upsert_node_embeddings(new_reflection_nodes)

        return new_reflection_nodes

    # ------------------------------------------------------------------
    # Embedding helper (mirrors orchestrator._upsert_node_embeddings)
    # ------------------------------------------------------------------

    def _upsert_node_embeddings(self, nodes: list[MemoryNode]) -> int:
        if not self._embedder or not nodes:
            return 0
        if not hasattr(self._vector_store, "upsert_embeddings"):
            return 0

        # Collect texts and use batch embedding when available
        texts: list[str] = []
        text_indices: list[int] = []  # maps texts index -> nodes index
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

