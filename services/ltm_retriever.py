from __future__ import annotations
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from gaama.adapters import GraphStoreAdapter, NodeStoreAdapter
from gaama.core import (
    MemoryNode,
    MemoryPack,
    QueryFilters,
    RetrievalBudget,
)
from gaama.services.interfaces import (
    ForgettingEngine,
    IntegrationComposer,
    RetrieveOptions,
    RetrievalEngine,
)
from gaama.services.pagerank import (
    edges_from_core_edges,
    personalized_pagerank,
)

if TYPE_CHECKING:
    from gaama.adapters.interfaces import VectorStoreAdapter

# Node kinds mapped to budget slots
NODE_KIND_BUDGET_CONFIG = [
    ("fact", "max_facts"),
    ("reflection", "max_reflections"),
    ("skill", "max_skills"),
    ("episode", "max_episodes"),
]

NODE_KNN_K = 40
LAMBDA = 0.01
GEL_BELIEF = 1.0


def _scope_matches(node: MemoryNode, filters: QueryFilters) -> bool:
    """Check if node's scopes match all non-None fields in filters."""
    scopes = getattr(node, "scopes", None) or []
    if filters.agent_id is not None:
        if not any(s.agent_id == filters.agent_id for s in scopes):
            return False
    if filters.user_id is not None:
        if not any(s.user_id == filters.user_id for s in scopes):
            return False
    if filters.task_id is not None:
        if not any(s.task_id == filters.task_id for s in scopes):
            return False
    return True


def _belief_weight(node: MemoryNode) -> float:
    """Return belief-based weight: GEL-sourced nodes get GEL_BELIEF, others 1.0."""
    if node.tags.get("source") == "gel":
        return GEL_BELIEF
    return 1.0


def _node_content(node: MemoryNode) -> str:
    """Return the primary content string for a node based on its kind."""
    kind = (node.kind or "").strip().lower()
    if kind == "fact":
        text = (node.fact_text or "").strip()
    elif kind == "episode":
        text = (node.summary or "").strip()
    elif kind == "reflection":
        text = (node.reflection_text or "").strip()
    elif kind == "skill":
        text = (node.skill_description or node.name or "").strip()
    elif kind == "entity":
        text = (node.name or "").strip()
    else:
        text = (node.name or "").strip()
    # Prepend timestamp if available

    #if session_date and text:
    #    return f"{session_date}: {text}"
    return text


class NodeKNNPageRankRetrievalEngine(RetrievalEngine):
    """LTM retrieval: node KNN -> seed distribution -> PPR on graph edges ->
    additive score (w1*PPR + w2*sim) -> pack by kind.
    """

    def __init__(
        self,
        node_store: NodeStoreAdapter,
        graph_store: GraphStoreAdapter,
        vector_store: "VectorStoreAdapter",
        node_knn_k: int = NODE_KNN_K,
        ppr_alpha: float = 0.6,
        ppr_max_iterations: int = 100,
        expansion_depth: int = 1,
    ) -> None:
        self._node_store = node_store
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._node_knn_k = node_knn_k
        self._ppr_alpha = ppr_alpha
        self._ppr_max_iterations = ppr_max_iterations
        self._expansion_depth = expansion_depth

    def retrieve(
        self, query: str, options: RetrieveOptions
    ) -> Tuple[MemoryPack, List[Tuple[str, str, float]]]:
        """Retrieve memories; returns (MemoryPack, list of (node_id, content, relevance_score)).
        relevance_score is the final score used to sort the retrieved memories.
        Order of the list matches pack order: facts, reflections, skills, episodes.
        """
        if getattr(options, "semantic_only", False):
            return self._retrieve_semantic(query, options)

        if getattr(options, "hybrid", False):
            return self._retrieve_hybrid(query, options)

        budget = options.budget
        filters = options.filters
        ppr_w = getattr(options, "ppr_score_weight", None)
        sim_w = getattr(options, "sim_score_weight", None)
        if ppr_w is not None and sim_w is not None:
            ppr_score_weight = ppr_w
            sim_score_weight = sim_w
        else:
            ppr_score_weight = 1.0
            sim_score_weight = 1.0
        deg_corr = getattr(options, "degree_correction", None)
        degree_correction = bool(deg_corr) if deg_corr is not None else False
        exp_depth = getattr(options, "expansion_depth", None)
        expansion_depth = int(exp_depth) if exp_depth is not None else self._expansion_depth
        edge_type_weights = getattr(options, "edge_type_weights", None)
        # 1) KNN with 2x budget nodes
        total_budget = sum(getattr(budget, slot, 0) for _, slot in NODE_KIND_BUDGET_CONFIG)
        knn_top_k = max(2 * total_budget, 2 * self._node_knn_k)
        knn_results: Sequence[Tuple[MemoryNode, float]] = self._vector_store.search(
            query, filters, top_k=knn_top_k, kind="node"
        )
        if not knn_results:
            return MemoryPack(), []

        # 2) Seed selection by sim squared -- top-k nodes become PPR starting mass.
        knn_nodes: Dict[str, MemoryNode] = {}
        selection_scores: Dict[str, float] = {}
        sim_scores: Dict[str, float] = {}
        for node, sim in knn_results:
            nid = node.node_id
            knn_nodes[nid] = node
            sim_val = max(0.0, float(sim))
            selection_scores[nid] = sim_val #** 2
            sim_scores[nid] = sim_val

        _knn_original_nids = set(knn_nodes.keys())  # track which nodes came from KNN vs graph
        sorted_nids = sorted(selection_scores.keys(), key=lambda n: selection_scores[n], reverse=True)
        top_nids = sorted_nids[: self._node_knn_k]

        seed_weights = {nid: selection_scores[nid] for nid in top_nids}

        total = sum(seed_weights.values())
        if total <= 0:
            return MemoryPack(), []
        seeds = {nid: w / total for nid, w in seed_weights.items()}

        # 3) Graph expansion from seed nodes
        seed_ids = list(seeds.keys())
        expanded_edges = list(self._graph_store.query_edges(seed_ids, expansion_depth))

        # 4) Collect all node IDs from edges; fetch missing nodes
        all_node_ids = set()
        for e in expanded_edges:
            src = getattr(e, "source_id", None)
            tgt = getattr(e, "target_id", None)
            if src is not None:
                all_node_ids.add(src)
            if tgt is not None:
                all_node_ids.add(tgt)
        missing_ids = [nid for nid in all_node_ids if nid not in knn_nodes]
        if missing_ids:
            for n in self._node_store.get_nodes(missing_ids):
                knn_nodes[n.node_id] = n

        knn_nodes = {
            nid: n for nid, n in knn_nodes.items()
            if _scope_matches(n, filters)
        }

        # 4b) Compute sim scores for graph-discovered nodes that have no sim score.
        #     These nodes were found via graph expansion but not in the KNN results,
        #     so they have sim=0.  We compute their similarity to the query embedding
        #     so that the final score (ppr + sim) is fair.
        #     Only keep graph-discovered nodes above a sim threshold to prevent
        #     hub-connected noise from flooding the candidate pool.
        missing_sim_ids = [nid for nid in knn_nodes if nid not in sim_scores]
        if missing_sim_ids and hasattr(self._vector_store, 'compute_similarity'):
            extra_sims = self._vector_store.compute_similarity(query, missing_sim_ids)
            sim_scores.update(extra_sims)

        # 5) Build edge transition weights; PPR
        edge_tuples = edges_from_core_edges(
            expanded_edges,
            edge_type_weights=edge_type_weights,
        )
        ppr_scores = personalized_pagerank(
            seeds, edge_tuples,
            alpha=self._ppr_alpha,
            max_iterations=self._ppr_max_iterations,
            degree_correction=degree_correction,
        )

        # 6) Rank: max-normalize both PPR and sim, then additive scoring.
        #    Ensures both signals contribute equally on a [0, 1] scale.
        # Merge all KNN nodes into final candidate list (KNN nodes not in PPR get ppr=0)
        all_candidate_ids = set(knn_nodes.keys())

        # Max-normalize PPR scores
        max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0
        ppr_norm = {nid: ppr_scores.get(nid, 0.0) / max_ppr if max_ppr > 0 else 0.0
                    for nid in all_candidate_ids}

        # Max-normalize sim scores
        all_sims = [sim_scores.get(nid, 0.0) for nid in all_candidate_ids]
        max_sim = max(all_sims) if all_sims else 1.0
        sim_norm = {nid: sim_scores.get(nid, 0.0) / max_sim if max_sim > 0 else 0.0
                    for nid in all_candidate_ids}

        scored: List[Tuple[float, MemoryNode]] = []
        for nid, node in knn_nodes.items():
            ppr = ppr_norm.get(nid, 0.0)
            sim = sim_norm.get(nid, 0.0)
            score = ppr_score_weight * ppr + sim_score_weight * sim
            score *= _belief_weight(node)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 7) Budget per kind -> MemoryPack and (node_id, content, score) in same order
        budgetless = getattr(options, "budgetless", False)
        max_words = getattr(options, "max_memory_words", 600)
        buckets: Dict[str, List[str]] = {kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG}
        scored_items_by_kind: Dict[str, List[Tuple[str, str, float]]] = {kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG}
        episode_candidates: List[Tuple[str, str, float]] = []
        max_episodes = getattr(budget, "max_episodes", 0)
        word_count = 0
        for score_val, node in scored:
            kind = (node.kind or "").strip().lower()
            if kind not in buckets:
                continue
            content = _node_content(node)
            if not content:
                continue
            if budgetless:
                item_words = len(content.split())
                if word_count + item_words > max_words and word_count > 0:
                    continue
                if kind == "episode":
                    episode_candidates.append((node.node_id, content, score_val))
                else:
                    buckets[kind].append(content)
                    scored_items_by_kind[kind].append((node.node_id, content, score_val))
                word_count += item_words
            else:
                cap_field = next((slot for k, slot in NODE_KIND_BUDGET_CONFIG if k == kind), None)
                cap = getattr(budget, cap_field, 0) if cap_field else 0
                if kind == "episode":
                    if len(episode_candidates) < max_episodes:
                        episode_candidates.append((node.node_id, content, score_val))
                elif len(buckets[kind]) < cap:
                    buckets[kind].append(content)
                    scored_items_by_kind[kind].append((node.node_id, content, score_val))

        # Order episodes by sequence (ascending); legacy nodes with no sequence sort last
        if episode_candidates:
            def _seq_key(nid: str):
                node = knn_nodes.get(nid)
                seq = getattr(node, "sequence", None) if node else None
                return (seq is None, seq if seq is not None else 0)
            ordered_episode_ids = sorted(
                [nid for nid, _, _ in episode_candidates],
                key=_seq_key,
            )
            by_id = {nid: (c, s) for nid, c, s in episode_candidates}
            buckets["episode"] = [by_id[nid][0] for nid in ordered_episode_ids]
            scored_items_by_kind["episode"] = [(nid, by_id[nid][0], by_id[nid][1]) for nid in ordered_episode_ids]

        pack = MemoryPack(
            facts=buckets.get("fact", []),
            reflections=buckets.get("reflection", []),
            skills=buckets.get("skill", []),
            episodes=buckets.get("episode", []),
            scores={
                "facts": [s for _, _, s in scored_items_by_kind.get("fact", [])],
                "reflections": [s for _, _, s in scored_items_by_kind.get("reflection", [])],
                "skills": [s for _, _, s in scored_items_by_kind.get("skill", [])],
                "episodes": [s for _, _, s in scored_items_by_kind.get("episode", [])],
            },
        )
        scored_items: List[Tuple[str, str, float]] = []
        for kind, _ in NODE_KIND_BUDGET_CONFIG:
            scored_items.extend(scored_items_by_kind.get(kind, []))
        return pack, scored_items

    # ------------------------------------------------------------------
    # Semantic-only retrieval (no graph expansion, no PPR)
    # ------------------------------------------------------------------
    def _retrieve_semantic(
        self, query: str, options: RetrieveOptions
    ) -> Tuple[MemoryPack, List[Tuple[str, str, float]]]:
        """Pure semantic KNN retrieval -- no graph expansion or PPR.

        Retrieves the top nodes by cosine similarity across all node kinds
        (facts, episodes, reflections, skills), applies budget caps, and
        returns a MemoryPack.  Faster than the PPR path and provides a
        clean baseline for measuring graph-based retrieval improvements.
        """
        budget = options.budget
        filters = options.filters

        # Total budget across all kinds
        total_budget = sum(
            getattr(budget, slot, 0) for _, slot in NODE_KIND_BUDGET_CONFIG
        )
        top_k = max(10 * total_budget, 2 * self._node_knn_k)

        # KNN search
        knn_results: Sequence[Tuple[MemoryNode, float]] = self._vector_store.search(
            query, filters, top_k=top_k, kind="node"
        )
        if not knn_results:
            return MemoryPack(), []

        # Filter by scope (agent_id, user_id, task_id)
        knn_results = [
            (n, s) for n, s in knn_results
            if _scope_matches(n, filters)
        ]

        # Score = similarity * belief weight (descending)
        scored: List[Tuple[float, MemoryNode]] = [
            (max(0.0, float(sim)) * _belief_weight(node), node)
            for node, sim in knn_results if sim > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Budget per kind -> MemoryPack
        budgetless = getattr(options, "budgetless", False)
        max_words = getattr(options, "max_memory_words", 600)
        buckets: Dict[str, List[str]] = {kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG}
        scored_items_by_kind: Dict[str, List[Tuple[str, str, float]]] = {
            kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG
        }
        episode_candidates: List[Tuple[str, str, float]] = []
        max_episodes = getattr(budget, "max_episodes", 0)
        node_map: Dict[str, MemoryNode] = {}
        word_count = 0

        for score_val, node in scored:
            kind = (node.kind or "").strip().lower()
            if kind not in buckets:
                continue
            content = _node_content(node)
            if not content:
                continue
            if budgetless:
                item_words = len(content.split())
                if word_count + item_words > max_words and word_count > 0:
                    continue
                if kind == "episode":
                    episode_candidates.append((node.node_id, content, score_val))
                    node_map[node.node_id] = node
                else:
                    buckets[kind].append(content)
                    scored_items_by_kind[kind].append((node.node_id, content, score_val))
                word_count += item_words
            else:
                cap_field = next((slot for k, slot in NODE_KIND_BUDGET_CONFIG if k == kind), None)
                cap = getattr(budget, cap_field, 0) if cap_field else 0
                if kind == "episode":
                    if len(episode_candidates) < max_episodes:
                        episode_candidates.append((node.node_id, content, score_val))
                        node_map[node.node_id] = node
                elif len(buckets[kind]) < cap:
                    buckets[kind].append(content)
                    scored_items_by_kind[kind].append((node.node_id, content, score_val))

        # Order episodes by temporal sequence
        if episode_candidates:
            def _seq_key(nid: str):
                node = node_map.get(nid)
                seq = getattr(node, "sequence", None) if node else None
                return (seq is None, seq if seq is not None else 0)
            ordered_ids = sorted(
                [nid for nid, _, _ in episode_candidates], key=_seq_key
            )
            by_id = {nid: (c, s) for nid, c, s in episode_candidates}
            buckets["episode"] = [by_id[nid][0] for nid in ordered_ids]
            scored_items_by_kind["episode"] = [
                (nid, by_id[nid][0], by_id[nid][1]) for nid in ordered_ids
            ]

        pack = MemoryPack(
            facts=buckets.get("fact", []),
            reflections=buckets.get("reflection", []),
            skills=buckets.get("skill", []),
            episodes=buckets.get("episode", []),
            scores={
                "facts": [s for _, _, s in scored_items_by_kind.get("fact", [])],
                "reflections": [s for _, _, s in scored_items_by_kind.get("reflection", [])],
                "skills": [s for _, _, s in scored_items_by_kind.get("skill", [])],
                "episodes": [s for _, _, s in scored_items_by_kind.get("episode", [])],
            },
        )
        scored_items: List[Tuple[str, str, float]] = []
        for kind, _ in NODE_KIND_BUDGET_CONFIG:
            scored_items.extend(scored_items_by_kind.get(kind, []))
        return pack, scored_items


    def _retrieve_hybrid(
        self, query: str, options: RetrieveOptions, llm=None, llm_model: str = "gpt-4o-mini",
    ) -> Tuple[MemoryPack, List[Tuple[str, str, float]]]:
        """Hybrid retrieval: semantic base + selective PPR upgrades.

        1. Run semantic and PPR retrieval in parallel.
        2. Use semantic as the base pack.
        3. For each PPR-discovered item NOT in semantic:
           - Only upgrade if it scores higher than the weakest semantic item
             OF THE SAME TYPE (same-type replacement only).
           - Reflections: always keep semantic's (PPR reflections are noise).
        4. Trim to max_memory_words.
        """
        import concurrent.futures

        budget = options.budget
        max_words = getattr(options, "max_memory_words", 600)

        # --- Run both retrievers in parallel ---
        sem_options = RetrieveOptions(
            budget=budget, filters=options.filters,
            ppr_score_weight=0.0, sim_score_weight=1.0,
            semantic_only=True,
        )
        ppr_options = RetrieveOptions(
            budget=budget, filters=options.filters,
            ppr_score_weight=getattr(options, "ppr_score_weight", 1.0),
            sim_score_weight=getattr(options, "sim_score_weight", 1.0),
            edge_type_weights=getattr(options, "edge_type_weights", None),
            expansion_depth=getattr(options, "expansion_depth", None),
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_sem = pool.submit(self._retrieve_semantic, query, sem_options)
            fut_ppr = pool.submit(self.retrieve, query, ppr_options)
            sem_pack, sem_items = fut_sem.result()
            ppr_pack, ppr_items = fut_ppr.result()

        # --- Build score maps from retrieval items ---
        def _score_map(items: List[Tuple[str, str, float]]) -> Dict[str, float]:
            m: Dict[str, float] = {}
            for _, content, score in items:
                key = content.strip()[:200]
                if key not in m or score > m[key]:
                    m[key] = score
            return m

        sem_score_map = _score_map(sem_items)
        ppr_score_map = _score_map(ppr_items)

        # --- Start with semantic pack as base ---
        # For each type, build (content, score) sorted by score
        result_by_kind: Dict[str, List[Tuple[str, float]]] = {}
        for kind, attr in [("fact", "facts"), ("episode", "episodes"),
                           ("reflection", "reflections"), ("skill", "skills")]:
            items = []
            for content in getattr(sem_pack, attr, []):
                key = content.strip()[:200]
                score = sem_score_map.get(key, 0.0)
                items.append((content, score))
            items.sort(key=lambda x: -x[1])
            result_by_kind[kind] = items

        # --- Collect PPR-only items by type ---
        sem_keys = set()
        for kind, attr in [("fact", "facts"), ("episode", "episodes"),
                           ("reflection", "reflections"), ("skill", "skills")]:
            for content in getattr(sem_pack, attr, []):
                sem_keys.add(content.strip()[:200])

        # --- Build sim-only score map for PPR items (fair comparison with semantic) ---
        # PPR items get their sim score (not ppr+sim) for comparison
        # This way both semantic and PPR items are scored on the same scale
        ppr_only_by_kind: Dict[str, List[Tuple[str, float]]] = {
            "fact": [], "episode": [], "reflection": [], "skill": []
        }
        for kind, attr in [("fact", "facts"), ("episode", "episodes"),
                           ("reflection", "reflections"), ("skill", "skills")]:
            for content in getattr(ppr_pack, attr, []):
                key = content.strip()[:200]
                if key not in sem_keys:
                    # Use sem_score_map if available (same embedding sim),
                    # otherwise fall back to ppr_score but halved as rough sim estimate
                    sim_score = sem_score_map.get(key, 0.0)
                    if sim_score == 0.0:
                        # Item not in semantic's KNN -- compute approximate sim
                        # from ppr_score (ppr+sim) by subtracting estimated ppr component
                        ppr_combined = ppr_score_map.get(key, 0.0)
                        sim_score = max(0.0, ppr_combined - 0.5)  # rough: subtract avg ppr
                    ppr_only_by_kind[kind].append((content, sim_score))
            ppr_only_by_kind[kind].sort(key=lambda x: -x[1])

        # --- Same-type upgrade: PPR items replace weakest semantic items ---
        # Skip reflections -- keep semantic's reflections only
        # Both sides use sim scores for fair comparison
        for kind in ["episode", "fact"]:
            sem_items_list = result_by_kind[kind]
            ppr_candidates = ppr_only_by_kind[kind]

            if not sem_items_list or not ppr_candidates:
                continue

            for ppr_content, ppr_sim in ppr_candidates:
                # Find weakest semantic item of this type
                weakest_content, weakest_score = sem_items_list[-1]

                if ppr_sim > weakest_score:
                    # Replace weakest with PPR item (using sim score)
                    sem_items_list[-1] = (ppr_content, ppr_sim)
                    # Re-sort to maintain order
                    sem_items_list.sort(key=lambda x: -x[1])

        # --- Build final pack (count-capped only, word trimming done by eval pipeline) ---
        buckets: Dict[str, List[str]] = {kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG}
        scored_items: List[Tuple[str, str, float]] = []

        for kind, _ in NODE_KIND_BUDGET_CONFIG:
            cap_field = next((slot for k, slot in NODE_KIND_BUDGET_CONFIG if k == kind), None)
            cap = getattr(budget, cap_field, 0) if cap_field else 0
            items = result_by_kind.get(kind, [])[:cap]
            for content, score in items:
                buckets[kind].append(content)
                scored_items.append(("", content, score))

        # Sort episodes chronologically
        if buckets.get("episode"):
            buckets["episode"].sort()

        pack = MemoryPack(
            facts=buckets.get("fact", []),
            reflections=buckets.get("reflection", []),
            skills=buckets.get("skill", []),
            episodes=buckets.get("episode", []),
        )
        return pack, scored_items


class LTMForgettingEngine(ForgettingEngine):
    """LTM forgetting: query nodes by filters and return candidate IDs for deletion."""

    def __init__(self, node_store: NodeStoreAdapter) -> None:
        self._node_store = node_store

    def forget(self, selector: QueryFilters) -> Sequence[str]:
        candidates = self._node_store.query(selector, limit=100)
        return [node.node_id for node in candidates]


class LTMIntegrationComposer(IntegrationComposer):
    """Compose MemoryPack node content into prompt-ready format."""

    def compose(self, query: str, memory_pack: MemoryPack, mode: str) -> dict:
        prompt_lines = [f"Query: {query}", "Facts:"]
        for t in memory_pack.facts:
            prompt_lines.append(f"- {t}")
        prompt_lines.append("Reflections:")
        for t in memory_pack.reflections:
            prompt_lines.append(f"- {t}")
        prompt_lines.append("Skills:")
        for t in memory_pack.skills:
            prompt_lines.append(f"- {t}")
        prompt_lines.append("Episodes:")
        for t in memory_pack.episodes:
            prompt_lines.append(f"- {t}")
        return {
            "prompt_pack": "\n".join(prompt_lines),
            "structured_state": {
                "facts": list(memory_pack.facts),
                "reflections": list(memory_pack.reflections),
                "skills": list(memory_pack.skills),
                "episodes": list(memory_pack.episodes),
            },
            "tool_hints": list(memory_pack.skills),
        }
