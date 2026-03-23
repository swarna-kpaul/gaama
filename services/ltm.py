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

def _node_content(node: MemoryNode) -> str:
    """Return the primary content string for a node based on its kind."""
    kind = (node.kind or "").strip().lower()
    if kind == "fact":
        text = (node.fact_text or "").strip()
    elif kind == "episode":
        text = (node.summary or "").strip()
        session_date = (node.tags or {}).get("session_date", "")
        text = f"[{session_date}] {text}"
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
        adaptive_ppr_model = getattr(options, "adaptive_ppr_model", None)
        # 1) KNN with 2× top_k
        knn_results: Sequence[Tuple[MemoryNode, float]] = self._vector_store.search(
            query, filters, top_k=2 * self._node_knn_k, kind="node"
        )
        if not knn_results:
            return MemoryPack(), []

        # 2) Seed selection by sim² — top-k nodes become PPR starting mass.
        knn_nodes: Dict[str, MemoryNode] = {}
        selection_scores: Dict[str, float] = {}
        sim_scores: Dict[str, float] = {}
        for node, sim in knn_results:
            nid = node.node_id
            knn_nodes[nid] = node
            sim_val = max(0.0, float(sim))
            selection_scores[nid] = sim_val ** 2
            sim_scores[nid] = sim_val

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

        if filters.agent_id is not None:
            knn_nodes = {
                nid: n for nid, n in knn_nodes.items()
                if any(s.agent_id == filters.agent_id for s in (n.scopes or []))
            }

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

        # 6) Rank: additive scoring.
        #    score = ppr_w * ppr + sim_w * sim
        #    If adaptive_ppr_model is set, override ppr_score_weight per query.
        if adaptive_ppr_model is not None:
            query_emb = list(self._vector_store._embedder.embed(query))
            ppr_score_weight = adaptive_ppr_model.predict(query_emb)

        scored: List[Tuple[float, MemoryNode]] = []
        for nid, node in knn_nodes.items():
            ppr = ppr_scores.get(nid, 0.0)
            sim = sim_scores.get(nid, 0.0)
            score = ppr_score_weight * ppr + sim_score_weight * sim
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 7) Budget per kind -> MemoryPack and (node_id, content, score) in same order
        buckets: Dict[str, List[str]] = {kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG}
        scored_items_by_kind: Dict[str, List[Tuple[str, str, float]]] = {kind: [] for kind, _ in NODE_KIND_BUDGET_CONFIG}
        episode_candidates: List[Tuple[str, str, float]] = []
        max_episodes = getattr(budget, "max_episodes", 0)
        for score_val, node in scored:
            kind = (node.kind or "").strip().lower()
            if kind not in buckets:
                continue
            content = _node_content(node)
            if not content:
                continue
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
