"""Hybrid BM25 + semantic-vector canonicalization service.

When a new memory node is about to be persisted, the ``NodeCanonicalizer``
checks whether a semantically equivalent node already exists using a shared
``HybridSearcher``.  If the best fused score exceeds a configurable threshold
the **existing** node's ID is reused; otherwise a hash-based canonical ID is
used.

Edge canonicalization resolves each new edge via label matching so duplicate
or semantically equivalent edges reuse the same edge_id.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Sequence, TYPE_CHECKING

from gaama.core import Edge, MemoryNode, QueryFilters
from gaama.infra.serialization import node_to_embed_text

if TYPE_CHECKING:
    from gaama.adapters.sqlite_memory import SqliteMemoryStore
    from gaama.services.hybrid_search import HybridSearcher


# ---------------------------------------------------------------------------
# Stable ID helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _hash_prefix(content: str, prefix: str, length: int = 16) -> str:
    if not content:
        content = ""
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{prefix}-{h[:length]}"


def canonical_id_entity(name: str, aliases: Sequence[str] | None = None) -> str:
    norm_name = normalize_text(name)
    if aliases:
        norm_aliases = "|".join(sorted(normalize_text(a) for a in aliases))
        content = f"entity|{norm_name}|{norm_aliases}"
    else:
        content = f"entity|{norm_name}" if norm_name else f"entity|{hash(name)}"
    return _hash_prefix(content, "entity")


def canonical_id_fact(fact_text: str) -> str:
    """Stable node_id for Fact: based on normalized fact_text."""
    norm = normalize_text(fact_text)
    content = f"fact|{norm}" if norm else f"fact|{hash(fact_text)}"
    return _hash_prefix(content, "fact")


def canonical_id_note_fact(key: str, value: str) -> str:
    norm_key = normalize_text(key)
    norm_val = normalize_text(str(value))
    content = f"fact|stm|{norm_key}|{norm_val}" if (norm_key or norm_val) else f"fact|stm|{hash((key, value))}"
    return _hash_prefix(content, "fact")


def canonical_id_task(task_id: str) -> str:
    norm = normalize_text(task_id)
    return f"task-{norm}" if norm else f"task-{hash(task_id)}"


def canonical_id_edge(source_id: str, target_id: str, label: str, edge_type: str = "") -> str:
    norm_s = normalize_text(source_id)
    norm_t = normalize_text(target_id)
    norm_l = normalize_text(label)
    norm_et = normalize_text(edge_type or "RELATED_TO")
    content = f"edge|{norm_s}|{norm_t}|{norm_l}|{norm_et}"
    return _hash_prefix(content, "edge")


# ---------------------------------------------------------------------------
# Configuration / result types
# ---------------------------------------------------------------------------


@dataclass
class CanonicalizationConfig:
    match_threshold: float = 0.95
    top_k: int = 10


@dataclass
class CanonicalMatch:
    node_id: str
    matched_existing: bool
    match_score: float
    match_source: str


@dataclass
class EdgeMatch:
    edge_id: str
    matched_existing: bool
    match_score: float
    match_source: str


# ---------------------------------------------------------------------------
# Node canonicalization
# ---------------------------------------------------------------------------


class NodeCanonicalizer:
    """Hybrid BM25 + semantic canonicalization for nodes."""

    def __init__(
        self,
        config: CanonicalizationConfig | None = None,
        memory_store: "SqliteMemoryStore | None" = None,
        hybrid_searcher: "HybridSearcher | None" = None,
    ) -> None:
        self._config = config or CanonicalizationConfig()
        self._memory_store = memory_store
        self._hybrid_searcher = hybrid_searcher

    @property
    def indexed_count(self) -> int:
        if self._memory_store is not None:
            return self._memory_store.fts_doc_count()
        return 0

    def index_node(self, node: MemoryNode) -> None:
        text = node_to_embed_text(node)
        if not text or self._memory_store is None:
            return
        scopes = getattr(node, "scopes", None) or []
        self._memory_store.insert_fts(
            node.node_id, text, scopes=scopes if scopes else None,
        )

    def index_nodes(self, nodes: Sequence[MemoryNode]) -> int:
        if self._memory_store is None:
            return 0
        for node in nodes:
            text = node_to_embed_text(node)
            if text:
                scopes = getattr(node, "scopes", None) or []
                self._memory_store.insert_fts(
                    node.node_id, text, scopes=scopes if scopes else None,
                )
        return sum(1 for n in nodes if node_to_embed_text(n))

    def remove_node(self, node_id: str) -> None:
        if self._memory_store is not None:
            self._memory_store.delete_fts(node_id)

    def resolve_node(self, node: MemoryNode) -> CanonicalMatch:
        text = node_to_embed_text(node)
        if not text or self._hybrid_searcher is None:
            return CanonicalMatch(node_id=node.node_id, matched_existing=False, match_score=0.0, match_source="hash_fallback")
        filters = QueryFilters()
        fused = self._hybrid_searcher.search(text, filters, self._config.top_k)
        if fused:
            best_id, best_score = fused[0]
            if best_score >= self._config.match_threshold and best_id != node.node_id:
                return CanonicalMatch(node_id=best_id, matched_existing=True, match_score=best_score, match_source="fused")
            return CanonicalMatch(node_id=node.node_id, matched_existing=False, match_score=best_score, match_source="hash_fallback")
        return CanonicalMatch(node_id=node.node_id, matched_existing=False, match_score=0.0, match_source="hash_fallback")

    def resolve_nodes(self, nodes: Sequence[MemoryNode]) -> List[CanonicalMatch]:
        return [self.resolve_node(n) for n in nodes]


# ---------------------------------------------------------------------------
# Edge canonicalization (label-based BM25 matching)
# ---------------------------------------------------------------------------


def _edge_fts_content(edge: Edge) -> str:
    """Content for edge matching: label + edge_type."""
    label = (getattr(edge, "label", None) or "").strip()
    et = (getattr(edge, "edge_type", None) or "").strip()
    s = (getattr(edge, "source_id", None) or "").strip()
    t = (getattr(edge, "target_id", None) or "").strip()
    return " ".join(filter(None, [label, et, s, t]))


class EdgeCanonicalizer:
    """Label-based canonicalization for edges.

    Resolves each new edge via hybrid search over edge labels.
    When a match above threshold is found with the same endpoints,
    reuses that edge_id so duplicates are not created.
    """

    def __init__(
        self,
        config: CanonicalizationConfig | None = None,
        memory_store: "SqliteMemoryStore | None" = None,
        hybrid_searcher: "HybridSearcher | None" = None,
        graph_store=None,
    ) -> None:
        self._config = config or CanonicalizationConfig()
        self._memory_store = memory_store
        self._hybrid_searcher = hybrid_searcher
        self._graph_store = graph_store

    def resolve_edge(
        self, edge: Edge,
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
    ) -> EdgeMatch:
        fallback_id = canonical_id_edge(
            edge.source_id, edge.target_id,
            getattr(edge, "label", "") or "",
            getattr(edge, "edge_type", "") or "",
        )
        query = _edge_fts_content(edge)
        if not query or self._hybrid_searcher is None:
            return EdgeMatch(edge_id=fallback_id, matched_existing=False, match_score=0.0, match_source="hash_fallback")
        filters = QueryFilters(agent_id=agent_id, user_id=user_id, task_id=task_id)
        fused = self._hybrid_searcher.search(query, filters, self._config.top_k, kind="edge")
        if not fused:
            return EdgeMatch(edge_id=fallback_id, matched_existing=False, match_score=0.0, match_source="hash_fallback")
        best_id, best_score = fused[0]
        if best_score < self._config.match_threshold:
            return EdgeMatch(edge_id=fallback_id, matched_existing=False, match_score=best_score, match_source="hash_fallback")
        if self._graph_store and best_id != edge.edge_id:
            existing = self._graph_store.get_edges_by_ids([best_id])
            if existing:
                ex = existing[0]
                if ex.source_id == edge.source_id and ex.target_id == edge.target_id:
                    return EdgeMatch(edge_id=best_id, matched_existing=True, match_score=best_score, match_source="fused")
        return EdgeMatch(edge_id=fallback_id, matched_existing=False, match_score=best_score, match_source="hash_fallback")

    def index_edge(
        self, edge: Edge,
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
    ) -> None:
        if self._memory_store is None:
            return
        content = _edge_fts_content(edge)
        if content and hasattr(self._memory_store, "insert_fts_edge"):
            self._memory_store.insert_fts_edge(edge.edge_id, content, agent_id=agent_id, user_id=user_id, task_id=task_id)

    def index_edges(
        self, edges: Sequence[Edge],
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
    ) -> int:
        if self._memory_store is None:
            return 0
        count = 0
        for edge in edges:
            content = _edge_fts_content(edge)
            if content and hasattr(self._memory_store, "insert_fts_edge"):
                self._memory_store.insert_fts_edge(edge.edge_id, content, agent_id=agent_id, user_id=user_id, task_id=task_id)
                count += 1
        return count

    def remove_edge(self, edge_id: str) -> None:
        if self._memory_store is not None and hasattr(self._memory_store, "delete_fts_edge"):
            self._memory_store.delete_fts_edge(edge_id)
