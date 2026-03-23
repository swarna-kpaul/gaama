"""Hybrid search: BM25 (FTS5) + semantic (vector) in parallel, with score fusion.

Used by NodeCanonicalizer, EdgeCanonicalizer, and LTM retrieval engines. BM25 and semantic
search run in separate threads; results are fused with configurable weights.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

from gaama.core import QueryFilters

if TYPE_CHECKING:
    from gaama.adapters.interfaces import EmbeddingAdapter, VectorStoreAdapter
    from gaama.adapters.sqlite_memory import SqliteMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Weights and limits for hybrid search."""

    bm25_weight: float = 0.4
    semantic_weight: float = 0.6


def _run_bm25(
    memory_store: "SqliteMemoryStore",
    query: str,
    filters: QueryFilters,
    limit: int,
    kind: str = "node",
) -> List[Tuple[str, float]]:
    if kind == "edge":
        raw = memory_store.search_fts_edges(query, filters=filters, limit=limit)
    else:
        raw = memory_store.search_fts(query, filters=filters, limit=limit)
    if not raw:
        return []
    max_score = raw[0][1]
    if max_score <= 0:
        return [(eid, 0.0) for eid, _ in raw]
    return [(eid, score / max_score) for eid, score in raw]


def _run_semantic(
    vector_store: "VectorStoreAdapter",
    query: str,
    filters: QueryFilters,
    limit: int,
    kind: str = "node",
) -> List[Tuple[str, float]]:
    try:
        if kind == "edge":
            # vector_store.search(..., kind='edge') returns Sequence[Tuple[Edge, float]]
            results = vector_store.search(query, filters, limit, kind="edge")
            return [(e.edge_id, s) for e, s in results]
        if hasattr(vector_store, "search_with_scores"):
            results = vector_store.search_with_scores(query, filters, limit)
            return [(n.node_id, s) for n, s in results]
        nodes = vector_store.search(query, filters, limit)
        return [(n.node_id, max(0.0, 1.0 - i * 0.05)) for i, n in enumerate(nodes)]
    except Exception:
        logger.debug("Semantic search failed in hybrid search", exc_info=True)
        return []


def _fuse(
    bm25_hits: List[Tuple[str, float]],
    semantic_hits: List[Tuple[str, float]],
    bm25_weight: float,
    semantic_weight: float,
) -> List[Tuple[str, float]]:
    all_ids: dict[str, dict[str, float]] = {}
    for nid, score in bm25_hits:
        all_ids.setdefault(nid, {"bm25": 0.0, "semantic": 0.0})
        all_ids[nid]["bm25"] = score
    for nid, score in semantic_hits:
        all_ids.setdefault(nid, {"bm25": 0.0, "semantic": 0.0})
        all_ids[nid]["semantic"] = score
    if not all_ids:
        return []
    w_bm25, w_sem = bm25_weight, semantic_weight
    if not bm25_hits and semantic_hits:
        w_bm25, w_sem = 0.0, 1.0
    elif bm25_hits and not semantic_hits:
        w_bm25, w_sem = 1.0, 0.0
    fused = [
        (nid, w_bm25 * s["bm25"] + w_sem * s["semantic"])
        for nid, s in all_ids.items()
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


class HybridSearcher:
    """
    Runs BM25 (SQLite FTS5) and semantic (vector) search in parallel, then
    fuses scores. Requires memory_store (for FTS) and vector_store + embedder
    (for semantic). Callers use search() to get (node_id, score) list.
    """

    def __init__(
        self,
        memory_store: "SqliteMemoryStore",
        vector_store: "VectorStoreAdapter",
        embedder: Optional["EmbeddingAdapter"] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._config = config or HybridSearchConfig()

    def search(
        self,
        query: str,
        filters: QueryFilters,
        top_k: int,
        kind: str = "node",
    ) -> List[Tuple[str, float]]:
        """
        Run BM25 and semantic search in parallel, fuse scores, return
        (entity_id, fused_score) sorted descending. For kind='node' entity_id
        is node_id; for kind='edge' it is edge_id.
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = pool.submit(
                _run_bm25,
                self._memory_store,
                query,
                filters,
                top_k,
                kind,
            )
            fut_sem = pool.submit(
                _run_semantic,
                self._vector_store,
                query,
                filters,
                top_k,
                kind,
            )
            bm25_hits = fut_bm25.result()
            semantic_hits = fut_sem.result()

        fused = _fuse(
            bm25_hits,
            semantic_hits,
            self._config.bm25_weight,
            self._config.semantic_weight,
        )
        return fused[:top_k]
