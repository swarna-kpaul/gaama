"""No-op vector store when embeddings are not configured (LTM retrieval returns empty)."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from gaama.core import MemoryNode, QueryFilters


class NullEdgeVectorStore:
    """Vector store that never stores or returns anything. Used when no embedder is configured."""

    def upsert_embeddings(
        self,
        items: Iterable[MemoryNode],
        agent_id: str | None = None,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> Sequence[str]:
        return []

    def search(
        self,
        query: str,
        filters: QueryFilters,
        top_k: int,
        kind: str = "node",
    ) -> Sequence[Tuple[MemoryNode, float]]:
        return []

    def clear_ltm(self) -> None:
        pass
