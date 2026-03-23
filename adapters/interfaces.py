from __future__ import annotations

from typing import Iterable, Protocol, Sequence, Tuple, Union

from gaama.core import Edge, MemoryNode, QueryFilters


class EmbeddingAdapter(Protocol):
    """Produces vector embeddings for text."""

    def embed(self, text: str) -> Sequence[float]:
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call. Default falls back to sequential."""
        ...


class GraphStoreAdapter(Protocol):
    def upsert_nodes(self, nodes: Iterable[MemoryNode]) -> Sequence[str]:
        ...

    def upsert_edges(self, edges: Iterable[Edge]) -> Sequence[str]:
        ...

    def query_neighbors(self, node_ids: Sequence[str], depth: int) -> Sequence[MemoryNode]:
        ...

    def query_edges(self, node_ids: Sequence[str], depth: int) -> Sequence[Edge]:
        """Return edges in the subgraph induced by BFS from node_ids up to depth."""
        ...

    def get_edges_for_nodes(self, node_ids: Sequence[str]) -> Sequence[Edge]:
        """Return all edges where source_id OR target_id is in *node_ids* (1-hop)."""
        ...

    def get_edges_by_ids(self, edge_ids: Sequence[str]) -> Sequence[Edge]:
        """Return edges whose edge_id is in *edge_ids*."""
        ...


class VectorStoreAdapter(Protocol):
    """Store and search node embeddings."""

    def upsert_embeddings(
        self,
        items: Iterable[MemoryNode],
        agent_id: str | None = None,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> Sequence[str]:
        """Persist node embeddings (uses node.embedding). Returns node_ids."""
        ...

    def search(
        self,
        query: str,
        filters: QueryFilters,
        top_k: int,
        kind: str = "node",
    ) -> Sequence[Tuple[MemoryNode, float]]:
        """KNN search over node embeddings. Returns (node, similarity) pairs."""
        ...


class NodeStoreAdapter(Protocol):
    """Node retrieval interface: persist and look up memory nodes by id or filters."""

    def upsert_nodes(self, nodes: Iterable[MemoryNode]) -> Sequence[str]:
        ...

    def get_nodes(self, node_ids: Sequence[str]) -> Sequence[MemoryNode]:
        ...

    def query(self, filters: QueryFilters, limit: int) -> Sequence[MemoryNode]:
        ...

    def get_last_episode_node(self, agent_id: str) -> MemoryNode | None:
        """Return the most recent episode node for the given agent_id, or None."""
        ...

    def get_max_episode_sequence(self, agent_id: str) -> int:
        """Return the maximum sequence among episode nodes for agent_id, or 0 if none."""
        ...

    def get_max_sequence(self, agent_id: str) -> int:
        """Return the maximum sequence among all nodes for agent_id, or 0 if none."""
        ...


class BlobStoreAdapter(Protocol):
    def put_blob(self, key: str, data: bytes) -> None:
        ...

    def get_blob(self, key: str) -> bytes:
        ...


class LLMAdapter(Protocol):
    """Protocol for LLM completion."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        ...
