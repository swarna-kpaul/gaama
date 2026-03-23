from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from gaama.core import LTMDirectoryEntry
from gaama.infra.vector_math import cosine_similarity


class LTMDirectoryIndex:
    def __init__(self) -> None:
        self._entries: Dict[str, LTMDirectoryEntry] = {}
        self._embeddings: Dict[str, Sequence[float]] = {}
        self._agent_ids: Dict[str, Optional[str]] = {}

    def upsert(self, entries: Iterable[LTMDirectoryEntry]) -> None:
        for entry in entries:
            self._entries[entry.node_id] = entry
            if entry.embedding:
                self._embeddings[entry.node_id] = list(entry.embedding)
            self._agent_ids[entry.node_id] = getattr(entry, "agent_id", None)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        agent_id: Optional[str] = None,
    ) -> Sequence[LTMDirectoryEntry]:
        if not self._embeddings:
            return []
        scores = []
        for node_id, vector in self._embeddings.items():
            if agent_id is not None and self._agent_ids.get(node_id) != agent_id:
                continue
            score = cosine_similarity(query_vector, vector)
            scores.append((score, self._entries[node_id]))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scores[:top_k]]
