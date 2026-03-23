"""Vector store backed by SQLite using the sqlite-vec extension (vec0 virtual table).

Stores node embeddings only.  Edge embeddings are no longer used.
"""
from __future__ import annotations

import struct
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence, Tuple, TYPE_CHECKING

from gaama.adapters.interfaces import EmbeddingAdapter, VectorStoreAdapter
from gaama.core import MemoryNode, QueryFilters

if TYPE_CHECKING:
    from gaama.adapters.interfaces import NodeStoreAdapter

try:
    from sqlite_vec import serialize_float32 as _serialize_f32_lib
except ImportError:
    _serialize_f32_lib = None


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _serialize_f32(vec: Sequence[float]) -> bytes:
    if _serialize_f32_lib is not None:
        return _serialize_f32_lib(vec)
    return struct.pack(f"{len(vec)}f", *vec)


class SqliteVectorStore(VectorStoreAdapter):
    """Vector store using SQLite + sqlite-vec.  Node embeddings only."""

    def __init__(
        self,
        path: Path,
        embedder: EmbeddingAdapter | None = None,
        node_store: "NodeStoreAdapter | None" = None,
        dimension: int = 1536,
    ) -> None:
        self._path = Path(path)
        self._embedder = embedder
        self._node_store = node_store
        self._dimension = dimension
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path.as_posix(), timeout=60)
        _load_sqlite_vec(conn)
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL DEFAULT 'node',
                    entity_id TEXT NOT NULL,
                    agent_id TEXT,
                    user_id TEXT,
                    task_id TEXT
                )
                """
            )
            # Drop old unique constraint if present (migration from single-scope)
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_embedding_rows_entity ON embedding_rows(kind, entity_id)"
                )
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                        id INTEGER PRIMARY KEY,
                        embedding float[{self._dimension}] distance_metric=cosine
                    )
                    """
                )
            except sqlite3.OperationalError:
                pass

    # ------------------------------------------------------------------

    def upsert_embeddings(
        self,
        items: Iterable[MemoryNode],
        agent_id: str | None = None,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> Sequence[str]:
        entity_ids: list[str] = []
        for node in items:
            embedding = getattr(node, "embedding", None)
            if not embedding:
                continue
            emb = list(embedding)[: self._dimension]
            if len(emb) != self._dimension:
                continue
            eid = node.node_id
            scopes = getattr(node, "scopes", None) or []
            if not scopes:
                scopes_to_write = [{"agent_id": agent_id, "user_id": user_id, "task_id": task_id}]
            else:
                scopes_to_write = [
                    {"agent_id": s.agent_id, "user_id": s.user_id, "task_id": s.task_id}
                    for s in scopes
                ]

            with self._connect() as conn:
                # Delete all existing rows for this entity
                old_rows = conn.execute(
                    "SELECT id FROM embedding_rows WHERE kind = 'node' AND entity_id = ?",
                    (eid,),
                ).fetchall()
                for (old_id,) in old_rows:
                    conn.execute("DELETE FROM vec_embeddings WHERE id = ?", (old_id,))
                conn.execute(
                    "DELETE FROM embedding_rows WHERE kind = 'node' AND entity_id = ?",
                    (eid,),
                )
                # Insert one row per scope, all sharing the same embedding vector
                emb_blob = _serialize_f32(emb)
                for scope_dict in scopes_to_write:
                    conn.execute(
                        "INSERT INTO embedding_rows(kind, entity_id, agent_id, user_id, task_id) VALUES ('node', ?, ?, ?, ?)",
                        (eid, scope_dict["agent_id"], scope_dict["user_id"], scope_dict["task_id"]),
                    )
                    (rid,) = conn.execute("SELECT last_insert_rowid()").fetchone()
                    conn.execute(
                        "INSERT INTO vec_embeddings(id, embedding) VALUES (?, ?)",
                        (rid, emb_blob),
                    )
                entity_ids.append(eid)
        return entity_ids

    def clear_ltm(self) -> None:
        try:
            with self._connect() as conn:
                try:
                    conn.execute("DELETE FROM vec_embeddings")
                except sqlite3.OperationalError:
                    pass
                conn.execute("DELETE FROM embedding_rows")
                conn.commit()
        except Exception:
            conn = sqlite3.connect(self._path.as_posix(), timeout=60)
            try:
                conn.execute("DELETE FROM embedding_rows")
                conn.commit()
            finally:
                conn.close()

    def search(
        self,
        query: str,
        filters: QueryFilters,
        top_k: int,
        kind: str = "node",
    ) -> Sequence[Tuple[MemoryNode, float]]:
        if not self._embedder:
            raise RuntimeError("Embedding adapter is required for vector search.")
        query_vec = list(self._embedder.embed(query))[: self._dimension]
        if len(query_vec) != self._dimension:
            return []
        query_blob = _serialize_f32(query_vec)
        # Fetch extra to account for multi-scope duplicates
        k = max(top_k * 5, 20)

        allowed_ids: list[str] | None = None
        if self._node_store and (
            filters.tags or filters.workspace_id is not None or filters.time_range is not None
        ):
            candidates = self._node_store.query(filters, limit=100_000)
            allowed_ids = [n.node_id for n in candidates]
            if not allowed_ids:
                return []

        with self._connect() as conn:
            where_clauses = ["v.embedding MATCH ?", "k = ?", "e.kind = 'node'"]
            params: list[object] = [query_blob, k]
            if filters.agent_id is not None:
                where_clauses.append("e.agent_id = ?")
                params.append(filters.agent_id)
            if filters.user_id is not None:
                where_clauses.append("e.user_id = ?")
                params.append(filters.user_id)
            if filters.task_id is not None:
                where_clauses.append("e.task_id = ?")
                params.append(filters.task_id)
            if allowed_ids is not None:
                placeholders = ",".join(["?"] * len(allowed_ids))
                where_clauses.append(f"e.entity_id IN ({placeholders})")
                params.extend(allowed_ids)
            rows = conn.execute(
                f"""
                SELECT e.entity_id, v.distance
                FROM vec_embeddings v
                JOIN embedding_rows e ON e.id = v.id
                WHERE {' AND '.join(where_clauses)}
                """,
                tuple(params),
            ).fetchall()

        if not rows or not self._node_store:
            return []

        # Deduplicate by entity_id, keeping the best (lowest) distance
        best_distance: dict[str, float] = {}
        for entity_id, distance in rows:
            d = float(distance or 0)
            if entity_id not in best_distance or d < best_distance[entity_id]:
                best_distance[entity_id] = d

        unique_ids = list(best_distance.keys())
        nodes = self._node_store.get_nodes(unique_ids)
        by_id = {n.node_id: n for n in nodes}

        scored: list[Tuple[MemoryNode, float]] = []
        for entity_id, dist in sorted(best_distance.items(), key=lambda x: x[1]):
            node = by_id.get(entity_id)
            if not node:
                continue
            scored.append((node, max(0.0, 1.0 - dist / 2.0)))
            if len(scored) >= top_k:
                break
        return scored

    def search_with_scores(
        self, query: str, filters: QueryFilters, top_k: int
    ) -> Sequence[Tuple[MemoryNode, float]]:
        return self.search(query, filters, top_k, kind="node")
