"""SQLite-backed memory store: nodes, edges, FTS5 (BM25) full-text index.

Vector embeddings use the same DB via SqliteVectorStore (sqlite-vec).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from gaama.adapters.interfaces import GraphStoreAdapter, NodeStoreAdapter
from gaama.core import Edge, MemoryNode, QueryFilters, Scope
from gaama.infra.serialization import deserialize_node, node_to_embed_text, serialize_node


def _node_class_str(node: MemoryNode) -> str:
    kind = getattr(node, "kind", None)
    return (kind or "MemoryNode").strip() or "MemoryNode"


class SqliteMemoryStore(NodeStoreAdapter, GraphStoreAdapter):
    """SQLite store: nodes table, edges table, and FTS5 for BM25 search."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path.as_posix(), timeout=60)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_class TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            try:
                legacy_col = "node_" + "type"
                conn.execute(f"ALTER TABLE nodes RENAME COLUMN {legacy_col} TO node_class")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    edge_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    weight REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            # Keep label column; add if missing
            try:
                conn.execute("ALTER TABLE edges ADD COLUMN label TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            # Legacy belief_score column: keep for backward compat reads
            try:
                conn.execute("ALTER TABLE edges ADD COLUMN belief_score REAL NOT NULL DEFAULT 1.0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE edges DROP COLUMN metadata")
            except sqlite3.OperationalError:
                pass
            # Recreate FTS table if schema is old/incompatible
            fts_info = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='node_fts'"
            ).fetchone()
            if fts_info:
                cols = [str(row[1]) for row in conn.execute("PRAGMA table_info(node_fts)").fetchall()]
                expected = ["node_id", "agent_id", "user_id", "task_id", "content"]
                if cols != expected:
                    try:
                        conn.execute("DROP TABLE node_fts")
                    except sqlite3.OperationalError:
                        pass
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(
                        node_id UNINDEXED,
                        agent_id UNINDEXED,
                        user_id UNINDEXED,
                        task_id UNINDEXED,
                        content
                    )
                    """
                )
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS edge_fts USING fts5(
                        edge_id UNINDEXED,
                        agent_id UNINDEXED,
                        user_id UNINDEXED,
                        task_id UNINDEXED,
                        content
                    )
                    """
                )
            except sqlite3.OperationalError:
                pass

    # ---- NodeStoreAdapter ----

    def upsert_nodes(self, nodes: Iterable[MemoryNode]) -> Sequence[str]:
        node_ids: list[str] = []
        with self._connect() as conn:
            for node in nodes:
                # Merge scopes with existing node on conflict
                existing_row = conn.execute(
                    "SELECT payload FROM nodes WHERE node_id = ?", (node.node_id,)
                ).fetchone()
                if existing_row:
                    existing_node = deserialize_node(existing_row[0])
                    merged_scopes = list(existing_node.scopes)
                    for s in node.scopes:
                        if s not in merged_scopes:
                            merged_scopes.append(s)
                    node.scopes = merged_scopes

                payload = serialize_node(node)
                conn.execute(
                    """
                    INSERT INTO nodes (node_id, node_class, payload, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        node_class=excluded.node_class,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at
                    """,
                    (node.node_id, _node_class_str(node), payload, node.updated_at.isoformat()),
                )
                node_ids.append(node.node_id)
                self._upsert_fts_in_conn(conn, node)
        return node_ids

    def _upsert_fts_in_conn(self, conn: sqlite3.Connection, node: MemoryNode) -> None:
        try:
            conn.execute("DELETE FROM node_fts WHERE node_id = ?", (node.node_id,))
            text = node_to_embed_text(node)
            if text:
                scopes = getattr(node, "scopes", None) or []
                if not scopes:
                    conn.execute(
                        "INSERT INTO node_fts(node_id, agent_id, user_id, task_id, content) VALUES (?, ?, ?, ?, ?)",
                        (node.node_id, None, None, None, text),
                    )
                else:
                    for s in scopes:
                        conn.execute(
                            "INSERT INTO node_fts(node_id, agent_id, user_id, task_id, content) VALUES (?, ?, ?, ?, ?)",
                            (node.node_id, s.agent_id, s.user_id, s.task_id, text),
                        )
        except sqlite3.OperationalError:
            pass

    def insert_fts(
        self, node_id: str, content: str, *,
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
        scopes: Sequence[Scope] | None = None,
    ) -> None:
        if not content:
            return
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM node_fts WHERE node_id = ?", (node_id,))
                if scopes:
                    for s in scopes:
                        conn.execute(
                            "INSERT INTO node_fts(node_id, agent_id, user_id, task_id, content) VALUES (?, ?, ?, ?, ?)",
                            (node_id, s.agent_id, s.user_id, s.task_id, content),
                        )
                else:
                    conn.execute(
                        "INSERT INTO node_fts(node_id, agent_id, user_id, task_id, content) VALUES (?, ?, ?, ?, ?)",
                        (node_id, agent_id, user_id, task_id, content),
                    )
        except sqlite3.OperationalError:
            pass

    def delete_fts(self, node_id: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM node_fts WHERE node_id = ?", (node_id,))
        except sqlite3.OperationalError:
            pass

    def search_fts(
        self, query: str, filters: QueryFilters | None = None, limit: int = 10,
    ) -> Sequence[Tuple[str, float]]:
        try:
            allowed_ids: list[str] | None = None
            if filters is not None and (
                filters.tags or filters.workspace_id is not None or filters.time_range is not None
            ):
                candidates = self.query(filters, limit=100_000)
                allowed_ids = [n.node_id for n in candidates]
                if not allowed_ids:
                    return []
            with self._connect() as conn:
                where_clauses = ["node_fts MATCH ?"]
                params: list[object] = [query]
                if filters and filters.agent_id is not None:
                    where_clauses.append("agent_id = ?")
                    params.append(filters.agent_id)
                if filters and filters.user_id is not None:
                    where_clauses.append("user_id = ?")
                    params.append(filters.user_id)
                if filters and filters.task_id is not None:
                    where_clauses.append("task_id = ?")
                    params.append(filters.task_id)
                if allowed_ids is not None:
                    placeholders = ",".join(["?"] * len(allowed_ids))
                    where_clauses.append(f"node_id IN ({placeholders})")
                    params.extend(allowed_ids)
                # Fetch extra rows to account for multi-scope duplicates
                fetch_limit = limit * 3
                params.append(fetch_limit)
                rows = conn.execute(
                    f"""
                    SELECT node_id, bm25(node_fts) AS sc
                    FROM node_fts
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY sc
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
                # Deduplicate by node_id, keeping the best BM25 score
                seen: dict[str, float] = {}
                for row in rows:
                    nid = row[0]
                    score = float(-row[1]) if row[1] is not None else 0.0
                    if nid not in seen or score > seen[nid]:
                        seen[nid] = score
                result = sorted(seen.items(), key=lambda x: x[1], reverse=True)
                return result[:limit]
        except sqlite3.OperationalError:
            return []

    def fts_doc_count(self) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM node_fts").fetchone()
                return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    # ---- Edge FTS (for EdgeCanonicalizer label matching) ----

    def insert_fts_edge(
        self, edge_id: str, content: str, *,
        agent_id: str | None = None, user_id: str | None = None, task_id: str | None = None,
    ) -> None:
        if not content:
            return
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM edge_fts WHERE edge_id = ?", (edge_id,))
                conn.execute(
                    "INSERT INTO edge_fts(edge_id, agent_id, user_id, task_id, content) VALUES (?, ?, ?, ?, ?)",
                    (edge_id, agent_id, user_id, task_id, content),
                )
        except sqlite3.OperationalError:
            pass

    def delete_fts_edge(self, edge_id: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM edge_fts WHERE edge_id = ?", (edge_id,))
        except sqlite3.OperationalError:
            pass

    def search_fts_edges(
        self, query: str, filters: QueryFilters | None = None, limit: int = 10,
    ) -> Sequence[Tuple[str, float]]:
        try:
            with self._connect() as conn:
                where_clauses = ["edge_fts MATCH ?"]
                params: list[object] = [query]
                if filters and filters.agent_id is not None:
                    where_clauses.append("agent_id = ?")
                    params.append(filters.agent_id)
                if filters and filters.user_id is not None:
                    where_clauses.append("user_id = ?")
                    params.append(filters.user_id)
                if filters and filters.task_id is not None:
                    where_clauses.append("task_id = ?")
                    params.append(filters.task_id)
                params.append(limit)
                rows = conn.execute(
                    f"""
                    SELECT edge_id, bm25(edge_fts) AS sc
                    FROM edge_fts
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY sc
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
                return [(row[0], float(-row[1]) if row[1] is not None else 0.0) for row in rows]
        except sqlite3.OperationalError:
            return []

    def clear_ltm(self) -> None:
        with self._connect() as conn:
            try:
                conn.execute("DELETE FROM node_fts")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("DELETE FROM edge_fts")
            except sqlite3.OperationalError:
                pass
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            conn.commit()

    def get_nodes(self, node_ids: Sequence[str]) -> Sequence[MemoryNode]:
        if not node_ids:
            return []
        placeholders = ",".join(["?"] * len(node_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload FROM nodes WHERE node_id IN ({placeholders})",
                tuple(node_ids),
            ).fetchall()
        return [deserialize_node(row[0]) for row in rows]

    def query(self, filters: QueryFilters, limit: int) -> Sequence[MemoryNode]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM nodes").fetchall()
        nodes = [deserialize_node(row[0]) for row in rows]
        matched = [node for node in nodes if _matches_filters(node, filters)]
        return matched[:limit]

    def get_last_episode_node(self, agent_id: str) -> MemoryNode | None:
        """Return the episode node for *agent_id* that has no outgoing NEXT
        edge to another episode node (i.e. the tail of the temporal chain)."""
        with self._connect() as conn:
            # Filter in DB: episode nodes for this agent_id, excluding those
            # that have an outgoing NEXT edge (tail of chain only)
            rows = conn.execute(
                """
                SELECT n.node_id, n.payload
                FROM nodes n
                WHERE n.node_class = 'episode'
                  AND (
                    EXISTS (
                      SELECT 1 FROM json_each(json_extract(n.payload, '$.scopes')) AS s
                      WHERE json_extract(s.value, '$.agent_id') = ?
                    )
                    OR json_extract(n.payload, '$.scope.agent_id') = ?
                  )
                  AND n.node_id NOT IN (
                    SELECT source_id FROM edges WHERE edge_type = 'NEXT'
                  )
                LIMIT 1
                """,
                (agent_id, agent_id),
            ).fetchall()
        if not rows:
            return None
        return deserialize_node(rows[0][1])

    def get_max_episode_sequence(self, agent_id: str) -> int:
        """Return the maximum sequence among episode nodes for agent_id, or 0 if none."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(CAST(json_extract(n.payload, '$.sequence') AS INTEGER))
                FROM nodes n
                WHERE n.node_class = 'episode'
                  AND (
                    EXISTS (
                      SELECT 1 FROM json_each(json_extract(n.payload, '$.scopes')) AS s
                      WHERE json_extract(s.value, '$.agent_id') = ?
                    )
                    OR json_extract(n.payload, '$.scope.agent_id') = ?
                  )
                """,
                (agent_id, agent_id),
            ).fetchone()
        if not row or row[0] is None:
            return 0
        try:
            return max(0, int(row[0]))
        except (TypeError, ValueError):
            return 0

    def get_max_sequence(self, agent_id: str) -> int:
        """Return the maximum sequence among all nodes for agent_id, or 0 if none."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(CAST(json_extract(n.payload, '$.sequence') AS INTEGER))
                FROM nodes n
                WHERE (
                    EXISTS (
                        SELECT 1 FROM json_each(json_extract(n.payload, '$.scopes')) AS s
                        WHERE json_extract(s.value, '$.agent_id') = ?
                    )
                    OR json_extract(n.payload, '$.scope.agent_id') = ?
                )
                """,
                (agent_id, agent_id),
            ).fetchone()
        if not row or row[0] is None:
            return 0
        try:
            return max(0, int(row[0]))
        except (TypeError, ValueError):
            return 0

    # ---- GraphStoreAdapter ----

    def upsert_edges(self, edges: Iterable[Edge]) -> Sequence[str]:
        edge_ids: list[str] = []
        with self._connect() as conn:
            for edge in edges:
                edge_type_str = getattr(edge.edge_type, "value", str(edge.edge_type))
                label = getattr(edge, "label", "")
                conn.execute(
                    """
                    INSERT INTO edges (edge_id, edge_type, source_id, target_id, weight, created_at, belief_score, label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(edge_id) DO UPDATE SET
                        edge_type=excluded.edge_type,
                        source_id=excluded.source_id,
                        target_id=excluded.target_id,
                        weight=excluded.weight,
                        created_at=excluded.created_at,
                        belief_score=excluded.belief_score,
                        label=excluded.label
                    """,
                    (
                        edge.edge_id,
                        edge_type_str,
                        edge.source_id,
                        edge.target_id,
                        edge.weight,
                        edge.created_at.isoformat(),
                        1.0,
                        label,
                    ),
                )
                edge_ids.append(edge.edge_id)
        return edge_ids

    def query_neighbors(self, node_ids: Sequence[str], depth: int) -> Sequence[MemoryNode]:
        if not node_ids or depth <= 0:
            return []
        visited = set(node_ids)
        frontier = set(node_ids)
        with self._connect() as conn:
            for _ in range(depth):
                if not frontier:
                    break
                placeholders = ",".join(["?"] * len(frontier))
                rows = conn.execute(
                    f"""
                    SELECT source_id, target_id FROM edges
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                    """,
                    tuple(frontier) + tuple(frontier),
                ).fetchall()
                next_frontier = set()
                for source_id, target_id in rows:
                    if source_id not in visited:
                        next_frontier.add(source_id)
                    if target_id not in visited:
                        next_frontier.add(target_id)
                visited.update(next_frontier)
                frontier = next_frontier
        if not visited:
            return []
        return list(self.get_nodes(list(visited)))

    def query_edges(self, node_ids: Sequence[str], depth: int) -> Sequence[Edge]:
        if not node_ids or depth <= 0:
            return []
        visited = set(node_ids)
        frontier = set(node_ids)
        with self._connect() as conn:
            for _ in range(depth):
                if not frontier:
                    break
                placeholders = ",".join(["?"] * len(frontier))
                rows = conn.execute(
                    f"""
                    SELECT source_id, target_id FROM edges
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                    """,
                    tuple(frontier) + tuple(frontier),
                ).fetchall()
                next_frontier = set()
                for source_id, target_id in rows:
                    if source_id not in visited:
                        next_frontier.add(source_id)
                    if target_id not in visited:
                        next_frontier.add(target_id)
                visited.update(next_frontier)
                frontier = next_frontier
        if not visited:
            return []
        placeholders = ",".join(["?"] * len(visited))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT edge_id, edge_type, source_id, target_id, weight, created_at, label
                FROM edges
                WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
                """,
                tuple(visited) + tuple(visited),
            ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def get_edges_for_nodes(self, node_ids: Sequence[str]) -> Sequence[Edge]:
        if not node_ids:
            return []
        placeholders = ",".join(["?"] * len(node_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT edge_id, edge_type, source_id, target_id, weight, created_at, label
                FROM edges
                WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                """,
                tuple(node_ids) + tuple(node_ids),
            ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def get_edges_by_ids(self, edge_ids: Sequence[str]) -> Sequence[Edge]:
        if not edge_ids:
            return []
        placeholders = ",".join(["?"] * len(edge_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT edge_id, edge_type, source_id, target_id, weight, created_at, label
                FROM edges
                WHERE edge_id IN ({placeholders})
                """,
                tuple(edge_ids),
            ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    def list_all_edges(self, limit: int = 50_000) -> Sequence[Edge]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT edge_id, edge_type, source_id, target_id, weight, created_at, label FROM edges LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_edge(row) for row in rows]

    @staticmethod
    def _row_to_edge(row: tuple) -> Edge:
        edge_id, edge_type_s, source_id, target_id, weight, created_at_s, label = row
        try:
            created_at = datetime.fromisoformat(created_at_s)
        except (TypeError, ValueError):
            created_at = datetime.utcnow()
        return Edge(
            edge_id=edge_id,
            edge_type=str(edge_type_s) if edge_type_s else "",
            source_id=source_id,
            target_id=target_id,
            created_at=created_at,
            label=str(label) if label else "",
            weight=float(weight) if weight is not None else 1.0,
        )


def _matches_filters(node: MemoryNode, filters: QueryFilters) -> bool:
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
    for key, value in filters.tags.items():
        if node.tags.get(key) != value:
            return False
    return True


def get_first_episode_in_batch(
    episode_nodes: Sequence[MemoryNode], edges: Sequence[Edge]
) -> MemoryNode | None:
    """Return the episode in *episode_nodes* that has no incoming NEXT edge
    (head of the temporal chain in this batch)."""
    next_targets = {
        e.target_id
        for e in edges
        if (getattr(e, "edge_type", None) or "").strip().upper() == "NEXT"
    }
    for n in episode_nodes:
        if n.node_id not in next_targets:
            return n
    return None
