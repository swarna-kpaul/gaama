"""Build graph edges from extracted nodes and LLM edge specs."""
from __future__ import annotations

from datetime import datetime
from typing import List, Sequence
from uuid import uuid4

from gaama.core import ALLOWED_EDGE_TYPES, Edge, MemoryNode
from gaama.services.interfaces import EdgeSpec


def _normalize_edge_type(raw: str) -> str:
    """Return a valid edge type; default RELATED_TO."""
    v = (raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    if v in ALLOWED_EDGE_TYPES:
        return v
    return "RELATED_TO"


def build_edges_from_nodes(
    nodes: Sequence[MemoryNode],
    edge_specs: Sequence[EdgeSpec] = (),
) -> List[Edge]:
    """Build Edge list from a batch of nodes and edge specs.

    Indices in each EdgeSpec refer to the *nodes* list order.
    """
    now = datetime.utcnow()
    edges: List[Edge] = []
    node_list = list(nodes)
    n = len(node_list)

    for spec in edge_specs:
        if spec.source_index < 0 or spec.source_index >= n or spec.target_index < 0 or spec.target_index >= n:
            continue
        source_id = node_list[spec.source_index].node_id
        target_id = node_list[spec.target_index].node_id
        if source_id == target_id:
            continue
        edges.append(Edge(
            edge_id=f"edge-{uuid4().hex}",
            edge_type=_normalize_edge_type(spec.edge_type),
            source_id=source_id,
            target_id=target_id,
            created_at=now,
            label=spec.label or "",
            weight=1.0,
        ))

    return edges
