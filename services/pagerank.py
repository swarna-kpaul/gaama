"""Local personalized PageRank for knowledge-graph retrieval.

Edge weights define transition probabilities.  Edge-type-aware base weights
control how much PPR mass flows along each relationship type.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

EdgeTuple = Tuple[str, str, float]

# ---------------------------------------------------------------------------
# Edge-type-aware base weights for PPR transition probabilities.
# These modulate the stored edge.weight before per-source normalization.  Higher weight = more PPR mass flows along
# that edge type.
# ---------------------------------------------------------------------------
DEFAULT_EDGE_TYPE_WEIGHTS: Dict[str, float] = {
    "SUBJECT":       1.0,
    "OBJECT":        1.0,
    "INVOLVES":      0.9,
    "MENTIONS":      0.7,
    "TRIGGERED_BY":  0.7,
    "ABOUT":         0.7,
    "SUPPORTED_BY":  0.6,
    "PRODUCED":      0.6,
    "DERIVED_FROM":  0.6,
    "LEARNED_FROM":  0.6,
    "USES_TOOL":     0.6,
    "APPLIES_TO":    0.6,
    "RELATED_TO":    0.5,
    "CONTRADICTS":   0.5,
    "REFINES":       0.5,
    "NEXT":          0.5,
}

_DEFAULT_EDGE_TYPE_FALLBACK = 0.5


def personalized_pagerank(
    seed_weights: dict[str, float],
    edges: Sequence[EdgeTuple],
    *,
    alpha: float = 0.85,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    degree_correction: bool = False,
) -> Dict[str, float]:
    """Compute personalized PageRank on a local subgraph.

    Returns PPR scores normalized to [0, 1] via max-normalization.
    If *degree_correction* is True, divides raw PPR by log(1 + degree)
    before normalizing.
    """
    if not seed_weights:
        return {}

    total_seed = sum(seed_weights.values())
    if total_seed <= 0:
        return {nid: 0.0 for nid in seed_weights}
    v: Dict[str, float] = {nid: w / total_seed for nid, w in seed_weights.items()}

    node_ids: set[str] = set(v.keys())
    out_edges: Dict[str, List[Tuple[str, float]]] = {}
    for source_id, target_id, weight in edges:
        node_ids.add(source_id)
        node_ids.add(target_id)
        w = max(0.0, float(weight))
        out_edges.setdefault(source_id, []).append((target_id, w))

    out_degree: Dict[str, float] = {}
    for nid, targets in out_edges.items():
        out_degree[nid] = sum(w for _, w in targets)
    for nid in node_ids:
        out_degree.setdefault(nid, 0.0)

    for nid in node_ids:
        v.setdefault(nid, 0.0)

    in_contrib: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in node_ids}
    for source_id, targets in out_edges.items():
        deg = out_degree[source_id]
        if deg <= 0:
            continue
        for target_id, w in targets:
            in_contrib[target_id].append((source_id, w / deg))

    r = dict(v)
    for _ in range(max_iterations):
        sink_mass = sum(r.get(i, 0.0) for i in node_ids if out_degree[i] <= 0)
        r_new = {}
        for j in node_ids:
            incoming = (1.0 - alpha + alpha * sink_mass) * v[j]
            for i, frac in in_contrib.get(j, []):
                incoming += alpha * r.get(i, 0.0) * frac
            r_new[j] = incoming
        diff = sum(abs(r_new.get(n, 0) - r.get(n, 0)) for n in node_ids)
        r = r_new
        if diff < tolerance:
            break

    if degree_correction:
        deg: Dict[str, int] = {nid: len(out_edges.get(nid, [])) for nid in node_ids}
        scores = {nid: r[nid] / math.log(1 + max(1, deg[nid])) for nid in node_ids}
    else:
        scores = dict(r)

    # Max-normalize to [0, 1] so PPR is on the same scale as similarity scores
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0:
        return scores
    return {nid: scores[nid] / max_score for nid in node_ids}


def edges_from_core_edges(
    core_edges: Sequence,
    edge_type_weights: Optional[Dict[str, float]] = None,
    hub_dampening_threshold: int = 50,
) -> List[EdgeTuple]:
    """Convert ``Edge`` objects to ``(source_id, target_id, P_ij)`` transition weights.

    Each edge's base weight is looked up from *edge_type_weights* (defaults to
    ``DEFAULT_EDGE_TYPE_WEIGHTS``) and multiplied by the stored ``edge.weight``.
    Both directions are created per core edge, then per-source normalised so
    for each source i, sum_j P_ij = 1 (row-stochastic).

    *hub_dampening_threshold* caps the effective outgoing degree of any single
    node.  Nodes with degree > threshold have their outgoing edge weights
    scaled by ``threshold / degree``, preventing mega-hub entities from
    distributing PPR mass uniformly across hundreds of neighbours.
    Set to 0 to disable hub dampening.
    """
    _type_w = edge_type_weights if edge_type_weights is not None else DEFAULT_EDGE_TYPE_WEIGHTS

    directed: List[Tuple[str, str, float]] = []
    for e in core_edges:
        src = getattr(e, "source_id", None)
        tgt = getattr(e, "target_id", None)
        edge_type = getattr(e, "edge_type", "RELATED_TO") or "RELATED_TO"
        base_w = _type_w.get(edge_type, _DEFAULT_EDGE_TYPE_FALLBACK)
        stored_w = max(0.0, float(getattr(e, "weight", 1.0)))
        w = base_w * stored_w
        if src is None or tgt is None:
            continue
        directed.append((src, tgt, w))
        directed.append((tgt, src, w))

    # Group by source to compute degree and apply hub dampening
    by_source: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for s, t, w in directed:
        by_source[s].append((t, w))

    # Hub dampening: scale down outgoing weights for high-degree nodes
    if hub_dampening_threshold > 0:
        for s, targets in by_source.items():
            deg = len(targets)
            if deg > hub_dampening_threshold:
                scale = hub_dampening_threshold / deg
                by_source[s] = [(t, w * scale) for t, w in targets]

    # Per-source normalization: P_ij = w_tilde_ij / sum_k w_tilde_ik
    out: List[EdgeTuple] = []
    for s, targets in by_source.items():
        total = sum(w for _, w in targets)
        if total <= 0:
            continue
        for t, w in targets:
            out.append((s, t, w / total))
    return out

