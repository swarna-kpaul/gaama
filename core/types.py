"""Canonical types for the Agentic Memory SDK.

Five node kinds (Entity, Fact, Episode, Reflection, Skill) stored as a single
MemoryNode dataclass with a ``kind`` discriminator.  Edges are typed structural
relationships (SUBJECT, OBJECT, ABOUT, …); belief scores live on nodes, not edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_NODE_KINDS = frozenset({"entity", "fact", "episode", "reflection", "skill"})

ALLOWED_EDGE_TYPES = frozenset({
    "SUBJECT", "OBJECT", "ABOUT", "SUPPORTED_BY",
    "CONTRADICTS", "REFINES", "INVOLVES", "MENTIONS",
    "PRODUCED", "TRIGGERED_BY", "NEXT", "DERIVED_FROM",
    "LEARNED_FROM", "HAS_SKILL", "RELATED_TO",
    "USES_TOOL", "APPLIES_TO",
})


# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRef:
    ref_type: str
    ref_id: str
    span_id: Optional[str] = None


@dataclass(frozen=False, eq=True)
class Scope:
    """Ownership / visibility context for a memory node."""
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    task_id: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.agent_id, self.user_id, self.task_id))


@dataclass
class TraceEvent:
    """Trace event; only event_type, actor, and content are required."""

    event_type: str
    actor: str
    content: str
    event_id: str = field(default_factory=lambda: uuid4().hex)
    tool_name: Optional[str] = None
    args_hash: Optional[str] = None
    result_hash: Optional[str] = None
    ts: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    task_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Knowledge-graph node
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    """Single node type for LTM.  ``kind`` discriminates Entity / Fact /
    Episode / Reflection / Skill; each kind uses a subset of the optional
    content fields below.
    """

    node_id: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    provenance: List[ProvenanceRef] = field(default_factory=list)
    embedding: Optional[Sequence[float]] = None
    kind: str = ""
    scopes: List[Scope] = field(default_factory=list)

    @property
    def scope(self) -> Scope:
        """Primary scope (first in list), or empty Scope."""
        return self.scopes[0] if self.scopes else Scope()

    @scope.setter
    def scope(self, value: Scope) -> None:
        """Backward-compat setter: replaces scopes with a single-element list."""
        self.scopes = [value]

    # --- Common: ordering (used for all kinds) ---
    sequence: Optional[int] = None  # global order for this agent (1-based)

    # --- Entity fields ---
    name: str = ""
    aliases: Sequence[str] = field(default_factory=list)

    # --- Fact fields ---
    fact_text: str = ""
    belief: float = 1.0
    polarity: bool = True
    time_valid_from: Optional[datetime] = None
    time_valid_to: Optional[datetime] = None

    # --- Episode fields ---
    summary: str = ""
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    outcome: str = ""

    # --- Reflection fields ---
    reflection_text: str = ""

    # --- Skill fields ---
    skill_description: str = ""
    utility: float = 0.0


# ---------------------------------------------------------------------------
# Knowledge-graph edge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    edge_id: str
    edge_type: str
    source_id: str
    target_id: str
    created_at: datetime
    label: str = ""
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Query / budget helpers
# ---------------------------------------------------------------------------

@dataclass
class QueryFilters:
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    workspace_id: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    tags: Dict[str, str] = field(default_factory=dict)
    skip_stm: bool = False


@dataclass
class RetrievalBudget:
    """Limits how many nodes are retrieved per node kind.

    LTM retrieval runs a node KNN search, then buckets nodes by kind
    (fact, reflection, skill, episode) and caps each bucket.
    """
    max_tokens: int = 1500
    max_items: int = 20
    max_facts: int = 8
    max_reflections: int = 4
    max_skills: int = 4
    max_episodes: int = 4


# ---------------------------------------------------------------------------
# Memory pack (retrieval result)
# ---------------------------------------------------------------------------

@dataclass
class MemoryPack:
    """Retrieved memory as node content strings per category."""
    facts: Sequence[str] = field(default_factory=list)
    reflections: Sequence[str] = field(default_factory=list)
    skills: Sequence[str] = field(default_factory=list)
    episodes: Sequence[str] = field(default_factory=list)
    citations: Sequence[ProvenanceRef] = field(default_factory=list)
    scores: Optional[Dict[str, Sequence[float]]] = None

    def to_text(self, include_citations: bool = True) -> str:
        lines: list[str] = []
        if self.facts:
            lines.append("## Facts")
            for t in self.facts:
                lines.append(f"- {t}")
            lines.append("")
        if self.reflections:
            lines.append("## Reflections")
            for t in self.reflections:
                lines.append(f"- {t}")
            lines.append("")
        if self.skills:
            lines.append("## Skills")
            for t in self.skills:
                lines.append(f"- {t}")
            lines.append("")
        if self.episodes:
            lines.append("## Episodes (in temporal order)")
            for t in self.episodes:
                lines.append(f"- {t}")
            lines.append("")
        if include_citations and self.citations:
            lines.append("## Citations")
            for citation in self.citations:
                citation_text = f"- {citation.ref_type}"
                if citation.ref_id:
                    citation_text += f": {citation.ref_id}"
                meta = getattr(citation, "metadata", None)
                if meta:
                    citation_text += f" ({', '.join(f'{k}={v}' for k, v in list(meta.items())[:2])})"
                lines.append(citation_text)
            lines.append("")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Other data types (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class ForgetReport:
    deleted_node_ids: Sequence[str] = field(default_factory=list)
    tombstone_ids: Sequence[str] = field(default_factory=list)
    rewired_edge_ids: Sequence[str] = field(default_factory=list)
    policy_applied: str = ""


@dataclass
class IntegrationBundle:
    prompt_pack: Optional[str] = None
    structured_state: Dict[str, Any] = field(default_factory=dict)
    tool_hints: Sequence[str] = field(default_factory=list)


@dataclass
class EvalScore:
    name: str
    value: float
    rationale: str = ""


@dataclass
class EvalReport:
    scores: Sequence[EvalScore] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    critiques: Sequence[str] = field(default_factory=list)


@dataclass
class PolicyDelta:
    updated_thresholds: Dict[str, float] = field(default_factory=dict)
    retrieval_weights: Dict[str, float] = field(default_factory=dict)
    integration_budgets: Dict[str, int] = field(default_factory=dict)
    notes: Sequence[str] = field(default_factory=list)


@dataclass
class STMContextItem:
    item_id: str
    content: str
    created_at: datetime
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class STMWorkingNote:
    key: str
    value: Any
    confidence: float
    provenance: Sequence[ProvenanceRef] = field(default_factory=list)
    pinned_until: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None


@dataclass
class STMEpisode:
    episode_id: str
    task_id: Optional[str]
    start_ts: datetime
    end_ts: datetime
    summary: str
    key_events: Sequence[str] = field(default_factory=list)
    outcomes: Sequence[str] = field(default_factory=list)
    unresolved_items: Sequence[str] = field(default_factory=list)
    embedding: Optional[Sequence[float]] = None
    belief: float = 0.0
    agent_id: Optional[str] = None


@dataclass
class BeliefSignal:
    item_id: str
    score: float
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class LTMDirectoryEntry:
    node_id: str
    node_class: str
    title: str
    summary: str
    entities: Sequence[str] = field(default_factory=list)
    time_range: Optional[Tuple[datetime, datetime]] = None
    embedding: Optional[Sequence[float]] = None
    agent_id: Optional[str] = None
