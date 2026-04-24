"""Canonical types for the Agentic Memory SDK.

Five node kinds (Entity, Fact, Episode, Reflection, Skill) stored as a single
MemoryNode dataclass with a ``kind`` discriminator.  Edges are typed structural
relationships (NEXT, DERIVED_FROM, HAS_CONCEPT, …); belief scores live on nodes,
not edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_NODE_KINDS = frozenset({"entity", "fact", "episode", "reflection", "skill", "concept"})

ALLOWED_EDGE_TYPES = frozenset({
    "NEXT", "DERIVED_FROM", "DERIVED_FROM_FACT",
    "HAS_CONCEPT", "ABOUT_CONCEPT",
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

    # --- Concept fields ---
    concept_label: str = ""

    # --- Skill fields ---
    skill_description: str = ""
    utility: float = 0.0

    # --- Common scoring ---
    relevance_score: float = 1.0


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

    def trim_by_words(self, max_words: int, budget: Optional["RetrievalBudget"] = None) -> "MemoryPack":
        """Return a new MemoryPack trimmed so that to_text() stays under *max_words*.

        Removal is distributed **proportionally** across categories according to
        their budget caps so that no single category is drained before others.

        For facts, reflections, and skills the last item is the lowest-ranked
        (LTM fills buckets in descending score order).  For episodes -- which are
        reordered chronologically -- the item with the **lowest relevance score**
        is removed instead, preserving temporal order of the survivors.

        Scores are read from ``self.scores`` (populated by LTM retrieval).
        If scores are unavailable the function falls back to removing the last item.
        If *budget* is None the four categories are treated as equal weight.
        """
        categories = ["facts", "reflections", "skills", "episodes"]

        # Mutable copy of item lists
        items: Dict[str, List[str]] = {
            cat: list(getattr(self, cat)) for cat in categories
        }

        def _word_count() -> int:
            return sum(len(s.split()) for cat in categories for s in items[cat])

        if _word_count() <= max_words:
            return self

        # Budget weights (proportional share per category)
        budget_map = {
            "facts": budget.max_facts if budget else 1,
            "reflections": budget.max_reflections if budget else 1,
            "skills": budget.max_skills if budget else 1,
            "episodes": budget.max_episodes if budget else 1,
        }
        total_budget = sum(budget_map.values()) or 1

        # Parallel score lists (used for score-aware removal of episodes)
        score_lists: Dict[str, List[float]] = {}
        for cat in categories:
            if self.scores and cat in self.scores:
                score_lists[cat] = list(self.scores[cat])
            else:
                score_lists[cat] = []

        # Compute per-item word counts (cache for efficiency)
        word_counts: Dict[str, List[int]] = {
            cat: [len(s.split()) for s in items[cat]] for cat in categories
        }

        # Iteratively remove items until under budget.
        # Each round, pick the category that is most over-represented relative
        # to its budget share and remove its lowest-ranked item.
        while _word_count() > max_words:
            # Find which categories still have items
            non_empty = [cat for cat in categories if items[cat]]
            if not non_empty:
                break

            # For each non-empty category compute how over-represented it is:
            #   actual_share / target_share  (higher = more over-represented)
            total_words = _word_count() or 1
            worst_cat = None
            worst_ratio = -1.0
            for cat in non_empty:
                cat_words = sum(word_counts[cat])
                target_share = budget_map[cat] / total_budget
                actual_share = cat_words / total_words
                ratio = actual_share / target_share if target_share > 0 else float("inf")
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_cat = cat

            if worst_cat is None:
                break

            # For episodes: remove the item with the lowest relevance score
            # (preserves chronological order of survivors).
            # For other categories: remove the last item (already lowest-ranked).
            if worst_cat == "episodes" and score_lists.get("episodes"):
                min_idx = min(
                    range(len(score_lists["episodes"])),
                    key=lambda i: score_lists["episodes"][i],
                )
                items["episodes"].pop(min_idx)
                word_counts["episodes"].pop(min_idx)
                score_lists["episodes"].pop(min_idx)
            else:
                items[worst_cat].pop()
                word_counts[worst_cat].pop()
                if score_lists.get(worst_cat):
                    score_lists[worst_cat].pop()

        return MemoryPack(
            facts=items["facts"],
            reflections=items["reflections"],
            skills=items["skills"],
            episodes=items["episodes"],
            citations=list(self.citations),
            scores={cat: score_lists[cat] for cat in categories} if any(score_lists.values()) else None,
        )


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


# ---------------------------------------------------------------------------
# GEL (Graph Edit Learning) types
# ---------------------------------------------------------------------------

@dataclass
class GELConfig:
    """Configuration for the Graph Edit Learning system."""
    enabled: bool = False
    reward_threshold: float = 0.7
    max_analysis_questions: int = 3
    max_edits_per_query: int = 4
    verify_after_edit: bool = False
    batch_only: bool = True
    gel_fact_belief: float = 0.85
    max_facts_per_query: int = 2
    max_concepts_per_query: int = 2
    dedup_similarity_threshold: float = 0.90
    verify_edits_before_insert: bool = True


@dataclass
class SubQuestionResult:
    """Result of graph exploration for a single analysis question."""
    question: str
    reasoning: str
    retrieved_pack: MemoryPack
    scored_items: List[Tuple[str, str, float]]
    subgraph_nodes: List[MemoryNode] = field(default_factory=list)
    subgraph_edges: List[Edge] = field(default_factory=list)


@dataclass
class GELEditOp:
    """A single graph edit operation proposed by GEL."""
    op_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    root_cause: str = ""


@dataclass
class GELReport:
    """Report from a single GEL invocation."""
    query: str
    reward_before: float
    reward_after: float = 0.0
    analysis_questions_generated: int = 0
    chain_of_thought: str = ""
    edits_planned: int = 0
    edits_executed: int = 0
    edits_skipped: int = 0
    edit_ops: List[GELEditOp] = field(default_factory=list)
