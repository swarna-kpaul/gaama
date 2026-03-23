from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Protocol, Sequence

from gaama.core import (
    EvalReport,
    MemoryNode,
    MemoryPack,
    PolicyDelta,
    QueryFilters,
    RetrievalBudget,
    TraceEvent,
)


@dataclass(frozen=True)
class EdgeSpec:
    """Index-based edge from extractor; indices refer to the combined node list.

    ``edge_type`` is one of the structural relationship types (SUBJECT,
    OBJECT, ABOUT, INVOLVES, …).  ``label`` is an optional freeform
    description.
    """
    source_index: int
    target_index: int
    edge_type: str = ""
    label: str = ""


@dataclass
class ExtractResult:
    """Result of memory extraction: nodes and optional index-based edge specs for the same batch."""
    nodes: Sequence[MemoryNode]
    edge_specs: Sequence[EdgeSpec] = ()


@dataclass
class EpisodeSummaryResult:
    summary: str
    outcomes: Sequence[str]
    unresolved_items: Sequence[str]


@dataclass
class STMNoteItem:
    key: str
    value: object
    confidence: float


class EpisodeSummarizer(Protocol):
    def summarize(self, events: Sequence[TraceEvent]) -> EpisodeSummaryResult:
        ...


class STMNoteExtractor(Protocol):
    def extract_notes(self, events: Sequence[TraceEvent]) -> Sequence[STMNoteItem]:
        ...


@dataclass
class CreateOptions:
    agent_id: str | None = None
    episode_id: str | None = None
    allow_reflection: bool = True
    allow_skills: bool = True
    user_id: str | None = None
    task_id: str | None = None
    chunk_size: int = 0
    """If > 0, process events in chunks of this size; 0 = process all events in one go. Ignored when max_tokens_per_chunk is set."""
    chunk_overlap: int = 0
    """Number of events to overlap between consecutive chunks (0 <= chunk_overlap < chunk_size). Ignored if chunk_size <= 0 or max_tokens_per_chunk is set."""
    max_tokens_per_chunk: Optional[int] = None
    """If set, events are divided into chunks by token count: fill buckets until token count reaches this limit, then start next chunk. Takes precedence over chunk_size. No overlap."""


RetrievalSource = Literal["stm", "ltm", "both"]


@dataclass
class RetrieveOptions:
    filters: QueryFilters
    budget: RetrievalBudget
    sources: RetrievalSource = "both"
    ppr_score_weight: float | None = None
    sim_score_weight: float | None = None
    degree_correction: bool | None = None
    expansion_depth: int | None = None
    edge_type_weights: dict[str, float] | None = None
    """Override edge-type weights for PPR transitions. None = use defaults."""
    adaptive_ppr_model: object | None = None
    """AdaptivePPRModel instance for query-adaptive PPR weighting. None = use fixed ppr_score_weight."""
    use_llm_budget: bool = False
    """If True and orchestrator has budget_llm_adapter + budget_llm_model_name, derive LTM budget from query via LLM."""
    budget_llm_model_name: str | None = None
    """Model name to use for budget derivation. If set, overrides orchestrator's budget_llm_model_name for this call."""
    max_memory_items: int | None = None
    """When use_llm_budget is True: max total items (facts + reflections + skills + episodes) to retrieve. The LLM distributes this across categories. If None, uses budget.max_items."""


@dataclass
class UpdateOptions:
    reason: str


class TraceNormalizer(Protocol):
    def normalize(self, raw_events: Iterable[TraceEvent]) -> Sequence[TraceEvent]:
        ...


class MemoryExtractor(Protocol):
    def extract(self, events: Sequence[TraceEvent]) -> ExtractResult:
        ...


class RetrievalEngine(Protocol):
    def retrieve(
        self, query: str, options: RetrieveOptions
    ) -> tuple[MemoryPack, Sequence[tuple[str, str, float]]]:
        """Returns (MemoryPack, list of (node_id, content, relevance_score))."""
        ...


class ForgettingEngine(Protocol):
    def forget(self, selector: QueryFilters) -> Sequence[str]:
        ...


class IntegrationComposer(Protocol):
    def compose(self, query: str, memory_pack: MemoryPack, mode: str) -> dict:
        ...


class Evaluator(Protocol):
    def evaluate(self, dataset_id: str) -> EvalReport:
        ...

    def improve(self, report: EvalReport) -> PolicyDelta:
        ...
