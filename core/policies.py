from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional


@dataclass
class ForgettingPolicy:
    ttl_by_type: Dict[str, timedelta] = field(default_factory=dict)
    decay_half_life_days: Optional[int] = None
    access_keep_threshold: Optional[int] = None
    contradiction_suppresses: bool = True
    sensitivity_retention_days: Optional[int] = None
    budget_per_user: Optional[int] = None
    budget_per_task: Optional[int] = None


@dataclass
class SegmentationPolicy:
    max_events_per_episode: int = 50
    max_minutes_per_episode: int = 30
    allow_cross_task_merge: bool = False


@dataclass
class RetrievalPolicy:
    vector_weight: float = 0.6
    graph_weight: float = 0.3
    time_weight: float = 0.1
    graph_depth: int = 2
    rerank_top_k: int = 20


@dataclass
class ExtractionPolicy:
    min_confidence: float = 0.4
    min_novelty: float = 0.3
    include_sensitive: bool = False
    require_provenance: bool = True


@dataclass
class STMPolicy:
    context_max_items: int = 50
    context_max_tokens: int = 1200
    # Fixed rolling window size for STM working notes: at most this many notes; oldest (by updated_at) dropped when over.
    working_notes_max_items: int = 200
    episode_max_minutes: int = 30
    episode_max_events: int = 100
    belief_threshold: float = 0.6
    # Max messages in the single trace-event buffer (rolling window). Oldest dropped only after flushed and created.
    trace_buffer_max_events: int = 200


@dataclass
class ConsolidationPolicy:
    promote_on_session_end: bool = True
    promote_on_belief: bool = True
    promote_interval_minutes: int = 30
    require_validation: bool = True


