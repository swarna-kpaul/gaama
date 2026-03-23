from __future__ import annotations

from typing import Iterable, Sequence

from gaama.core import (
    EvalReport,
    PolicyDelta,
    TraceEvent,
)
from gaama.services.interfaces import (
    Evaluator,
    TraceNormalizer,
)


class DefaultTraceNormalizer(TraceNormalizer):
    def normalize(self, raw_events: Iterable[TraceEvent]) -> Sequence[TraceEvent]:
        return list(raw_events)


class SimpleEvaluator(Evaluator):
    """Simple evaluator stub: returns empty evaluation report."""

    def evaluate(self, dataset_id: str) -> EvalReport:
        from gaama.core import EvalScore
        return EvalReport(scores=[EvalScore(name="stub", value=0.0, rationale="No grader configured.")])

    def improve(self, report: EvalReport) -> PolicyDelta:
        return PolicyDelta(notes=["No policy updates without graders."])
