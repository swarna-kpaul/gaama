from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractionPolicy:
    min_confidence: float = 0.4
    min_novelty: float = 0.3
    include_sensitive: bool = False
    require_provenance: bool = True
