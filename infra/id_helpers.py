"""Stable hash-based ID generation for memory nodes."""
from __future__ import annotations

import hashlib
import re
from typing import Sequence


def normalize_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _hash_prefix(content: str, prefix: str, length: int = 16) -> str:
    if not content:
        content = ""
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{prefix}-{h[:length]}"


def canonical_id_entity(name: str, aliases: Sequence[str] | None = None) -> str:
    norm_name = normalize_text(name)
    if aliases:
        norm_aliases = "|".join(sorted(normalize_text(a) for a in aliases))
        content = f"entity|{norm_name}|{norm_aliases}"
    else:
        content = f"entity|{norm_name}" if norm_name else f"entity|{hash(name)}"
    return _hash_prefix(content, "entity")
