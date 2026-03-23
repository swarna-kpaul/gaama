from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from typing import Any, Dict

from gaama.core import Edge, MemoryNode, ProvenanceRef, Scope

_MEMORY_NODE_FIELDS = {f.name for f in fields(MemoryNode)}


def node_to_embed_text(node: MemoryNode) -> str:
    """Return the text used for embedding, dispatched by ``node.kind``."""
    kind = (getattr(node, "kind", "") or "").strip().lower()
    if kind == "fact":
        t = (getattr(node, "fact_text", "") or "").strip()
        if t:
            return t
    elif kind == "episode":
        t = (getattr(node, "summary", "") or "").strip()
        if t:
            return t
    elif kind == "reflection":
        t = (getattr(node, "reflection_text", "") or "").strip()
        if t:
            return t
    elif kind == "skill":
        t = (getattr(node, "skill_description", "") or "").strip()
        if t:
            return t
    # entity or fallback: use name / first alias
    name = (getattr(node, "name", "") or "").strip()
    if name:
        return name
    aliases = getattr(node, "aliases", None) or ()
    if aliases:
        first = (str(aliases[0]) or "").strip()
        if first:
            return first
    return ""


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_dataclass(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def serialize_dataclass(instance: Any) -> Dict[str, Any]:
    data = asdict(instance)
    return _serialize_value(data)


def serialize_node(node: MemoryNode) -> str:
    payload = serialize_dataclass(node)
    payload["_node_class"] = "MemoryNode"
    if not payload.get("kind") and getattr(node, "kind", ""):
        payload["kind"] = getattr(node, "kind", "")
    # Ensure scopes is serialized (asdict handles it), remove legacy scope key
    payload.pop("scope", None)
    return json.dumps(payload)


def deserialize_node(payload: str) -> MemoryNode:
    data = json.loads(payload)
    raw = data.pop("_node_class", None)
    if not raw:
        legacy_class_key = "_" + "node" + "_type"
        raw = data.pop(legacy_class_key, None)
    legacy_map: Dict[str, str] = {
        "Episode": "episode",
        "Event": "event",
        "Entity": "entity",
        "Fact": "fact",
        "Reflection": "reflection",
        "Skill": "skill",
        "Task": "task",
        "PolicyHint": "policy_hint",
        "episode": "episode",
        "event": "event",
        "entity": "entity",
        "fact": "fact",
        "reflection": "reflection",
        "skill": "skill",
        "task": "task",
        "policy_hint": "policy_hint",
        "MemoryNode": "MemoryNode",
    }
    if raw and str(raw).strip():
        r = str(raw).strip()
        kind = legacy_map.get(r, legacy_map.get(r.lower(), None))
        if kind is not None:
            if kind != "MemoryNode":
                data["kind"] = kind
        else:
            raise ValueError(f"Unknown node class: {raw}")
    data.pop("node_" + "type", None)

    # Migrate legacy top-level agent_id/user_id/task_id and single scope into scopes list
    scopes_raw = data.pop("scopes", None)
    scope_raw_single = data.pop("scope", None)

    legacy_scope: dict = {}
    for legacy_key in ("agent_id", "user_id", "task_id"):
        if legacy_key in data:
            legacy_scope[legacy_key] = data.pop(legacy_key)

    if isinstance(scopes_raw, list) and scopes_raw:
        data["scopes"] = scopes_raw
    elif isinstance(scope_raw_single, dict):
        merged = {**scope_raw_single, **{k: v for k, v in legacy_scope.items() if k not in scope_raw_single}}
        data["scopes"] = [merged]
    elif legacy_scope:
        data["scopes"] = [legacy_scope]
    else:
        data["scopes"] = []

    # Drop workspace_id (removed field)
    data.pop("workspace_id", None)

    coerced = _coerce_dates(data)

    # Reconstruct Scope objects from dicts in scopes list
    raw_scopes = coerced.get("scopes")
    if isinstance(raw_scopes, list):
        coerced["scopes"] = [
            Scope(
                agent_id=s.get("agent_id") if isinstance(s, dict) else None,
                user_id=s.get("user_id") if isinstance(s, dict) else None,
                task_id=s.get("task_id") if isinstance(s, dict) else None,
            )
            for s in raw_scopes
        ]

    # Map legacy fields into new schema
    if "confidence" in coerced and "belief" not in coerced:
        coerced["belief"] = coerced.pop("confidence")
    else:
        coerced.pop("confidence", None)
    if "outcomes" in coerced:
        outcomes = coerced.pop("outcomes")
        if outcomes and not coerced.get("outcome"):
            coerced["outcome"] = outcomes[0] if isinstance(outcomes, list) else str(outcomes)

    kwargs = {k: v for k, v in coerced.items() if k in _MEMORY_NODE_FIELDS}
    return MemoryNode(**kwargs)


def _coerce_dates(data: Dict[str, Any]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = {}
    _date_suffixes = ("_at", "_from", "_to")
    _date_keys = ("ts", "start_ts", "end_ts")
    for key, value in data.items():
        if isinstance(value, str) and _looks_like_iso(value) and (
            any(key.endswith(s) for s in _date_suffixes) or key in _date_keys
        ):
            coerced[key] = datetime.fromisoformat(value)
        elif key == "provenance" and isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, dict):
                    out.append(ProvenanceRef(
                        ref_type=str(item.get("ref_type", "")),
                        ref_id=str(item.get("ref_id", "")),
                        span_id=item.get("span_id"),
                    ))
                else:
                    out.append(item)
            coerced[key] = out
        elif isinstance(value, dict):
            coerced[key] = _coerce_dates(value)
        elif isinstance(value, list):
            coerced[key] = [_coerce_list_item(item) for item in value]
        else:
            coerced[key] = value
    return coerced


def _coerce_list_item(item: Any) -> Any:
    if isinstance(item, dict):
        return _coerce_dates(item)
    return item


def _looks_like_iso(value: str) -> bool:
    return "T" in value and ":" in value
