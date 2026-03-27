"""
Create LTM from LoCoMo conversation sessions.
================================================
Processes each sample: parse conversation sessions, ingest events, create LTM.
Supports resume via per-sample progress tracking.

Usage:
    python run_create_ltm.py                          # all samples
    python run_create_ltm.py --sample-ids conv-26     # specific sample
    python run_create_ltm.py --limit 3                # first 3 samples
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# SDK path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent.parent.parent  # gaama/evals/locomo -> gaama/evals -> gaama -> agentic-memory
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gaama.api import AgenticMemorySDK, create_default_sdk  # noqa: E402
from gaama.config.settings import (  # noqa: E402
    SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings,
)
from gaama.core import TraceEvent  # noqa: E402
from gaama.services.interfaces import CreateOptions  # noqa: E402

sys.path.insert(0, str(_FILE_DIR))
from config import (  # noqa: E402
    SAMPLE_IDS, LLM_MODEL, EMBEDDING_MODEL, MAX_TOKENS_PER_CHUNK,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_ID = "agent-locomo"
USER_ID = "locomo"
DATA_PATH = _FILE_DIR / "locomo10.json"
DATA_DIR = _FILE_DIR / "data" / "ltm"
DB_NAME = "locomo_memory.sqlite"
PROGRESS_FILE = _FILE_DIR / "data" / "ltm" / "create_ltm_progress.json"


# ---------------------------------------------------------------------------
# SDK factory
# ---------------------------------------------------------------------------

def _make_sdk(
    data_dir: Path,
    llm_model: str = LLM_MODEL,
    db_name: str = DB_NAME,
) -> AgenticMemorySDK:
    """Create a GAAMA SDK instance."""
    settings = SDKSettings(
        storage=StorageSettings(
            sqlite_path=data_dir / db_name,
            blob_root=data_dir / "blobs",
        ),
        llm=LLMSettings(
            provider="openai",
            model=llm_model,
            api_key_env="OPENAI_API_KEY",
            max_tokens=16000,
        ),
        embedding=EmbeddingSettings(
            model=EMBEDDING_MODEL,
            api_key_env="OPENAI_API_KEY",
        ),
    )
    return create_default_sdk(settings, agent_id=AGENT_ID)


# ---------------------------------------------------------------------------
# Session parsing (ported from locomo_pipeline.py)
# ---------------------------------------------------------------------------

def _parse_session_date(date_str: str) -> Optional[datetime]:
    """Best-effort parse of LoCoMo date strings like '1:56 pm on 8 May, 2023'."""
    if not date_str:
        return None
    s = date_str.strip().replace(",", "").replace(".", "")
    patterns = [
        r"(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        r"(\d{1,2}:\d{2}\s*[ap]m)\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            time_s, day, month_s, year = m.groups()
            try:
                return datetime.strptime(f"{day} {month_s} {year} {time_s}", "%d %B %Y %I:%M %p")
            except ValueError:
                pass
    return None


def _get_sorted_sessions(conversation: dict) -> List[Tuple[int, list, str]]:
    """Return list of (session_idx, turns, date_string) sorted by session number."""
    sessions = []
    for key in conversation:
        m = re.match(r"^session_(\d+)$", key)
        if not m:
            continue
        idx = int(m.group(1))
        turns = conversation[key]
        date_key = f"session_{idx}_date_time"
        date_str = conversation.get(date_key) or conversation.get(f"session_{idx}_date") or ""
        sessions.append((idx, turns, date_str))
    sessions.sort(key=lambda x: x[0])
    return sessions


def _session_to_events(
    turns: list,
    session_idx: int,
    session_date: Optional[str],
    base_ts: datetime,
) -> List[TraceEvent]:
    """Convert one LoCoMo session (list of speaker/text dicts) to TraceEvents."""
    events: List[TraceEvent] = []
    for turn_idx, turn in enumerate(turns):
        speaker = turn.get("speaker", "user")
        text = turn.get("text", "")
        if not text:
            continue
        ts = base_ts + timedelta(seconds=len(events) * 30)
        metadata: Dict[str, Any] = {
            "session_date": session_date or "",
        }
        blip_caption = turn.get("blip_caption")
        if blip_caption:
            metadata["blip_caption"] = blip_caption
        img_url = turn.get("img_url")
        if img_url:
            metadata["img_url"] = img_url
        query = turn.get("query")
        if query:
            metadata["query"] = query
        events.append(
            TraceEvent(
                event_id=f"s{session_idx}_t{turn_idx}_{len(events)}",
                event_type="message",
                actor=speaker,
                content=f"[{session_date}] {speaker}: {text}" if session_date else f"{speaker}: {text}",
                ts=ts,
                metadata=metadata,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress(path: Path) -> Dict[str, Any]:
    """Load progress state from a JSON file."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_progress(path: Path, state: Dict[str, Any]) -> None:
    """Atomically save progress state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(data_path: Path) -> List[dict]:
    """Load locomo10.json."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return data


# ---------------------------------------------------------------------------
# Main LTM creation
# ---------------------------------------------------------------------------

def create_ltm(
    data_path: Path = DATA_PATH,
    data_dir: Path = DATA_DIR,
    sample_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    llm_model: str = LLM_MODEL,
) -> None:
    """Build LTM from conversation sessions for each sample_id.

    Each sample_id is treated as a separate task_id. Existing LTM is not
    cleared before loading (additive). Supports resume via progress file.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(data_path)

    if sample_ids is not None:
        sid_set = set(sample_ids)
        dataset = [item for item in dataset if item.get("sample_id") in sid_set]
    if limit is not None:
        dataset = dataset[:limit]

    fallback_base = datetime(2023, 5, 1, 12, 0, 0)

    # Progress tracking
    progress = _load_progress(PROGRESS_FILE)
    completed: Set[str] = set(progress.get("completed_ids", []))
    if completed:
        print(f"Resuming: {len(completed)} sample(s) already completed, skipping them.", flush=True)

    total = len(dataset)

    for idx, item in enumerate(dataset):
        sid = item.get("sample_id", f"conv-{idx}")

        if sid in completed:
            print(f"[{idx + 1}/{total}] SKIP {sid} (already done)", flush=True)
            continue

        print(f"\n[{idx + 1}/{total}] Processing {sid}...", flush=True)
        t0 = time.time()

        sdk = _make_sdk(data_dir, llm_model=llm_model)

        conversation = item.get("conversation", {})
        sessions = _get_sorted_sessions(conversation)

        create_opts = CreateOptions(
            user_id=USER_ID,
            task_id=sid,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )

        total_events = 0
        for sess_idx, turns, date_str in sessions:
            parsed_dt = _parse_session_date(date_str)
            base_ts = parsed_dt if parsed_dt else fallback_base + timedelta(hours=sess_idx)
            events = _session_to_events(turns, sess_idx, date_str, base_ts)
            if not events:
                continue
            sdk.ingest(events)
            total_events += len(events)

        sdk.create(create_opts)

        elapsed = time.time() - t0
        print(
            f"[{idx + 1}/{total}] Created LTM for {sid}: "
            f"{len(sessions)} sessions, {total_events} events, "
            f"{len(item.get('qa', []))} questions ({elapsed:.1f}s)",
            flush=True,
        )

        # Update progress
        completed.add(sid)
        progress["completed_ids"] = sorted(completed)
        progress["last_updated"] = datetime.now().isoformat()
        _save_progress(PROGRESS_FILE, progress)

    print(f"\nDone. Processed {total} samples into LTM at {data_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create LTM from LoCoMo conversation sessions.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Specific sample IDs to process (e.g. conv-26 conv-30). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples.",
    )
    parser.add_argument(
        "--max-tokens-per-chunk",
        type=int,
        default=MAX_TOKENS_PER_CHUNK,
        help=f"Token limit per chunk for LTM creation (default: {MAX_TOKENS_PER_CHUNK}).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model for extraction (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Override data directory (default: {DATA_DIR}).",
    )
    args = parser.parse_args()

    sample_ids = args.sample_ids if args.sample_ids else SAMPLE_IDS
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    print("=" * 70, flush=True)
    print("LoCoMo LTM Creation", flush=True)
    print(f"Samples: {sample_ids}", flush=True)
    print(f"Data dir: {data_dir}", flush=True)
    print(f"max_tokens_per_chunk: {args.max_tokens_per_chunk}", flush=True)
    print(f"LLM model: {args.llm_model}", flush=True)
    print("=" * 70, flush=True)

    create_ltm(
        data_path=DATA_PATH,
        data_dir=data_dir,
        sample_ids=sample_ids,
        limit=args.limit,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
