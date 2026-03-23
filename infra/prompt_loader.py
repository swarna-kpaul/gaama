"""Load prompt templates from the prompt library (markdown files)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

_PROMPTS_DIR: Path | None = None


def _get_prompts_dir() -> Path:
    global _PROMPTS_DIR
    if _PROMPTS_DIR is None:
        _PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
    return _PROMPTS_DIR


def load_prompt(name: str, variables: Dict[str, str] | None = None) -> str:
    """
    Load a prompt template by name (without .md) from the prompts folder.
    Substitute {{key}} with variables[key]. Variables are optional.
    """
    path = _get_prompts_dir() / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {name} ({path})")
    text = path.read_text(encoding="utf-8")
    if not variables:
        return text
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", value)
    # Leave any unreplaced placeholders as-is (or strip them if preferred)
    return text
