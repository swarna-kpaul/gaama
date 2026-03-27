from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StorageSettings:
    sqlite_path: Path
    blob_root: Path


@dataclass
class LLMSettings:
    """Configuration for LLM adapter. Provider-agnostic; adapter interprets fields."""
    provider: str = "openai"
    model: str = "gpt-5.1"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    max_tokens: int = 2048


@dataclass
class EmbeddingSettings:
    """Configuration for embedding adapter (e.g. OpenAI text-embedding)."""
    model: str = "text-embedding-3-large"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None


@dataclass
class SDKSettings:
    storage: StorageSettings
    llm: LLMSettings | None = None
    embedding: EmbeddingSettings | None = None

