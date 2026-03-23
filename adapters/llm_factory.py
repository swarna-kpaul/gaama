"""Factory for LLM adapters from config. Supports multiple providers."""
from __future__ import annotations

from gaama.adapters.interfaces import LLMAdapter
from gaama.adapters.openai_llm import OpenAILLMAdapter
from gaama.config.settings import LLMSettings


def create_llm_adapter(settings: LLMSettings) -> LLMAdapter:
    """Create an LLM adapter from settings. Provider-agnostic config."""
    provider = (settings.provider or "openai").lower()
    if provider == "openai":
        return OpenAILLMAdapter(settings)
    raise ValueError(f"Unknown LLM provider: {settings.provider}. Supported: openai")
