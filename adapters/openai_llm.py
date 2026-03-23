"""OpenAI-based LLM adapter. Uses openai package; config-driven (api key from env)."""
from __future__ import annotations

import os
from typing import Optional

from gaama.config.settings import LLMSettings


class OpenAILLMAdapter:
    """LLM adapter using OpenAI-compatible API (openai SDK)."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI adapter requires the openai package. Install with: pip install openai"
                ) from e
            api_key = os.environ.get(self._settings.api_key_env)
            if not api_key:
                raise ValueError(
                    f"LLM API key not set. Set environment variable: {self._settings.api_key_env}"
                )
            kwargs = {"api_key": api_key}
            if self._settings.base_url:
                kwargs["base_url"] = self._settings.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        client = self._get_client()
        completion_tokens = max_tokens or self._settings.max_tokens
        kwargs = dict(
            model=model or self._settings.model,
            messages=messages,
            max_completion_tokens=completion_tokens,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Newer OpenAI models require max_completion_tokens; older ones accept it too.
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()
