"""OpenAI-based embedding adapter. Uses openai package; config-driven (api key from env).

Includes an in-memory cache so that repeated embed() calls for the same text
(e.g. during canonicalization → upsert) hit the cache instead of the API.
Also provides embed_batch() for batching multiple texts in a single API call.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Sequence

from gaama.adapters.interfaces import EmbeddingAdapter
from gaama.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SIZE = 2048


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Embedding adapter using OpenAI embeddings API with LRU cache."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        self._settings = settings
        self._client = None
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI embedding adapter requires the openai package. Install with: pip install openai"
                ) from e
            api_key = os.environ.get(self._settings.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Embedding API key not set. Set environment variable: {self._settings.api_key_env}"
                )
            kwargs = {"api_key": api_key}
            if self._settings.base_url:
                kwargs["base_url"] = self._settings.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _cache_get(self, key: str) -> list[float] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: list[float]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def embed(self, text: str) -> Sequence[float]:
        text = text.strip()
        if not text:
            return []
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        client = self._get_client()
        response = client.embeddings.create(
            model=self._settings.model,
            input=text,
        )
        if not response.data:
            return []
        result = list(response.data[0].embedding)
        self._cache_put(text, result)
        return result

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts in one API call. Cached texts skip the API."""
        stripped = [t.strip() for t in texts]
        results: list[list[float] | None] = [None] * len(stripped)
        to_fetch: list[tuple[int, str]] = []  # (original_index, text)

        for i, t in enumerate(stripped):
            if not t:
                results[i] = []
                continue
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                to_fetch.append((i, t))

        if to_fetch:
            batch_texts = [t for _, t in to_fetch]
            client = self._get_client()
            response = client.embeddings.create(
                model=self._settings.model,
                input=batch_texts,
            )
            # OpenAI returns embeddings in same order as input
            for data_item in response.data:
                orig_idx, orig_text = to_fetch[data_item.index]
                emb = list(data_item.embedding)
                results[orig_idx] = emb
                self._cache_put(orig_text, emb)

        return [r if r is not None else [] for r in results]
