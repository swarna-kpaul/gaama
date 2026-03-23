"""Generate a concise answer to a query using retrieved memory and an LLM."""
from __future__ import annotations

from typing import TYPE_CHECKING

from gaama.core import MemoryPack
from gaama.infra.prompt_loader import load_prompt

if TYPE_CHECKING:
    from gaama.adapters.interfaces import LLMAdapter


def answer_from_memory(
    query: str,
    memory_pack: MemoryPack,
    llm: "LLMAdapter",
    *,
    include_citations: bool = False,
    max_tokens: int = 512,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Generate a concise answer to the query using the memory pack and the given LLM.

    Args:
        query: The user or agent question to answer.
        memory_pack: Retrieved memory (facts, reflections, skills, episodes).
        llm: LLM adapter used to generate the answer.
        include_citations: Whether to include citations in the memory text passed to the LLM.
        max_tokens: Maximum tokens for the LLM response.
        model: Optional model name override for this call.

    Returns:
        A concise answer string. Empty if memory_pack is empty or LLM returns nothing.
    """
    memory_text = memory_pack.to_text(include_citations=include_citations).strip()
    if not memory_text:
        return ""
    prompt = load_prompt(
        "answer_from_memory",
        {"query": query, "memory_text": memory_text},
    )
    raw = llm.complete(prompt, max_tokens=max_tokens, model=model, temperature=temperature)
    return (raw or "").strip()
