from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from gaama.adapters import (
    LocalBlobStore,
    NullEdgeVectorStore,
    SqliteMemoryStore,
    SqliteVectorStore,
    OpenAIEmbeddingAdapter,
    create_llm_adapter,
)
from gaama.config.settings import (
    EmbeddingSettings,
    SDKSettings,
    StorageSettings,
)
from gaama.core import (
    EvalReport,
    ForgetReport,
    IntegrationBundle,
    MemoryPack,
    QueryFilters,
    RetrievalBudget,
    TraceEvent,
)
from gaama.core.policies import ExtractionPolicy
from gaama.services.defaults import DefaultTraceNormalizer, SimpleEvaluator
from gaama.services.interfaces import CreateOptions, RetrieveOptions, RetrievalSource
from gaama.services.orchestrator import AgenticMemoryOrchestrator
from gaama.services.ltm_retriever import (
    NodeKNNPageRankRetrievalEngine,
    LTMForgettingEngine,
    LTMIntegrationComposer,
)
from gaama.services.llm_extractors import (
    NoOpMemoryExtractor,
)
from gaama.services.hybrid_search import HybridSearchConfig, HybridSearcher


class AgenticMemorySDK:
    """SDK scoped to one agent_id. All methods use this agent_id by default."""

    def __init__(self, orchestrator: AgenticMemoryOrchestrator, agent_id: str) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id is required for SDK creation.")
        self._orchestrator = orchestrator
        self._agent_id = agent_id.strip()

    def ingest(self, trace_events: Iterable[TraceEvent]) -> Sequence[str]:
        return self._orchestrator.ingest(trace_events, self._agent_id)

    def create(self, options: CreateOptions | None = None) -> Sequence[str]:
        opts = options or CreateOptions(agent_id=self._agent_id)
        if not opts.agent_id or not opts.agent_id.strip():
            opts.agent_id = self._agent_id
        return self._orchestrator.create(opts)

    def retrieve(
        self,
        query: str,
        filters: QueryFilters | RetrieveOptions | None = None,
        budget: RetrievalBudget | None = None,
        sources: RetrievalSource = "both",
        ppr_score_weight: float | None = None,
        sim_score_weight: float | None = None,
        degree_correction: bool | None = None,
        expansion_depth: int | None = None,
        edge_type_weights: dict[str, float] | None = None,
        semantic_only: bool = False,
        hybrid: bool = False,
        llm: object | None = None,
        llm_model: str | None = None,
        max_memory_words: int = 600,
        hybrid_fusion: str = "rrf",
        budgetless: bool = False,
    ) -> MemoryPack:
        # Allow passing a single RetrieveOptions as second argument: sdk.retrieve(query, options)
        if isinstance(filters, RetrieveOptions):
            return self._orchestrator.retrieve(query, filters)
        if filters is None:
            filters = QueryFilters(agent_id=self._agent_id)
        elif not filters.agent_id or not filters.agent_id.strip():
            filters.agent_id = self._agent_id
        if budget is None:
            budget = RetrievalBudget()
        options = RetrieveOptions(
            filters=filters,
            budget=budget,
            sources=sources,
            ppr_score_weight=ppr_score_weight,
            sim_score_weight=sim_score_weight,
            degree_correction=degree_correction,
            expansion_depth=expansion_depth,
            edge_type_weights=edge_type_weights,
            semantic_only=semantic_only,
            hybrid=hybrid,
            llm=llm,
            llm_model=llm_model,
            max_memory_words=max_memory_words,
            hybrid_fusion=hybrid_fusion,
            budgetless=budgetless,
        )
        return self._orchestrator.retrieve(query, options)

    def forget(self, selector: QueryFilters | None = None) -> ForgetReport:
        if selector is None:
            selector = QueryFilters(agent_id=self._agent_id)
        elif not selector.agent_id or not selector.agent_id.strip():
            selector.agent_id = self._agent_id
        return self._orchestrator.forget(selector)

    def clear_ltm(self) -> None:
        self._orchestrator.clear_ltm()

    def clear_trace_buffer(self) -> None:
        self._orchestrator.clear_trace_buffer(self._agent_id)

    def integrate(self, query: str, memory_pack: MemoryPack, mode: str = "prompt") -> IntegrationBundle:
        return self._orchestrator.integrate(query, memory_pack, mode, self._agent_id)

    def evaluate(self, dataset_id: str) -> EvalReport:
        return self._orchestrator.evaluate(dataset_id)

    def flush_stm_episode(self) -> None:
        """No-op stub: GAAMA does not include STM."""
        pass


def create_default_sdk(settings: SDKSettings, agent_id: str) -> AgenticMemorySDK:
    """Create SDK scoped to one agent_id."""
    if not agent_id or not agent_id.strip():
        raise ValueError("agent_id is required for SDK creation.")
    agent_id = agent_id.strip()
    memory_store = SqliteMemoryStore(settings.storage.sqlite_path)
    embedder = None
    if settings.embedding:
        embedder = OpenAIEmbeddingAdapter(settings.embedding)
    vector_store = SqliteVectorStore(
        path=settings.storage.sqlite_path,
        embedder=embedder,
        node_store=memory_store,
        dimension=1536,
    )
    blob_store = LocalBlobStore(settings.storage.blob_root)

    llm_adapter = None
    if settings.llm:
        llm_adapter = create_llm_adapter(settings.llm)

    normalizer = DefaultTraceNormalizer()
    # Extractor is NoOp; LTMCreator (inside orchestrator) handles fact/reflection extraction via LLM
    extractor = NoOpMemoryExtractor()
    if llm_adapter is not None:
        # Store llm on the extractor so the orchestrator can pass it to LTMCreator
        extractor._llm = llm_adapter

    hybrid_searcher = HybridSearcher(
        memory_store,
        vector_store,
        embedder,
        config=HybridSearchConfig(),
    )
    ltm_retriever = NodeKNNPageRankRetrievalEngine(
        memory_store, memory_store, vector_store
    )
    forgetter = LTMForgettingEngine(memory_store)
    integrator = LTMIntegrationComposer()
    evaluator = SimpleEvaluator()
    orchestrator = AgenticMemoryOrchestrator(
        normalizer=normalizer,
        extractor=extractor,
        ltm_retriever=ltm_retriever,
        forgetter=forgetter,
        integrator=integrator,
        evaluator=evaluator,
        node_store=memory_store,
        graph_store=memory_store,
        vector_store=vector_store,
        blob_store=blob_store,
        extraction_policy=ExtractionPolicy(),
        embedder=embedder,
        budget_llm_adapter=llm_adapter,
        budget_llm_model_name=settings.llm.model if settings.llm else None,
        budget_llm_max_tokens=512,
    )
    sdk = AgenticMemorySDK(orchestrator, agent_id)
    return sdk


def default_settings(root: Path) -> SDKSettings:
    return SDKSettings(
        storage=StorageSettings(
            sqlite_path=root / "gaama.sqlite",
            blob_root=root / "blobs",
        ),
    )
