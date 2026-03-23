# GAAMA: Graph Augmented Associative Memory for Agents

GAAMA is a long-term memory system for AI agents that combines a structured knowledge graph with neural-adaptive Personalized PageRank (PPR) retrieval. It extracts entities, facts, episodes, reflections, and skills from agent conversations, stores them in a knowledge graph with typed edges, and retrieves relevant memories using a hybrid BM25 + semantic vector search fused with graph-based PPR scoring.

## Architecture Overview

GAAMA's retrieval pipeline works in three stages:

1. **Knowledge Graph Construction**: Conversations are processed by an LLM extractor that produces typed memory nodes (entities, facts, episodes, reflections, skills) and structural edges (SUBJECT, OBJECT, INVOLVES, MENTIONS, SUPPORTED_BY, NEXT, etc.). Nodes are deduplicated via hybrid BM25+semantic canonicalization.

2. **Hybrid Retrieval with PPR**: At query time, semantic KNN finds seed nodes, which are expanded via graph edges. Personalized PageRank propagates relevance through the graph using edge-type-aware transition weights with hub dampening. The final score combines PPR and similarity: `score = w_ppr * PPR(node) + w_sim * sim(node, query)`.

3. **Neural Adaptive Weighting**: A lightweight MLP (gap regression) predicts per-query whether graph-based PPR helps or hurts retrieval quality. It learns from `(query_embedding, reward_gap)` pairs where `reward_gap = reward(ppr=1) - reward(ppr=0)`. At inference, queries where PPR helps get `ppr_weight=1.0`; others fall back to pure similarity with `ppr_weight=0.01`.

### Storage

- **SQLite** for nodes, edges, and FTS5 full-text index (BM25)
- **sqlite-vec** for vector embeddings (cosine similarity KNN)
- **Local blob store** for raw trace archives

## Installation

```bash
pip install openai sqlite-vec

# For neural PPR training
pip install torch

# For ROUGE-based evaluation
pip install rouge
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Quick Start

```python
from pathlib import Path
from gaama.api.client import create_default_sdk
from gaama.core import TraceEvent, RetrievalBudget, QueryFilters
from gaama.config.settings import SDKSettings, StorageSettings, LLMSettings, EmbeddingSettings
from gaama.services.interfaces import CreateOptions

# Configure
settings = SDKSettings(
    storage=StorageSettings(
        sqlite_path=Path("./data/memory.sqlite"),
        blob_root=Path("./data/blobs"),
    ),
    llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
    embedding=EmbeddingSettings(model="text-embedding-3-small"),
)

# Create SDK instance
sdk = create_default_sdk(settings, agent_id="my-agent")

# Ingest conversation events and extract long-term memories
events = [
    TraceEvent(event_type="message", actor="user",
               content="I visited the Eiffel Tower in Paris last week."),
    TraceEvent(event_type="message", actor="assistant",
               content="That sounds wonderful! How was the view from the top?"),
]
sdk._orchestrator.ingest(events, agent_id="my-agent")
node_ids = sdk.create(CreateOptions(
    agent_id="my-agent", user_id="user-1", task_id="conv-1",
))
print(f"Created {len(node_ids)} memory nodes")

# Retrieve relevant memories
pack = sdk.retrieve(
    query="What did the user do in Paris?",
    filters=QueryFilters(agent_id="my-agent", user_id="user-1", task_id="conv-1"),
    budget=RetrievalBudget(max_facts=5, max_episodes=10),
    ppr_score_weight=1.0,
    sim_score_weight=1.0,
)
print(pack.to_text())
```

## Running the LoCoMo Evaluation

The LoCoMo benchmark evaluates long-term memory over multi-session conversations (10 conversations, ~1540 questions across 4 categories: factual, temporal, inference, detailed).

```bash
cd gaama/evals/locomo

# Full pipeline: create LTM + evaluate
python run_evaluation.py

# Neural PPR training pipeline (requires existing LTM)
python run_neural_ppr.py           # all steps
python run_neural_ppr.py --step 1  # generate training data only
python run_neural_ppr.py --step 2  # summarize training data
python run_neural_ppr.py --step 3  # train model only
python run_neural_ppr.py --step 4  # evaluate only
```

## Key Results (LoCoMo-10, hypothesis eval with GPT-4o-mini)

| Configuration | Cat1 Factual | Cat2 Temporal | Cat3 Inference | Cat4 Detailed | Overall |
|---|---|---|---|---|---|
| Similarity only (ppr=0) | 69.2% | 48.4% | 35.9% | 81.4% | 69.5% |
| Fixed PPR (ppr=1.0) | 68.1% | 47.3% | **40.0%** | **82.1%** | 69.7% |
| **Neural Adaptive PPR + Hub Dampening** | **70.0%** | **49.7%** | **39.7%** | 81.9% | **70.4%** |

Neural adaptive PPR with hub dampening achieves the best overall accuracy by selectively applying graph-based retrieval where it helps (cat3 inference, cat4 detailed) while avoiding degradation on queries where similarity alone is sufficient (cat1 factual).

## File Structure

```
gaama/
├── __init__.py                     # Exports GAAMA, create_default_sdk
├── README.md
├── api/
│   └── client.py                   # AgenticMemorySDK, create_default_sdk
├── core/
│   ├── types.py                    # MemoryNode, Edge, MemoryPack, TraceEvent, etc.
│   └── policies.py                 # ExtractionPolicy
├── config/
│   └── settings.py                 # SDKSettings, StorageSettings, LLMSettings
├── services/
│   ├── orchestrator.py             # Main orchestrator (ingest, create, retrieve)
│   ├── ltm.py                      # LTM retrieval engine (KNN + PPR + hub dampening)
│   ├── pagerank.py                 # Personalized PageRank with edge-type weights
│   ├── neural_ppr.py               # Neural adaptive PPR model (gap regression MLP)
│   ├── llm_extractors.py           # LLM-based memory extraction
│   ├── graph_edges.py              # Edge construction from extracted nodes
│   ├── hybrid_search.py            # BM25 + semantic fusion search
│   ├── semantic_canonicalization.py # Node/edge deduplication
│   ├── answer_from_memory.py       # Generate answers from retrieved memory
│   ├── defaults.py                 # Default normalizer, evaluator
│   └── interfaces.py               # Protocol definitions
├── adapters/
│   ├── sqlite_memory.py            # SQLite node + edge store with FTS5
│   ├── sqlite_vector.py            # sqlite-vec vector store
│   ├── openai_embedding.py         # OpenAI embedding adapter
│   ├── openai_llm.py               # OpenAI LLM adapter
│   ├── llm_factory.py              # LLM adapter factory
│   ├── local_blob.py               # File-based blob store
│   └── interfaces.py               # Adapter protocols
├── infra/
│   ├── prompt_loader.py            # Load prompt templates from markdown
│   ├── serialization.py            # Node JSON serialization
│   └── vector_math.py              # Cosine similarity
├── prompts/                        # Prompt templates (markdown)
│   ├── ltm_extraction.md
│   ├── answer_from_memory.md
│   ├── episode_summary.md
│   ├── episode_summary_update.md
│   ├── retrieval_budget.md
│   └── stm_working_notes.md
└── evals/
    └── locomo/
        ├── locomo_pipeline.py      # Full LoCoMo evaluation pipeline
        ├── run_neural_ppr.py       # Neural PPR training + evaluation
        ├── run_evaluation.py       # Full evaluation runner
        └── locomo10.json           # LoCoMo-10 dataset (10 samples, 1540 questions)
```
