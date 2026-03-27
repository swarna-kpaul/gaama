# GAAMA: Graph Augmented Associative Memory for Agents

GAAMA is a long-term memory system for AI agents that combines a structured knowledge graph with Personalized PageRank (PPR) retrieval. It extracts facts, concepts, episodes, and reflections from agent conversations using a concept-node pipeline, stores them in a knowledge graph with typed edges, and retrieves relevant memories using semantic KNN or graph-enhanced PPR scoring.

## Architecture Overview

GAAMA's pipeline works in three stages:

### 1. Concept-Node LTM Creation

Conversations are processed in three steps (500 tokens per chunk, `gpt-4o-mini`):

- **Step 1 — Episodes**: Raw conversation turns become episode nodes (no LLM needed). Each turn is stored as-is with timestamps and metadata. Episodes are chained via `NEXT` edges.
- **Step 2 — Facts + Concepts**: An LLM extracts atomic facts and topic concepts from episodes. Facts connect to their source episodes via `DERIVED_FROM` edges. Concepts connect to episodes via `HAS_CONCEPT` and to facts via `ABOUT_CONCEPT` edges. Concepts serve as cross-cutting paths through the graph.
- **Step 3 — Reflections**: An LLM generates higher-order insights by synthesizing multiple facts. Reflections connect to source facts via `DERIVED_FROM_FACT` edges.

### 2. Retrieval (Two Modes)

**Semantic Retrieval** (`semantic_only=True`):
- Pure cosine similarity KNN over node embeddings
- Ranks all node kinds (facts, episodes, reflections) by relevance
- Trimmed to 1000 words with proportional budget allocation
- Best for direct factual and detailed questions

**PPR-Enhanced Retrieval** (`ppr_score_weight=0.1`):
- KNN finds seed nodes → top 40 seeds selected by similarity
- Graph expansion (depth=2) via concept/fact/episode edges
- Personalized PageRank propagates relevance through the graph (alpha=0.6)
- Final score: `score = 0.1 * PPR(node) + 1.0 * sim(node, query)`
- Best for temporal and cross-entity questions

### 3. Memory Pack with Word Trimming

Retrieved memories are packed into a `MemoryPack` (facts, reflections, skills, episodes) and trimmed to fit a word budget (default 1000 words). Trimming is proportional to budget caps — categories that are over-represented lose items first. Episodes are trimmed by lowest relevance score to preserve temporal order.

### Edge Types

The concept-node graph uses 8 edge types:

| Edge Type | Connects | Weight |
|---|---|---|
| `NEXT` | Episode → Episode (temporal chain) | 0.8 |
| `DERIVED_FROM` | Fact → Episode (provenance) | 0.8 |
| `HAS_CONCEPT` | Episode → Concept | 0.8 |
| `ABOUT_CONCEPT` | Fact → Concept | 0.8 |
| `DERIVED_FROM_FACT` | Reflection → Fact | 0.5 |

### Storage

- **SQLite** for nodes, edges, and FTS5 full-text index (BM25)
- **sqlite-vec** for vector embeddings (cosine similarity KNN)
- **Local blob store** for raw trace archives

## Installation

```bash
pip install openai sqlite-vec
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

# Ingest conversation events
events = [
    TraceEvent(event_type="message", actor="user",
               content="I visited the Eiffel Tower in Paris last week."),
    TraceEvent(event_type="message", actor="assistant",
               content="That sounds wonderful! How was the view from the top?"),
]
sdk._orchestrator.ingest(events, agent_id="my-agent")

# Create long-term memories (concept-node pipeline)
node_ids = sdk.create(CreateOptions(
    agent_id="my-agent", user_id="user-1", task_id="conv-1",
    max_tokens_per_chunk=500,
))
print(f"Created {len(node_ids)} memory nodes")

# Retrieve with semantic retrieval
pack = sdk.retrieve(
    query="What did the user do in Paris?",
    filters=QueryFilters(agent_id="my-agent", user_id="user-1", task_id="conv-1"),
    budget=RetrievalBudget(max_facts=60, max_reflections=20, max_skills=5, max_episodes=80),
    semantic_only=True,
    max_memory_words=1000,
)
print(pack.to_text())

# Retrieve with PPR-enhanced retrieval
pack = sdk.retrieve(
    query="What did the user do in Paris?",
    filters=QueryFilters(agent_id="my-agent", user_id="user-1", task_id="conv-1"),
    budget=RetrievalBudget(max_facts=60, max_reflections=20, max_skills=5, max_episodes=80),
    ppr_score_weight=0.1,
    sim_score_weight=1.0,
    expansion_depth=2,
    max_memory_words=1000,
)
print(pack.to_text())
```

## Running the LoCoMo Evaluation

The LoCoMo benchmark evaluates long-term memory over multi-session conversations (10 conversations, ~1540 questions across 4 categories: factual, temporal, inference, detailed).

Evaluation scripts are in `gaama/evals/locomo/`:

```bash
cd gaama/evals/locomo

# Step 1: Create LTM for all samples (or specific ones)
python run_create_ltm.py                          # all 10 samples
python run_create_ltm.py --sample-ids conv-26     # specific sample
python run_create_ltm.py --limit 3                # first 3 samples

# Step 2a: Evaluate with semantic retrieval
python run_semantic_eval.py                              # all samples
python run_semantic_eval.py --sample-ids conv-26 conv-30 # specific
python run_semantic_eval.py --max-words 1200             # custom word limit

# Step 2b: Evaluate with PPR retrieval
python run_ppr_eval.py                                   # all samples
python run_ppr_eval.py --sample-ids conv-26              # specific
python run_ppr_eval.py --ppr-weight 0.5                  # custom PPR weight

# Step 2c: RAG baseline (no LTM needed, requires index build first)
python run_rag_baseline.py --step all                    # index + evaluate
python run_rag_baseline.py --step all --sample-ids conv-26
python run_rag_baseline.py --step 2 --max-words 1000     # evaluate only (index must exist)
```

Results are saved as per-sample JSONL files in `gaama/evals/locomo/data/results/`.

### Default Configurations

| Parameter | Semantic | PPR | RAG |
|---|---|---|---|
| ppr_weight | 0.0 | 0.1 | — |
| sim_weight | 1.0 | 1.0 | — |
| facts budget | 60 | 60 | — |
| reflections budget | 20 | 20 | — |
| episodes budget | 80 | 80 | — |
| max_memory_words | 1000 | 1000 | 1000 |
| expansion_depth | — | 2 | — |
| alpha (PPR damping) | — | 0.6 | — |
| seeds (top-k KNN) | — | 40 | — |
| max_tokens_per_chunk | 500 | 500 | — |
| LLM model | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini |
| Embedding model | text-embedding-3-small | text-embedding-3-small | text-embedding-3-small |

## Key Results (LoCoMo-10, 1000w, fractional hypothesis eval with GPT-4o-mini)

### Overall (1540 questions, 10 conversations)

| Method | Reward |
|---|---|
| RAG Baseline | 0.750 |
| LTM Semantic | 0.780 |
| **LTM PPR=0.1** | **0.789** |

### By Category (Reward)

| Category | N | RAG | Semantic | PPR=0.1 |
|---|---|---|---|---|
| Multihop Reasoning | 282 | 0.675 | **0.722** | 0.722 |
| Temporal | 321 | 0.590 | 0.715 | **0.719** |
| Commonsense | 96 | 0.446 | 0.492 | **0.493** |
| Single Hop | 841 | 0.871 | 0.857 | **0.872** |

### Key Findings

- **PPR=0.1 is best overall**: +3.9pp reward over RAG, +1.0pp over Semantic
- **LTM crushes RAG on temporal**: +12.9pp — structured memory with timestamps and episode chaining wins
- **Minimal PPR weight (0.1)** works best — strong graph signal (PPR=1.0) displaces high-similarity nodes from the word-limited context, hurting accuracy
- **Single Hop**: RAG and PPR tied — raw conversation context preserves fine-grained detail

## File Structure

```
gaama/
├── __init__.py                     # Exports GAAMA, create_default_sdk
├── README.md
├── api/
│   └── client.py                   # AgenticMemorySDK, create_default_sdk
├── core/
│   ├── types.py                    # MemoryNode, Edge, MemoryPack, TraceEvent
│   └── policies.py                 # ExtractionPolicy
├── config/
│   └── settings.py                 # SDKSettings, StorageSettings, LLMSettings
├── services/
│   ├── orchestrator.py             # Main orchestrator (ingest, create, retrieve)
│   ├── ltm_creator.py              # Concept-node LTM creation pipeline
│   ├── ltm_retriever.py            # LTM retrieval (semantic, PPR, hybrid)
│   ├── pagerank.py                 # Personalized PageRank with edge-type weights
│   ├── llm_extractors.py           # LLM fact/reflection extractors
│   ├── graph_edges.py              # Edge construction
│   ├── hybrid_search.py            # BM25 + semantic fusion search
│   ├── answer_from_memory.py       # Generate answers from retrieved memory
│   ├── defaults.py                 # Default normalizer, evaluator
│   └── interfaces.py               # Protocol definitions
├── adapters/
│   ├── sqlite_memory.py            # SQLite node + edge store with FTS5
│   ├── sqlite_vector.py            # sqlite-vec vector store
│   ├── openai_embedding.py         # OpenAI embedding adapter
│   ├── openai_llm.py               # OpenAI LLM adapter
│   └── interfaces.py               # Adapter protocols
├── infra/
│   ├── prompt_loader.py            # Load prompt templates from markdown
│   ├── serialization.py            # Node JSON serialization
│   ├── id_helpers.py               # Canonical ID generation helpers
│   └── vector_math.py              # Cosine similarity
├── prompts/
│   ├── fact_generation.md           # Fact + concept extraction prompt
│   ├── reflection_generation.md     # Reflection generation prompt
│   └── answer_from_memory.md        # Answer generation prompt
└── evals/
    └── locomo/
        ├── locomo10.json            # LoCoMo-10 dataset (10 conversations, 1540 questions)
        ├── config.py                # Default evaluation configs
        ├── locomo_eval.py           # Core evaluation functions
        ├── run_create_ltm.py        # LTM creation CLI
        ├── run_semantic_eval.py     # Semantic retrieval evaluation CLI
        ├── run_ppr_eval.py          # PPR retrieval evaluation CLI
        └── run_rag_baseline.py      # RAG baseline evaluation CLI
```
