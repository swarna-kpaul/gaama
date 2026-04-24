"""Graph Edit Learning (GEL) -- post-retrieval corrective layer.

When retrieved memory is insufficient to answer a query, GEL dynamically
generates analysis questions, retrieves memory for each, reasons over
graph structure, and makes targeted graph edits to improve future retrievals.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

from gaama.adapters.interfaces import (
    EmbeddingAdapter,
    GraphStoreAdapter,
    LLMAdapter,
    NodeStoreAdapter,
    VectorStoreAdapter,
)
from gaama.core import (
    Edge,
    MemoryNode,
    MemoryPack,
    QueryFilters,
    Scope,
)
from gaama.core.types import (
    GELConfig,
    GELEditOp,
    GELReport,
    SubQuestionResult,
)
from gaama.infra.prompt_loader import load_prompt
from gaama.infra.serialization import node_to_embed_text
from gaama.services.graph_edges import make_edge
from gaama.services.interfaces import RetrievalEngine, RetrieveOptions

logger = logging.getLogger(__name__)


def _strip_json_block(text: str) -> str:
    """Remove markdown code fences around JSON if present."""
    text = text.strip()
    for pattern in (r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", r"^```\s*\n?(.*?)\n?```\s*$"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def _safe_json_parse(raw: str, llm: LLMAdapter, max_tokens: int = 4096) -> Dict[str, Any]:
    """Parse JSON from LLM output, with one retry on failure."""
    cleaned = _strip_json_block(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s -- retrying", e)
        retry_prompt = (
            f"Your previous response could not be parsed as valid JSON.\n\n"
            f"Error: {e}\n\n"
            f"Your previous output (first 500 chars):\n{raw[:500]}\n\n"
            f"Please return ONLY valid JSON with no markdown fences, no extra text."
        )
        raw2 = llm.complete(retry_prompt, max_tokens=max_tokens)
        cleaned2 = _strip_json_block(raw2 or "")
        try:
            return json.loads(cleaned2)
        except json.JSONDecodeError:
            logger.error("JSON retry also failed")
            return {}


def _node_content_brief(node: MemoryNode) -> str:
    """Short content string for a node (for graph summaries)."""
    kind = (node.kind or "").strip().lower()
    if kind == "fact":
        return (node.fact_text or "").strip()
    elif kind == "episode":
        return (node.summary or "").strip()
    elif kind == "reflection":
        return (node.reflection_text or "").strip()
    elif kind == "concept":
        return (node.concept_label or "").strip()
    elif kind == "skill":
        return (node.skill_description or "").strip()
    return (node.name or "").strip()


class GraphEditLearner:
    """Post-retrieval corrective layer for the knowledge graph.

    Two modes:
    - Batch mode (primary): Run after evaluation on failed questions with ground truth.
    - Online mode: Run after retrieval without ground truth, using LLM sufficiency judge.
    """

    def __init__(
        self,
        node_store: NodeStoreAdapter,
        graph_store: GraphStoreAdapter,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
        llm: LLMAdapter,
        retriever: RetrievalEngine,
        config: GELConfig | None = None,
    ) -> None:
        self._node_store = node_store
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm
        self._retriever = retriever
        self._config = config or GELConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def learn_from_failure(
        self,
        query: str,
        ground_truth: str | None,
        hypothesis: str,
        reward_before: float,
        memory_pack: MemoryPack,
        scored_items: List[Tuple[str, str, float]],
        scope: Scope,
        filters: QueryFilters,
        retrieve_options: RetrieveOptions | None = None,
    ) -> GELReport:
        """Run the full GEL pipeline on a failed query.

        Args:
            ground_truth: Reference answer. If None, runs in online mode
                (uses LLM judge for Phase 1, ground-truth-free prompts
                for Phases 2 & 4).

        Returns a GELReport with details of analysis and edits.
        """
        online_mode = ground_truth is None
        report = GELReport(query=query, reward_before=reward_before)

        # Phase 1: Check if GEL should run
        if online_mode:
            # Online mode: LLM judge
            judge_score = self._judge_retrieval(query, memory_pack, hypothesis)
            report.reward_before = judge_score
            if judge_score >= self._config.reward_threshold:
                return report
        else:
            if reward_before >= self._config.reward_threshold:
                return report

        # Phase 2: Generate analysis questions
        questions = self._generate_analysis_questions(
            query, ground_truth, hypothesis, memory_pack,
        )
        report.analysis_questions_generated = len(questions)
        if not questions:
            return report

        # Phase 3: Graph exploration (no LLM calls)
        results = self._explore_graph(questions, scope, filters, retrieve_options)

        # Phase 4: Chain-of-thought reasoning + edit planning
        cot, edit_ops = self._reason_and_plan(
            query, ground_truth, hypothesis, results, memory_pack,
        )
        report.chain_of_thought = cot
        report.edits_planned = len(edit_ops)
        report.edit_ops = edit_ops

        # Phase 4b: Verify edits (LLM quality gate)
        if self._config.verify_edits_before_insert and edit_ops:
            edit_ops = self._verify_edits(
                query, hypothesis, edit_ops, memory_pack, results,
            )
            report.edit_ops = edit_ops
            report.edits_planned = len(edit_ops)

        # Phase 5: Execute edits
        executed, skipped = self._execute_edits(edit_ops, scope)
        report.edits_executed = executed
        report.edits_skipped = skipped

        # Phase 6: Optional verification
        if self._config.verify_after_edit and retrieve_options is not None:
            report.reward_after = self._verify_improvement(
                query, ground_truth or "", scope, filters, retrieve_options,
            )

        return report

    # ------------------------------------------------------------------
    # Phase 1: Judge retrieval quality (online mode only)
    # ------------------------------------------------------------------

    def _judge_retrieval(
        self, query: str, memory_pack: MemoryPack, hypothesis: str = "",
    ) -> float:
        """Score retrieval sufficiency using LLM. Returns 0.0-1.0."""
        memory_text = memory_pack.to_text(include_citations=False)
        prompt = load_prompt("gel_judge", {
            "query": query,
            "hypothesis": hypothesis or "(No answer generated)",
            "memory_text": memory_text or "(empty)",
        })
        raw = self._llm.complete(prompt, max_tokens=256, temperature=0)
        data = _safe_json_parse(raw or "", self._llm, max_tokens=256)
        try:
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Phase 2: Dynamic question generation
    # ------------------------------------------------------------------

    def _generate_analysis_questions(
        self,
        query: str,
        ground_truth: str | None,
        hypothesis: str,
        memory_pack: MemoryPack,
    ) -> List[Dict[str, str]]:
        """Generate analysis questions tailored to this specific failure."""
        memory_text = memory_pack.to_text(include_citations=False)
        max_q = self._config.max_analysis_questions

        if ground_truth is not None:
            # Batch mode: use ground truth
            prompt = load_prompt("gel_decompose", {
                "query": query,
                "ground_truth": ground_truth,
                "hypothesis": hypothesis or "(No answer generated)",
                "retrieved_memory": memory_text or "(empty)",
                "max_questions": str(max_q),
            })
        else:
            # Online mode: no ground truth
            prompt = load_prompt("gel_decompose_online", {
                "query": query,
                "hypothesis": hypothesis or "(No answer generated)",
                "retrieved_memory": memory_text or "(empty)",
                "max_questions": str(max_q),
            })

        raw = self._llm.complete(prompt, max_tokens=2048, temperature=0)
        data = _safe_json_parse(raw or "", self._llm, max_tokens=2048)

        questions = data.get("analysis_questions", [])
        if not isinstance(questions, list):
            return []

        # Validate and cap
        valid: List[Dict[str, str]] = []
        for q in questions[:max_q]:
            if isinstance(q, dict) and q.get("question"):
                valid.append({
                    "question": str(q["question"]),
                    "reasoning": str(q.get("reasoning", "")),
                })
        return valid

    # ------------------------------------------------------------------
    # Phase 3: Graph exploration (no LLM calls)
    # ------------------------------------------------------------------

    def _explore_graph(
        self,
        questions: List[Dict[str, str]],
        scope: Scope,
        filters: QueryFilters,
        retrieve_options: RetrieveOptions | None = None,
    ) -> List[SubQuestionResult]:
        """For each analysis question, retrieve and expand subgraph."""
        from gaama.core import RetrievalBudget
        results: List[SubQuestionResult] = []

        for q in questions:
            question_text = q["question"]
            reasoning = q.get("reasoning", "")

            # Build retrieve options for this sub-question
            if retrieve_options is not None:
                from dataclasses import replace
                opts = replace(retrieve_options)
            else:
                opts = RetrieveOptions(
                    filters=filters,
                    budget=RetrievalBudget(
                        max_facts=5, max_reflections=2,
                        max_skills=0, max_episodes=10,
                    ),
                    sources="ltm",
                    max_memory_words=800,
                )

            # Retrieve
            pack, scored_items = self._retriever.retrieve(question_text, opts)

            # Retrieve related concepts for the retrieved memory set
            concept_nodes, concept_edges = self._retrieve_related_concepts(scored_items)

            results.append(SubQuestionResult(
                question=question_text,
                reasoning=reasoning,
                retrieved_pack=pack,
                scored_items=list(scored_items),
                subgraph_nodes=concept_nodes,
                subgraph_edges=concept_edges,
            ))

        return results

    def _retrieve_related_concepts(
        self, scored_items: Sequence[Tuple[str, str, float]],
    ) -> Tuple[List[MemoryNode], List[Edge]]:
        """Retrieve concept nodes connected to the retrieved memory set."""
        node_ids = [nid for nid, _, _ in scored_items if nid]
        if not node_ids:
            return [], []

        # Get edges for retrieved nodes
        edges = list(self._graph_store.get_edges_for_nodes(node_ids))

        # Filter to concept-related edges
        concept_edge_types = {"ABOUT_CONCEPT", "HAS_CONCEPT"}
        concept_edges = [e for e in edges if e.edge_type in concept_edge_types]

        # Collect concept node IDs (nodes not already in the retrieved set)
        retrieved_set = set(node_ids)
        concept_ids: Set[str] = set()
        for e in concept_edges:
            if e.source_id not in retrieved_set:
                concept_ids.add(e.source_id)
            if e.target_id not in retrieved_set:
                concept_ids.add(e.target_id)

        # Fetch concept nodes
        concept_nodes: List[MemoryNode] = []
        if concept_ids:
            concept_nodes = list(self._node_store.get_nodes(list(concept_ids)))

        return concept_nodes, concept_edges

    # ------------------------------------------------------------------
    # Phase 4: Chain-of-thought reasoning + edit planning
    # ------------------------------------------------------------------

    def _reason_and_plan(
        self,
        query: str,
        ground_truth: str | None,
        hypothesis: str,
        results: List[SubQuestionResult],
        memory_pack: MemoryPack | None = None,
    ) -> Tuple[str, List[GELEditOp]]:
        """Reason over retrieved memory and related concepts, then plan edits."""
        # Collect existing facts for dedup context
        existing_facts_set: set[str] = set()
        if memory_pack:
            for f in memory_pack.facts:
                existing_facts_set.add(f.strip())
        for result in results:
            if result.retrieved_pack:
                for f in result.retrieved_pack.facts:
                    existing_facts_set.add(f.strip())
        existing_facts_text = "\n".join(f"- {f}" for f in sorted(existing_facts_set)) or "(none)"

        # Format exploration results
        exploration_parts = []
        for idx, result in enumerate(results):
            lines = [
                f"### Analysis Question {idx + 1}",
                f"**Question**: {result.question}",
                f"**Reasoning**: {result.reasoning}",
                "",
                "**Retrieved Nodes:**",
            ]
            if not result.scored_items:
                lines.append("  (no results)")
            else:
                for nid, content, score in result.scored_items:
                    lines.append(f"  - [{nid}] score={score:.3f}: {content[:200]}")

            if result.subgraph_nodes:
                lines.append("")
                lines.append("**Related Concepts:**")
                for node in result.subgraph_nodes:
                    label = _node_content_brief(node)
                    lines.append(f"  - [{node.node_id}]: {label}")

            if result.subgraph_edges:
                lines.append("")
                lines.append("**Concept Edges:**")
                for e in result.subgraph_edges[:20]:
                    lines.append(f"  - {e.source_id} --{e.edge_type}--> {e.target_id}")

            exploration_parts.append("\n".join(lines))
        exploration_text = "\n\n".join(exploration_parts)

        # Build concept summary across all questions
        all_concepts: Dict[str, MemoryNode] = {}
        for result in results:
            for node in result.subgraph_nodes:
                if (node.kind or "").strip().lower() == "concept":
                    all_concepts[node.node_id] = node
        graph_lines = ["## Related Concepts"]
        if all_concepts:
            for nid, node in all_concepts.items():
                graph_lines.append(f"  - [{nid}]: {_node_content_brief(node)}")
        else:
            graph_lines.append("  (no concepts found)")
        graph_summary = "\n".join(graph_lines)

        if ground_truth is not None:
            # Batch mode: use ground truth
            prompt = load_prompt("gel_reason_and_plan", {
                "query": query,
                "ground_truth": ground_truth,
                "hypothesis": hypothesis or "(No answer generated)",
                "exploration_results": exploration_text,
                "graph_summary": graph_summary,
                "existing_facts": existing_facts_text,
                "max_edits": str(self._config.max_edits_per_query),
            })
        else:
            # Online mode: no ground truth
            prompt = load_prompt("gel_reason_and_plan_online", {
                "query": query,
                "hypothesis": hypothesis or "(No answer generated)",
                "exploration_results": exploration_text,
                "graph_summary": graph_summary,
                "existing_facts": existing_facts_text,
                "max_edits": str(self._config.max_edits_per_query),
            })

        raw = self._llm.complete(prompt, max_tokens=4096, temperature=0)
        data = _safe_json_parse(raw or "", self._llm, max_tokens=4096)

        # Extract chain of thought
        cot = data.get("chain_of_thought", {})
        if isinstance(cot, dict):
            cot_text = json.dumps(cot, indent=2)
        else:
            cot_text = str(cot)

        # Extract edit ops
        raw_ops = data.get("edit_ops", [])
        if not isinstance(raw_ops, list):
            return cot_text, []

        edit_ops: List[GELEditOp] = []
        for op in raw_ops[:self._config.max_edits_per_query]:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("op_type", "")).upper()
            params = op.get("params", {})
            if not isinstance(params, dict):
                params = {}
            root_cause = str(op.get("root_cause", ""))
            if op_type in ("CREATE_FACT", "CREATE_CONCEPT"):
                edit_ops.append(GELEditOp(
                    op_type=op_type,
                    params=params,
                    root_cause=root_cause,
                ))

        return cot_text, edit_ops

    # ------------------------------------------------------------------
    # Phase 4b: Verify edits (LLM quality gate)
    # ------------------------------------------------------------------

    def _verify_edits(
        self,
        query: str,
        hypothesis: str,
        edit_ops: List[GELEditOp],
        memory_pack: MemoryPack,
        results: List[SubQuestionResult],
    ) -> List[GELEditOp]:
        """Post-generation verification: LLM filters out inaccurate/irrelevant edits."""
        if not edit_ops:
            return []

        # Build the proposed edits text
        edit_lines = []
        for i, op in enumerate(edit_ops):
            if op.op_type == "CREATE_FACT":
                edit_lines.append(
                    f"[{i}] CREATE_FACT: \"{op.params.get('fact_text', '')}\""
                    f" (belief={op.params.get('belief', 0.85)})"
                )
            elif op.op_type == "CREATE_CONCEPT":
                edit_lines.append(
                    f"[{i}] CREATE_CONCEPT: \"{op.params.get('concept_label', '')}\""
                )
        proposed_text = "\n".join(edit_lines)

        # Build memory context from pack + sub-question results
        memory_text = memory_pack.to_text(include_citations=False)
        for result in results:
            if result.retrieved_pack:
                sub_text = result.retrieved_pack.to_text(include_citations=False)
                if sub_text:
                    memory_text += f"\n\n### Sub-question: {result.question}\n{sub_text}"

        prompt = load_prompt("gel_verify", {
            "query": query,
            "hypothesis": hypothesis or "(No answer generated)",
            "memory_text": memory_text or "(empty)",
            "proposed_edits": proposed_text,
        })

        raw = self._llm.complete(prompt, max_tokens=2048, temperature=0)
        data = _safe_json_parse(raw or "", self._llm, max_tokens=2048)

        passed_indices = data.get("passed_indices", [])
        if not isinstance(passed_indices, list):
            logger.warning("Verification returned non-list passed_indices; keeping all edits")
            return edit_ops

        # Log rejections
        for rej in data.get("rejections", []):
            if isinstance(rej, dict):
                logger.info(
                    "GEL verification rejected edit %s: %s",
                    rej.get("edit_index", "?"), rej.get("reason", "?"),
                )

        # Filter original edit_ops by passed indices
        verified_ops: List[GELEditOp] = []
        for idx in passed_indices:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(edit_ops):
                verified_ops.append(edit_ops[i])

        logger.info(
            "GEL verification: %d/%d edits passed",
            len(verified_ops), len(edit_ops),
        )
        return verified_ops

    # ------------------------------------------------------------------
    # Duplicate detection helpers
    # ------------------------------------------------------------------

    def _is_duplicate_fact(self, fact_text: str, scope: Scope) -> bool:
        """Check if a candidate fact is a near-duplicate of an existing fact."""
        threshold = self._config.dedup_similarity_threshold
        filters = QueryFilters(
            agent_id=scope.agent_id,
            user_id=scope.user_id,
            task_id=scope.task_id,
        )
        results = self._vector_store.search(
            fact_text, filters, top_k=5, kind="node"
        )
        for node, sim in results:
            if sim >= threshold:
                existing_text = (node.fact_text or node.summary or "").strip()
                if existing_text:
                    logger.info(
                        "Rejecting duplicate fact (sim=%.3f): '%s' ~= '%s'",
                        sim, fact_text[:60], existing_text[:60],
                    )
                    return True
        return False

    def _is_duplicate_concept(self, label: str, scope: Scope) -> bool:
        """Check if a concept with similar label already exists."""
        threshold = self._config.dedup_similarity_threshold
        filters = QueryFilters(
            agent_id=scope.agent_id,
            user_id=scope.user_id,
            task_id=scope.task_id,
        )
        results = self._vector_store.search(
            label, filters, top_k=5, kind="node"
        )
        for node, sim in results:
            if sim >= threshold and (node.kind or "").strip().lower() == "concept":
                logger.info(
                    "Rejecting duplicate concept (sim=%.3f): '%s' ~= '%s'",
                    sim, label, node.concept_label or node.name,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Phase 5: Execute edits
    # ------------------------------------------------------------------

    def _execute_edits(
        self, edits: List[GELEditOp], scope: Scope,
    ) -> Tuple[int, int]:
        """Execute planned edits. Returns (executed, skipped)."""
        executed = 0
        skipped = 0
        facts_created = 0
        concepts_created = 0

        for edit in edits:
            try:
                if edit.op_type == "CREATE_FACT" and facts_created >= self._config.max_facts_per_query:
                    logger.info("Skipping CREATE_FACT (max_facts_per_query=%d reached)", self._config.max_facts_per_query)
                    skipped += 1
                    continue

                if edit.op_type == "CREATE_CONCEPT" and concepts_created >= self._config.max_concepts_per_query:
                    logger.info("Skipping CREATE_CONCEPT (max_concepts_per_query=%d reached)", self._config.max_concepts_per_query)
                    skipped += 1
                    continue

                success = False
                if edit.op_type == "CREATE_FACT":
                    success = self._execute_create_fact(edit.params, scope)
                elif edit.op_type == "CREATE_CONCEPT":
                    success = self._execute_create_concept(edit.params, scope)

                if success:
                    executed += 1
                    if edit.op_type == "CREATE_FACT":
                        facts_created += 1
                    elif edit.op_type == "CREATE_CONCEPT":
                        concepts_created += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning("GEL edit %s failed: %s", edit.op_type, e)
                skipped += 1

        return executed, skipped

    _HEDGING_BLOCKLIST = [
        "not specified", "not mentioned", "not clear", "unknown",
        "may include", "not explicitly", "but it is not", "no information",
        "does not", "not recorded", "not available", "not discussed",
        "cannot determine", "unable to determine", "doesn't mention",
        "does not mention", "doesn't specify", "does not specify",
    ]

    def _execute_create_fact(self, params: Dict[str, Any], scope: Scope) -> bool:
        """Create a new fact node with edges to source episodes and concepts."""
        fact_text = str(params.get("fact_text", "")).strip()
        if not fact_text:
            return False

        # Reject hedging/negative facts
        lower = fact_text.lower()
        if any(phrase in lower for phrase in self._HEDGING_BLOCKLIST):
            logger.info("Rejecting hedging fact: %s", fact_text[:80])
            return False

        # Reject near-duplicates of existing facts
        if self._is_duplicate_fact(fact_text, scope):
            return False

        cap = self._config.gel_fact_belief
        belief = min(cap, max(0.1, float(params.get("belief", cap))))
        now = datetime.utcnow()

        node_id = f"fact-gel-{uuid4().hex[:12]}"
        node = MemoryNode(
            node_id=node_id,
            created_at=now,
            updated_at=now,
            kind="fact",
            fact_text=fact_text,
            belief=belief,
            scopes=[scope],
            tags={"source": "gel"},
        )

        # Embed and upsert
        self._embed_and_upsert_node(node, scope)

        # Create DERIVED_FROM edges to source episodes
        source_episode_ids = params.get("source_episode_ids", [])
        if isinstance(source_episode_ids, list):
            existing = self._node_store.get_nodes(source_episode_ids)
            existing_ids = {n.node_id for n in existing}
            edges = []
            for ep_id in source_episode_ids:
                if ep_id in existing_ids:
                    edges.append(make_edge(node_id, ep_id, "DERIVED_FROM"))
            if edges:
                self._graph_store.upsert_edges(edges)

        # Create ABOUT_CONCEPT edges
        concepts = params.get("concepts", [])
        if isinstance(concepts, list):
            self._connect_to_concepts(node_id, concepts, scope)

        return True

    def _execute_create_concept(self, params: Dict[str, Any], scope: Scope) -> bool:
        """Create a new concept node with edges to connected facts/episodes."""
        label = str(params.get("concept_label", "")).strip()
        if not label:
            return False

        # Reject orphan concepts (no connected nodes)
        connected_ids = params.get("connected_node_ids", [])
        if not isinstance(connected_ids, list) or not connected_ids:
            logger.info("Rejecting orphan concept (no connected_node_ids): %s", label)
            return False

        # Reject near-duplicate concept labels
        if self._is_duplicate_concept(label, scope):
            return False

        now = datetime.utcnow()
        # Use a deterministic-ish ID based on label
        label_slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        node_id = f"concept-{label_slug}"

        # Check if already exists
        existing = self._node_store.get_nodes([node_id])
        if not existing:
            node = MemoryNode(
                node_id=node_id,
                created_at=now,
                updated_at=now,
                kind="concept",
                concept_label=label,
                scopes=[scope],
                tags={"source": "gel"},
            )
            self._embed_and_upsert_node(node, scope)

        # Create edges to connected nodes
        connected_ids = params.get("connected_node_ids", [])
        if isinstance(connected_ids, list) and connected_ids:
            connected_nodes = self._node_store.get_nodes(connected_ids)
            edges: List[Edge] = []
            for n in connected_nodes:
                kind = (n.kind or "").strip().lower()
                if kind == "fact":
                    edges.append(make_edge(n.node_id, node_id, "ABOUT_CONCEPT"))
                elif kind == "episode":
                    edges.append(make_edge(n.node_id, node_id, "HAS_CONCEPT"))
            if edges:
                self._graph_store.upsert_edges(edges)

        return True

    # ------------------------------------------------------------------
    # Phase 6: Verify improvement (optional)
    # ------------------------------------------------------------------

    def _verify_improvement(
        self,
        query: str,
        ground_truth: str,
        scope: Scope,
        filters: QueryFilters,
        retrieve_options: RetrieveOptions,
    ) -> float:
        """Re-retrieve and re-judge to measure improvement."""
        from gaama.services.answer_from_memory import answer_from_memory

        pack, _ = self._retriever.retrieve(query, retrieve_options)
        hypothesis = answer_from_memory(query, pack, llm=self._llm, temperature=0)

        # Simple sufficiency judge
        score = self._judge_retrieval(query, pack)
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_and_upsert_node(self, node: MemoryNode, scope: Scope) -> None:
        """Embed a node's text and upsert to all stores."""
        text = node_to_embed_text(node)
        if text and self._embedder:
            emb = self._embedder.embed(text)
            if emb:
                node.embedding = list(emb)

        self._node_store.upsert_nodes([node])
        if node.embedding:
            self._vector_store.upsert_embeddings(
                [node],
                agent_id=scope.agent_id,
                user_id=scope.user_id,
                task_id=scope.task_id,
            )

    def _connect_to_concepts(
        self, node_id: str, concept_labels: List[str], scope: Scope,
    ) -> None:
        """Connect a node to concept nodes (creating them if needed)."""
        edges: List[Edge] = []
        for label in concept_labels:
            if not label:
                continue
            label_slug = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
            concept_id = f"concept-{label_slug}"

            # Create concept if it doesn't exist
            existing = self._node_store.get_nodes([concept_id])
            if not existing:
                now = datetime.utcnow()
                concept_node = MemoryNode(
                    node_id=concept_id,
                    created_at=now,
                    updated_at=now,
                    kind="concept",
                    concept_label=str(label),
                    scopes=[scope],
                    tags={"source": "gel"},
                )
                self._embed_and_upsert_node(concept_node, scope)

            edges.append(make_edge(node_id, concept_id, "ABOUT_CONCEPT"))

        if edges:
            self._graph_store.upsert_edges(edges)

