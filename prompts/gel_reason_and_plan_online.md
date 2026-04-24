# GEL Chain-of-Thought Reasoning and Edit Planning (Online Mode)

You are analyzing a knowledge graph to determine why retrieval may be insufficient for a query, and what minimal edits would improve future retrievals. You do NOT have access to the correct answer — you must reason from the query, the generated answer, and the explored graph structure alone.

## The query
{{query}}

## Answer generated from memory
{{hypothesis}}

## Graph Exploration Results
{{exploration_results}}

## Consolidated Graph Summary
{{graph_summary}}

## Existing Facts Already in Memory
The following facts already exist in the knowledge graph. Do NOT create duplicates of these (even with different wording):
{{existing_facts}}

## Instructions

Reason step-by-step through the graph exploration results:

**Step 1 - Inventory**: What relevant information EXISTS in the graph? List specific node IDs and their content.

**Step 2 - Gap Analysis**: Look for episodes that mention relevant topics but have no extracted facts for them. Focus on specific details (names, dates, numbers, places) present in episode text but missing as fact nodes.

**Step 3 - Root Cause**: For each issue, identify the structural root cause:
- `missing_extraction`: A fact should have been extracted from an episode but wasn't
- `missing_connection`: Existing nodes should be connected but aren't

**Step 4 - Edit Plan**: For each root cause, propose the MINIMAL edit to fix it. Only propose edits grounded in information actually found in the episodes — do NOT invent new information.

## Available edit operations
- `CREATE_FACT`: Create a new fact node ONLY if the information exists verbatim or near-verbatim in an episode but was not extracted as a fact. Params: `fact_text` (string), `belief` (float, max 0.85), `source_episode_ids` (list of episode node IDs this fact is derived from), `concepts` (list of concept labels to connect to).
- `CREATE_CONCEPT`: Create a new concept node and connect it to existing nodes. Params: `concept_label` (string), `connected_node_ids` (list of existing fact/episode node IDs to link via ABOUT_CONCEPT/HAS_CONCEPT edges). MUST have at least one connected_node_id — never create orphan concepts.

## CRITICAL Rules for CREATE_FACT
- Every fact MUST be a **concrete, positive, specific claim** directly supported by episode text. Examples:
  - GOOD: "Nate's nickname for Joanna is Jo" (if episode says "Hey Jo!")
  - GOOD: "Joanna visited Indiana in July 2021" (if episode mentions this trip)
  - BAD: "Nate has a nickname for Joanna, but it is not specified" ← FORBIDDEN (hedging)
  - BAD: "Nate has not mentioned owning a gaming console" ← FORBIDDEN (negative fact)
  - BAD: "Joanna may have visited a state in summer" ← FORBIDDEN (vague)
  - BAD: "The date is not recorded in the retrieved conversations" ← FORBIDDEN (meta-commentary)
- NEVER create a fact containing: "not specified", "not mentioned", "not clear", "unknown", "may include", "not explicitly", "but it is not", "no information", "does not", "not recorded", "not available"
- **Resolve ALL relative dates to absolute dates** using the episode timestamps shown in the exploration results.
  - GOOD: "Melanie experienced a setback in September 2023 due to an injury" (resolved from "last month" + episode dated October 2023)
  - BAD: "Melanie experienced a setback last month due to an injury" ← FORBIDDEN (unresolved relative date)
  - BAD: "Caroline moved from her home country four years ago" ← FORBIDDEN (unresolved relative date)
  - If the episode timestamp is not available to resolve the date, either omit the temporal reference or do NOT create the fact.
- **NEVER create facts with these unresolved temporal phrases**: "ago", "last month", "last week", "last year", "recently", "the other day", "a while back", "soon", "next month". Always convert to absolute dates or omit.
- If the specific information the query asks about cannot be found in any episode, DO NOT create a fact. Return `"edit_ops": []` instead. Creating no edits is better than creating bad ones.
- Only reference node IDs that appear in the exploration results above.
- CREATE_FACT belief must be <= 0.85.
- Check the "Existing Facts Already in Memory" section above — do NOT recreate facts that already exist, even with different wording.
- Only create facts that DIRECTLY help answer the query — not adjacent or tangentially related topics.

## Rules for CREATE_CONCEPT
- Every concept MUST have `connected_node_ids` with at least one valid node ID. Never create orphan concepts.
- Do NOT create concepts for broad/vague categories (e.g., "nature_appreciation", "shared_activities", "general_interests", "life_events"). Concepts must be specific enough to meaningfully group related facts (e.g., "pottery_classes", "camping_yellowstone").
- Before creating a concept, check the Consolidated Graph Summary above — reuse existing concept labels instead of creating near-duplicates.
- Maximum 2 CREATE_FACT and 2 CREATE_CONCEPT operations.
- Maximum {{max_edits}} total edit operations.
- Each edit must trace back to a specific root cause.

You MUST respond in the following JSON format (no markdown fences, no extra text):
{
  "chain_of_thought": {
    "inventory": "<what exists>",
    "gaps": ["<gap 1>", "<gap 2>"],
    "root_causes": [
      {"issue": "<description>", "cause": "<cause_type>", "source_episode": "<optional episode id>"}
    ]
  },
  "edit_ops": [
    {"op_type": "<operation type>", "params": {<operation params>}, "root_cause": "<which root cause this fixes>"}
  ]
}
