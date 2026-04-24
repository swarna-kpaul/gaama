# GEL Analysis Question Generation (Online Mode)

You are diagnosing why a memory system produced an incomplete or vague answer. You do NOT have access to the correct answer — you must reason from the query, the generated answer, and the retrieved memory.

The memory system stores information as a knowledge graph with these node types:
- **Episodes**: Summaries of conversation segments (temporal, sequential)
- **Facts**: Extracted factual claims with belief scores
- **Concepts**: Topic labels that connect related facts and episodes
- **Reflections**: Higher-level insights synthesized from multiple facts

## The query
{{query}}

## Answer generated from memory
{{hypothesis}}

## What was retrieved from memory
{{retrieved_memory}}

## Your task
Analyze the generated answer critically:
- What does the query ask for that the answer does NOT provide?
- Where is the answer vague or hedging instead of giving specifics?
- What entities, events, dates, or details does the query mention that don't appear in the retrieved memory?
- Is the answer generic where it should be specific?

Then generate 1-{{max_questions}} analysis questions to probe the knowledge graph for the missing information. Each question should target a DIFFERENT knowledge gap. Prefer fewer, more targeted questions over many broad ones.

Focus on:
- Specific entities or topics the query mentions that the memory seems to lack coverage on
- Time periods or events that might contain the missing details
- Connections between known facts that could surface relevant information
- Episode content that might have relevant details not extracted as facts

Generate questions that are SPECIFIC to what's missing — not generic exploration.

You MUST respond in the following JSON format (no markdown fences, no extra text):
{
  "analysis_questions": [
    {
      "question": "<specific retrieval query to probe the graph>",
      "reasoning": "<what gap this targets and what you hope to find>"
    }
  ]
}
