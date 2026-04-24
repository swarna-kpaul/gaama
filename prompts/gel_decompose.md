# GEL Analysis Question Generation

You are diagnosing why a memory system failed to answer a query correctly. The memory system stores information as a knowledge graph with these node types:
- **Episodes**: Summaries of conversation segments (temporal, sequential)
- **Facts**: Extracted factual claims with belief scores
- **Concepts**: Topic labels that connect related facts and episodes
- **Reflections**: Higher-level insights synthesized from multiple facts

## The failed query
{{query}}

## Expected answer
{{ground_truth}}

## Wrong/incomplete answer generated from memory
{{hypothesis}}

## What was retrieved from memory
{{retrieved_memory}}

## Your task
Generate 1-{{max_questions}} analysis questions that will help us explore the knowledge graph to understand what went wrong. Each question should retrieve different information from the graph. Prefer fewer, more targeted questions over many broad ones.

Think about:
- What specific facts might be missing from the graph?
- What time periods or events might contain the missing information?
- What connections between topics might be broken or missing?
- What higher-level patterns should have been synthesized as reflections?
- What irrelevant information might be crowding out useful results?

Generate questions that are SPECIFIC to this failure -- not generic. Each question should target a different aspect of the problem.

For each question, explain your reasoning: what you expect to find (or not find) and how it will help diagnose the graph issue.

You MUST respond in the following JSON format (no markdown fences, no extra text):
{
  "analysis_questions": [
    {
      "question": "<specific retrieval query to probe the graph>",
      "reasoning": "<what you expect to find/not find and why it matters>"
    }
  ]
}
