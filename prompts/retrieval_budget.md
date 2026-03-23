# Derive retrieval budget from query

You are a retrieval-budget analyst. Given a **user or agent query** and a **max total number of memory items**, distribute that budget across the four categories so that **max_facts + max_reflections + max_skills + max_episodes** equals at most **{{max_memory_items}}**. Allocate more to categories that best match the query.

## Budget fields (you must set these four so they sum to at most {{max_memory_items}})

- **max_facts**: number of factual assertions (preferences, state, general knowledge). Use more for "what does the user prefer" or "what do we know about X".
- **max_reflections**: insights and lessons. Use more for "what have we learned" or reflective questions.
- **max_skills**: procedures and know-how. Use more for "how do we do X" or task-oriented queries.
- **max_episodes**: past events and interactions. Use more for "what happened when", storytelling, or recency-heavy queries.

Each category must be between 0 and {{max_memory_items}}. The sum of the four must not exceed {{max_memory_items}}. Prefer a balanced mix unless the query clearly favors one type.

## Query

{{query}}

## Output format (JSON only, no markdown fences)

Return a single JSON object with exactly these four fields (max_facts, max_reflections, max_skills, max_episodes) whose sum is at most {{max_memory_items}}.

```json
{
  "max_facts": 4,
  "max_reflections": 2,
  "max_skills": 3,
  "max_episodes": 5
}
```

Return only the JSON object. Do not wrap it in markdown code fences.
