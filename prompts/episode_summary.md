# Episode summarization

Summarize the following conversation episode in 1–3 sentences. Focus on what was discussed, decided, or left open.

If there are clear outcomes (decisions, agreements, completed actions), list them briefly.
If there are unresolved questions or follow-ups, list them briefly.

## Output format (JSON only, no markdown)

```json
{
  "summary": "One to three sentence summary of the conversation.",
  "outcomes": ["outcome 1", "outcome 2"],
  "unresolved_items": ["open question or follow-up 1"]
}
```

## Conversation

{{conversation}}
