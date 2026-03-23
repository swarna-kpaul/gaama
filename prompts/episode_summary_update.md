# Episode summary update

You have an existing episode summary and new conversation turns. Update the summary to incorporate the new conversation. Keep the result to 1–3 sentences. Focus on what was discussed, decided, or left open.

If there are clear outcomes (decisions, agreements, completed actions), list them briefly.
If there are unresolved questions or follow-ups, list them briefly.

## Output format (JSON only, no markdown)

```json
{
  "summary": "Updated one to three sentence summary incorporating previous and new content.",
  "outcomes": ["outcome 1", "outcome 2"],
  "unresolved_items": ["open question or follow-up 1"]
}
```

## Previous summary

{{previous_summary}}

## New conversation

{{new_conversation}}
