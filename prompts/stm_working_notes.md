# Extract working notes from conversation

The following is a transcript of one or more conversation turns. **Participants may be humans or agents** (e.g. user and assistant, or agent-to-agent, or multi-party). Do not assume a single "user"; treat each speaker/actor as a participant whose statements can be remembered.

Extract **working notes**: key-value items that should be kept in short-term memory for context, continuity, or follow-up. Be comprehensive. Include any of the following when present:

**Preferences and settings**
- UI, theme, language, notification, or display preferences stated by any participant
- Communication style (e.g. concise, formal, technical level)
- Preferred tools, formats, or workflows

**Facts and context**
- Stated facts about people, organizations, projects, or the environment (e.g. "I work at X", "The API base URL is …", "Agent B handles billing")
- Roles, capabilities, or constraints of participants (human or agent)
- Current state, status, or phase of a task or process

**Commitments and intentions**
- Promises, commitments, or follow-ups (e.g. "I'll send the report by Friday", "Will retry after 5 min")
- Stated goals, next steps, or intentions of any participant

**Decisions and constraints**
- Decisions made in the conversation (e.g. "We're using approach A", "Scope limited to EU only")
- Constraints, rules, or boundaries mentioned (e.g. "No PII in logs", "Rate limit 100/min")
- Parameters or configuration chosen (e.g. model, timeout, batch size)

**Open items and follow-up**
- Open questions, unresolved topics, or "to be decided" items
- Blockers, dependencies, or waiting conditions
- Topics to revisit or pending confirmations

**Corrections and errors**
- Corrections to prior statements or misunderstandings
- Reported errors, failures, or retry conditions that affect context

**Identifiers and references**
- Important IDs, handles, or references (e.g. task_id, conversation_id, document version) that may be needed later
- Links between participants or resources (e.g. "this run is for job X")

**Actions taken**
- certain action taken in a context

Use **short, stable keys** in snake_case. Keys can be generic (e.g. `preference_theme`) or scoped (e.g. `agent_b_preference_theme`, `participant_deadline_report`) when the conversation involves multiple parties. Values can be strings, numbers, booleans, or short lists. Assign a **confidence** between 0.0 and 1.0 (1.0 = explicitly stated, lower if inferred or ambiguous).

## Output format (JSON only, no markdown)

Return a single JSON object with a key "notes" whose value is an array of objects:

```json
{
  "notes": [
    { "key": "preference_theme", "value": "dark", "confidence": 1.0 },
    { "key": "agent_b_capability", "value": "handles_billing", "confidence": 0.95 },
    { "key": "deadline_report", "value": "2025-02-15", "confidence": 0.9 },
    { "key": "decision_approach", "value": "approach_a", "confidence": 1.0 },
    { "key": "open_question_pricing", "value": "to be confirmed", "confidence": 0.8 }
  ]
}
```

If nothing should be remembered, return: `{"notes": []}`

## Conversation

{{conversation}}
