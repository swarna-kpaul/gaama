# Extract structured long-term memory from conversation

Analyze the conversation and extract a typed knowledge graph: **entities**, **facts**, **episodes**, **reflections**, and **skills**.

---

## Node types to extract

### 1. Entities
Any distinct person, place, organization, concept, tool, preference, topic, app, or configuration mentioned. 

Each entity has:
- **name** (required): concise, unambiguous label.
- **aliases** (optional): alternative names.

### 2. Facts
An atomic factual assertion about entities (e.g. "user prefers dark mode", "main stack is TypeScript"). Dont consider events, interactions as facts, rather they are episodes. 
**most of the user interactions should be considered as episodes. Only if some general knowledge that can be derived as applied in general cases should be considered as fact**
In the episodic data where temporal events are vaguely defined like "1 week before 1st anniversery", derive the exact info possibly with date as fact.  connect multiple episodes to derive general facts by doing multistep reasoning wherever possible.

Each fact has:
- **fact_text** (required): the complete fact statement in natural language.
- **belief** (0.0–1.0): confidence the fact is true. 1.0 = explicitly stated. 0.8 = implied.
- **polarity** (boolean): true if the fact is affirmative, false if negated.
- **subject**: index into the entities array (the entity the fact is *about*).
- **object**: index into the entities array (the entity the fact *references*).
- **supported_by_episodes** (optional): list of episode indices (evidence from which episodes).
- **contradicts_facts** (optional): list of fact indices this fact contradicts.
- **refines_facts** (optional): list of fact indices this fact refines or updates.

### 3. Episodes
**All raw conversations are considered episodes.**

#### CRITICAL: Atomic decomposition & full coverage
- **Every single line of conversation history must be represented as is as episode.** No part of any chat message should be modified or skipped.
- **If a single chat message contains multiple distinct pieces of information** (e.g., multiple facts, events, decisions, names, dates, locations, preferences, actions, or any other details), it MUST be decomposed into **multiple separate episode records** — one per atomic piece of information. Never merge unrelated details into a single episode. However **dont miss any words that appear in conversation**
- **The summary of each episode must preserve the exact words from the original chat message.** Do not paraphrase, generalize, or abstract away specifics. Include all concrete details: names, numbers, dates, times, locations, products, systems, amounts, reasons, outcomes, and any other specifics mentioned. 
- **Completeness check**: After extraction, the union of all episode summaries should allow someone to reconstruct the full chat history without loss of any word. If any words from the conversation is missing from the episodes, go back and add it.

Each episode has:
- **sequence** (required): integer order of occurrence in the conversation, starting from 1. List episodes in chronological order and assign 1 to the first, 2 to the second, and so on. When a single chat message is split into multiple episodes, assign consecutive sequence numbers and maintain chronological order.
- **summary** (required): a single sentence representing each conversation information without missing any words that appears in the conversation
- **outcome**: outcome in short
- **belief** (0.0–1.0): confidence the episode is true. 1.0 = explicitly stated. 0.8 = implied. in case of more ambuigity you can reduce the belief.
- **involves**: list of entity indices (active participants).
- **mentions**: list of entity indices (passively referenced).
- **produced_facts** (optional): list of fact indices — facts that were extracted from this episode.
- **triggered_by** (optional): entity index of the task or actor that triggered this episode.
- **next_episode** (mandatory): episode index of the immediately following episode (temporal chain) unless it is the last episode. In case its last episode put next episode as -1

### 4. Reflections
A generalized insight, preference, or lesson (e.g. "confidence should be shown with each retrieved memory").

Each reflection has:
- **reflection_text** (required): the insight in natural language.
- **belief** (0.0–1.0): confidence in the reflection.
- **about**: list of entity indices the reflection concerns.
- **supported_by_episodes** (optional): list of episode indices that provide evidence.
- **derived_from_facts** (optional): list of fact indices the reflection is based on.
- **contradicts_reflections** (optional): list of reflection indices this reflection contradicts.
- **refines_reflections** (optional): list of reflection indices this reflection refines.

### 5. Skills
A reusable procedure or know-how (e.g. "debug retrieval: check filters → candidates → ranker").

Each skill has:
- **name** (required): short label.
- **skill_description** (required): step-by-step instructions.
- **belief** (0.0–1.0): confidence the procedure is reliable.
- **uses_tool** (optional): list of entity indices — tools or libraries the skill uses.
- **applies_to** (optional): list of entity indices — task types or domain concepts the skill applies to.
- **learned_from_episodes** (optional): list of episode indices the skill was learned from.
- **refines_skills** (optional): list of skill indices this skill refines (versioning).
- **contradicts_skills** (optional): list of skill indices this skill contradicts.

---

## Index referencing

- **Entity references** (subject, object, involves, mentions, triggered_by, uses_tool, applies_to, about): zero-based indices into the `entities` array.
- **Fact references** (produced_facts, contradicts_facts, refines_facts, derived_from_facts): zero-based indices into the `facts` array.
- **Episode references** (supported_by_episodes, learned_from_episodes, next_episode): zero-based indices into the `episodes` array.
- **Reflection references** (contradicts_reflections, refines_reflections): zero-based indices into the `reflections` array.
- **Skill references** (refines_skills, contradicts_skills): zero-based indices into the `skills` array.

All cross-type references are optional. Only include them when a clear relationship exists.

---

## Output format (JSON only, no markdown fences)

Return a single JSON object:

```json
{
  "entities": [
    {"name": "user", "aliases": []},
    {"name": "dark mode", "aliases": []},
    {"name": "SQLite", "aliases": ["sqlite3"]}
  ],
  "facts": [
    {"fact_text": "user prefers dark mode", "belief": 0.95, "polarity": true, "subject": 0, "object": 1, "supported_by_episodes": [0]}
  ],
  "episodes": [
    {"sequence": 1, "summary": "User configured a local SQLite database at /data/app.db on 2024-01-15", "outcome": "database file created", "belief": 1.0, "involves": [0], "mentions": [2], "produced_facts": [], "triggered_by": 0, "next_episode": 1},
    {"sequence": 2, "summary": "User created the 'users' table with columns id, name, and email in the SQLite database", "outcome": "table created", "belief": 1.0, "involves": [0], "mentions": [2], "produced_facts": [0], "triggered_by": 0, "next_episode": 2},
    {"sequence": 3, "summary": "User created the 'sessions' table with columns id, user_id, and token in the SQLite database", "outcome": "table created", "belief": 1.0, "involves": [0], "mentions": [2], "produced_facts": [], "triggered_by": 0, "next_episode": -1}
  ],
  "reflections": [
    {"reflection_text": "always verify table schema after migration", "belief": 0.6, "about": [2], "supported_by_episodes": [0], "derived_from_facts": [0]}
  ],
  "skills": [
    {"name": "Debug retrieval pipeline", "skill_description": "check filters → check candidates → check ranker", "belief": 0.7, "uses_tool": [2], "applies_to": []}
  ]
}
```

If nothing meaningful to extract for a category, use an empty array. Omit optional cross-type reference fields when no relationship exists. Do **not** add markdown code fences around the JSON.

---

# DO NOT MISS CAPTURING ANY DETAILS FROM CONVERSATION HISTORY

** IMPORTANT **
# DO NOT MISS CAPTURING ANY EPISODIC INFORMATION LIKE DATE, TIME, LOCATION, EVENT, PRODUCT, SYSTEM, ACTION, OUTCOMES, PEOPLE, NAMES

# NEVER OMIT ANY DETAILS IN CONVERSATION EVEN IF IT SEEMS TO BE MINOR, IT SHOULD BE CAPTURED EITHER AS EPISODES, FACTS, SKILLS OR REFLECTIONS

# EPISODE EXTRACTION RULES — READ CAREFULLY
1. Go through the conversation line by line. Every chat message must produce at least one episode.
2. If a message contains multiple pieces of information (e.g., "I went to Paris on Monday and bought a laptop for $1200"), split it into separate episodes: one for going to Paris on Monday, one for buying a laptop for $1200.
3. Episode summaries must contain the EXACT details (words) from the chat — names, dates, numbers, locations, products, reasons, outcomes. Do NOT summarize or generalize.
4. After generating all episodes, re-read the entire conversation and verify that every piece of information is covered. Add any missing episodes.

## Conversation

{{conversation}}
