---
description: Identifies unnecessary, redundant, or overly detailed information that should be removed
---
Read the provided documentation carefully. Your goal is to make this documentation SMALLER without losing useful information. Report everything that should be removed or condensed.

Focus on:
- Implementation details that a USER of the API does not need (internal architecture, backend specifics, table schemas, storage internals)
- Duplicated information — same thing explained in multiple places
- Overly verbose explanations that could be shorter
- Private/internal APIs that are not needed to be known by end-user
- Configuration details for infrastructure the user doesn't control
- Information that belongs in a separate ops/admin guide, not an API reference
- Boilerplate text that adds no value

For each item:
1. Quote the text that should be removed or condensed
2. Explain WHY it is not needed for a developer using the API
3. If it should be condensed rather than removed, suggest the shorter version

Output format:
## REMOVE
- **Text**: "[quoted text]"
- **Why**: [reason it's not needed]

## CONDENSE
- **Text**: "[quoted text]"
- **Why**: [reason it's too verbose]
- **Suggested**: [shorter version]
