---
description: Reasons about WHY the API is designed this way — engineering consequences, not restated docstrings
---
Read the provided documentation. For every rule, constraint, class, method, and function, provide your REASONING about why it exists and what would break without it. Do NOT restate or paraphrase the documentation.

For each rule or constraint:
- Do NOT repeat what the docs say. We already have the docs.
- Instead explain: what engineering problem does this solve? What would go wrong in a real system if this rule didn't exist? Give a concrete failure scenario.
- Example of BAD output: "Flows cannot be empty. Why: a deployment needs at least one step."
- Example of GOOD output: "If flows were empty, run() would execute zero steps, build_result() would receive an empty document list, and the webhook would report 100% progress immediately — producing a silent no-op that looks like success to monitoring."

For each class, method, or function:
- Do NOT rewrite its docstring. We already have it.
- Instead explain: what would happen if this was removed? What problem would a developer face? What alternative design was likely considered and why this approach was chosen instead?

For symbols whose purpose you CANNOT reason about from the docs alone, say so — that means the docs failed to communicate the "why."

Output format:

## REASONING
For each item:
- **Symbol**: [name]
- **If removed**: [concrete consequence — what breaks, what fails silently, what becomes impossible]
- **Design choice**: [why this approach over alternatives, if you can infer it]

## CANNOT_REASON
- **Symbol**: [name]
- **Problem**: [what the docs don't explain that would be needed to understand the why]
