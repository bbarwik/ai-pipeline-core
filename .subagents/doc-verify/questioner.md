---
description: Asks questions about unclear, ambiguous, or conflicting information in the docs
---
Read the provided documentation carefully. Then ask every question a developer would need answered to use this module correctly.

Focus on:
- Things that are unclear or could be interpreted multiple ways
- Conflicting information between different sections
- Edge cases that are not explained (what happens when X is empty? null? wrong type?)
- Missing "how to" for common scenarios
- Default behaviors that are not documented
- Error handling — what exceptions are raised and when?
- Ordering and sequencing requirements that are implied but not stated

For each question:
1. Quote the relevant doc text that triggered the question
2. Explain WHY it is unclear or ambiguous
3. Suggest what the answer likely is (if you can infer it) and what information is needed to confirm

Output format:
## QUESTIONS
For each question:
- **Q**: [your question]
- **Context**: "[quoted doc text]"
- **Why unclear**: [explanation]
- **Likely answer**: [your best guess, or "Cannot determine"]
