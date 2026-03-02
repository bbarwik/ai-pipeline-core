---
description: Analyzes every example and error example in the docs — finds duplicates, useless tests, missing examples, and unclear patterns
---
Read the provided documentation carefully. Analyze EVERY test/example in the `## Examples` and `## Error Examples` sections. Your goal is to determine whether the examples actually help a developer learn and use this API correctly.

For each example, evaluate:
- Does it teach something NEW that no other example covers?
- Does it demonstrate a PUBLIC API that a developer would actually call?
- Can a developer understand the pattern without knowing test infrastructure (fixtures, mocks, monkeypatches)?
- Does it show realistic usage or just assert an implementation detail?

## DUPLICATES
Examples that test the same concept with trivially different inputs. For each group:
- **Examples**: [list the example titles]
- **What they all test**: [the single concept]
- **Keep**: [which one, if any, is the best representative]
- **Drop**: [which ones add nothing]

## USELESS
Examples that teach a developer nothing about how to use the API. Common signs:
- Tests for `__slots__`, `hasattr`, or attribute absence (migration/refactoring artifacts)
- Tests that only verify `isinstance` or protocol satisfaction
- Tests asserting trivially obvious behavior (e.g., "can be caught as Exception")
- Tests for internal/private symbols (prefixed with `_`) that users cannot call
- Tests that are pure implementation verification with no usage insight
For each:
- **Example**: [title]
- **Why useless**: [what's wrong with it]

## UNCLEAR
Examples that ARE useful but are hard to understand without additional context. For each:
- **Example**: [title]
- **Problem**: what makes it confusing (heavy mocking? unexplained fixtures? missing setup? unclear what's being demonstrated?)
- **What would help**: how the example could be improved (remove mock noise? add a comment? show the setup?)

## MISSING
Essential usage patterns that have NO example at all. Think about what a developer needs to do with this module:
- What is the most basic "hello world" usage? Is it shown?
- What are the 3-5 most common operations? Are they each demonstrated?
- Are there builder/fluent patterns that need a complete chain example?
- Are there async patterns that need proper await/gather demonstration?
- Are there integration points with other modules that need examples?
For each missing example:
- **Pattern**: [what should be demonstrated]
- **Why important**: [what a developer can't figure out without it]
- **Sketch**: [2-3 line pseudocode showing what the example should look like]

## RANKING
Rank all examples from most useful to least useful. For each:
- **#N**: [title] — [one sentence: why it's at this rank]

## VERDICT
Rate the example quality for this guide: EXCELLENT / GOOD / MEDIOCRE / POOR.
State: how many examples are genuinely useful vs total, and what's the single biggest gap.
