---
description: Finds over-engineering, unnecessary complexity, wrong abstractions, and performance anti-patterns
---
Audit the provided source files for complexity that exceeds what the requirements demand. Less code is better code — every line must justify its existence.

## OVER_ENGINEERING

- Abstraction layers that only delegate to another function/class without adding behavior (wrappers, facades, proxies)
- Classes where a plain function would suffice, or functions where inline code is clearer
- Generic type parameters always instantiated with the same concrete type
- "Just in case" parameters, feature branches, or configuration options with no internal callers
- Partially wired features — code for anticipated requirements that has some callers but no complete usage path or tests. (Note: symbols with ZERO callers are dead code, not bloat — bloat covers code that IS reachable but unnecessarily complex.)
- Files exceeding 500 lines of code (excluding blanks and comments) need splitting. Files exceeding 1000 lines are critical.

Suggested file splits for large files:
- Types/protocols into `_types.py` or `_protocols.py`
- Pure functions/utilities into `_utils.py`
- Constants/patterns into `_constants.py`

DO NOT flag these as over-engineering — they are deliberate patterns in typed frameworks:
- Generic typing on base classes designed for subclassing
- `__init_subclass__` validation chains that enforce constraints at class definition time
- Immutable builder patterns (methods returning new instances instead of mutating self)
- Content-addressing, hashing, or deduplication logic

## PERFORMANCE_ANTI_PATTERNS

- O(n²) or worse algorithms on unbounded input without a documented size bound (acceptable only when n is known small AND the simpler algorithm reduces code complexity)
- Database, network, or file I/O calls inside loops (N+1 pattern) — should be batched
- Object instantiation with expensive initialization inside tight loops (e.g., compiling regexes per iteration instead of at module level)
- Reading back from disk or network data that is already available in memory from a previous step

For each: cite the loop/call site AND the expensive operation inside it.

## REDUNDANT_OPERATIONS

Trace execution paths and find work that produces no useful result:

- Operations whose result is computed but never read or returned — wasted CPU/IO
- The same computation performed multiple times in the same code path when the result could be cached in a local variable
- Functions that unconditionally execute an expensive operation (DB call, file read, network request) that is only needed when a certain condition is true — the operation should be conditional or lazy

## DENSE_LOGIC

- Deeply nested conditionals (3+ levels) that obscure the main flow
- Complex f-string expressions or chained comprehensions that should be broken into named steps
- Functions exceeding 100 lines
- Logic where the "what" is clear but the "why" is impossible to determine without external knowledge — needs a comment explaining the reasoning

For each: quote the dense section, explain what makes it hard to follow, suggest a clearer structure.

## CANNOT_VERIFY
