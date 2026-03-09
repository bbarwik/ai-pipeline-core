---
description: Finds tests that cannot detect real bugs — wrong assertions, always-pass logic, duplicates, infrastructure flaws, and over-mocking
---
Read the provided test files and source code. Find tests that EXIST but provide no real assurance — a passing test that can't detect a bug is worse than no test because it creates false confidence.

Focus ONLY on the quality of existing tests. Do NOT report missing tests, missing edge cases, or missing validation coverage.

## WRONG_ASSERTIONS
Tests that assert incorrect or outdated behavior as if it were correct:
- Test name or comments suggest a bug exists (e.g., "proves_bug", "corrupted", "broken") but the bug has been fixed — the test now asserts the wrong thing
- Comments like "documents current behavior" on assertions that verify a known limitation rather than correct behavior
- Assertions that lock in error behavior that should be fixed, not preserved

For each: quote the assertion, explain what correct behavior should be, and why the current assertion is wrong.

## WEAK_ASSERTIONS
Tests with assertions too loose to catch real regressions:
- `assert result is not None` — passes for any non-None value, doesn't verify correctness
- `assert len(items) >= 1` or `> 0` — allows silent data loss (10 items → 1 item still passes)
- `isinstance(result, SomeType)` without checking the instance's actual values
- `assert "substring" in long_string` when the exact output structure matters

For each: quote the weak assertion and explain what real bug would slip through undetected.

## BRITTLE_ASSERTIONS
Tests with assertions too tight — they break on any legitimate change even when behavior is correct:
- Exact string equality on multi-line formatted output (e.g., `assert result == "long exact string..."`) — any format change, reordering, or new field breaks the test even though behavior is correct
- Hardcoded counts or sizes that depend on implementation details rather than behavioral contracts (e.g., asserting exact number of log lines, exact byte sizes)
- Tests that assert on internal representation rather than observable behavior

For each: quote the brittle assertion, explain what legitimate change would break it, and suggest a more resilient assertion.

## ALWAYS_PASS
Tests with logic errors that make them trivially true regardless of code behavior:
- `assert x or True`, `assert True`
- `try: code_under_test() except ExpectedException: pass` — test passes whether the exception fires or not
- Comparing a variable to itself
- Setup code that guarantees the assertion is trivially satisfied (e.g., asserting a mock returns what you told it to return)
- Tests that finish before the async behavior they're testing has time to fire (e.g., testing a heartbeat/timer without waiting for it)

For each: explain the logic flaw that makes the test always pass.

## OVER_MOCKED
Tests where mocking is so extensive that they verify mock behavior, not real code:
- More lines of `mock.patch` / `monkeypatch.setattr` setup than lines of meaningful assertions
- Mocks returning hardcoded values that skip the actual code path being "tested"
- Only assertion is `mock.assert_called_with(...)` — tests wiring/routing, not behavior
- Mock return values whose shape doesn't match the real API (mock returns `str` but real function returns `Document`)

For each: cite what's mocked vs what's real, and describe what refactoring would silently break production but pass this test.

## DUPLICATED_TESTS
Multiple tests that verify the same behavior with trivially different inputs and add no additional coverage:
- **Tests**: list of test function names
- **What they all verify**: the single concept being repeatedly tested
- **Keep**: the most representative test
- **Remove**: which ones add nothing new

## INFRASTRUCTURE_BUGS
Test infrastructure that causes false results:
- `asyncio.Event`, `asyncio.Lock`, or other async primitives initialized at module scope or class scope — these bind to the event loop active at creation time, which crashes or silently fails when a different loop runs the test
- `pytest.importorskip()` in `conftest.py` — silently skips the entire test directory if one optional dependency is missing, hiding real failures
- Fixtures with shared mutable state across tests (e.g., a module-level list that tests append to without cleanup)
- `monkeypatch.setattr` on module-level objects without ensuring teardown

For each: explain the failure mode and when it would produce a false pass or false fail.

## MISLEADING_NAMES
Tests whose function name contradicts what the test actually verifies:
- Name implies failure testing but test asserts success
- Name references a specific behavior but assertions check a completely different property
- Name implies a bug still exists but the code shows the bug was fixed

For each: cite the name, what it implies, what the test actually does, and suggest a better name.

## STALE_XFAIL
`@pytest.mark.xfail` markers on tests where the underlying bug may have already been fixed. `xfail` should only be used temporarily to prove a bug exists before fixing it. After the fix, the marker must be removed so the test actually runs.

Look for: xfail tests whose reason text describes something that appears to be fixed in the current source code.

## CANNOT_VERIFY
