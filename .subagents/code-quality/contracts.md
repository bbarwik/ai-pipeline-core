---
description: Enforces architectural invariants — async correctness, immutability, serialization, export discipline, and protocol separation
---
Audit the provided source files against core architectural rules for async Python frameworks.

## ASYNC_CORRECTNESS

Every `async def` function must contain at least one `await`, `async for`, or `async with`. Functions without async operations must not be declared `async` — it wastes coroutine overhead and misleads readers.

Exceptions: Protocol method stubs, ABC base class methods meant for override, in-memory test implementations.

Blocking I/O inside async functions — these block the entire event loop:
- `open()` for file I/O (use `aiofiles` or run in executor)
- `time.sleep()` (use `asyncio.sleep()`)
- Synchronous HTTP clients (`requests.*`, `urllib`)
- Synchronous database drivers
- `os.path.*` on network filesystems
- `socket.getaddrinfo()` and other synchronous DNS

For each: cite the async function and the blocking call inside it.

## IMMUTABILITY

- Mutable module-level variables (`list`, `dict`, `set` that get modified at runtime) — only constants and initialized-once immutables are acceptable at module scope
- Pydantic models missing `model_config = ConfigDict(frozen=True)` without explicit justification
- Dataclasses missing `@dataclass(frozen=True, slots=True)`
- `copy.copy()` (shallow copy) on objects containing mutable internal state — must use `copy.deepcopy()` or explicitly copy mutable fields
- `model_copy()` overrides that forget to clear `__dict__` entries for `@cached_property` fields — the cached value becomes stale after the copy

For each: cite the mutable pattern and explain what could go wrong.

## SERIALIZATION

Non-JSON-serializable types used as inputs or outputs at framework boundaries (function parameters or return types on decorated pipeline functions):
- `frozenset`, `set`, `Callable`, `type` — not JSON-serializable
- `NewType` wrapping non-serializable types
- Custom objects without `__json__` or Pydantic serialization

For each: cite the parameter/return type and suggest the serializable alternative.

## EXPORTS

- Public modules (files without `_` prefix) missing `__all__` definition — public API must be explicit
- `__all__` listing internal `_`-prefixed symbols that should not be exported
- Internal `_`-prefixed symbols imported and re-exported from other modules without a public facade
- Private symbols (`_ClassName`, `_function`) imported across module boundaries — if another module needs it, it should be public or the boundary is wrong

For each: cite the file and the problematic export/import.

## PROTOCOL_SEPARATION

Protocol (or ABC) definitions must not be mixed with their concrete implementations in the same file. Protocols define interfaces — implementations should be in separate files.

Exceptions: files specifically named `_protocol.py`, `_types.py`, or `base.py` (these are conventional locations for protocols alongside minimal base implementations).

## CROSS_MODULE_CONSISTENCY

When multiple files are in your context, reason about whether modules that interact agree on their contract:

- **Path/format mismatch**: if one module writes data to a path or in a format, and another module reads it, verify they use the same path construction and format parsing. A mismatch means data is written but never found, or found but unparsable.
- **Data model completeness**: if a parent-child or producer-consumer relationship exists, verify both sides maintain the reference. If a parent creates children but stores no reference to them, or children store no reference back, the relationship is unqueryable.
- **Schema agreement**: if one module generates a schema/DDL/config and another module queries against it, verify the generated structure supports the queries (e.g., columns exist, indexes cover filter conditions).

For each: cite both sides of the contract (producer and consumer) and explain the mismatch.

## CANNOT_VERIFY
