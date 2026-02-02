"""Summary generation types and constants for document stores."""

from collections.abc import Callable, Coroutine

type SummaryGenerator = Callable[[str, str], Coroutine[None, None, str]]
"""Async callable: (document_name, content_excerpt) -> summary string.
Returns empty string on failure. Must handle recursion prevention internally."""

SUMMARY_EXCERPT_CHARS: int = 5_000
