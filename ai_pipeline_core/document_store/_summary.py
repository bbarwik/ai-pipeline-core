"""Summary generation types and constants for document stores."""

from collections.abc import Callable, Coroutine

from ai_pipeline_core.documents._types import DocumentSha256

type SummaryGenerator = Callable[[str, str], Coroutine[None, None, str]]
"""Async callable: (document_name, content_excerpt) -> summary string.
Returns empty string on failure. Must handle recursion prevention internally."""

type SummaryUpdateFn = Callable[[DocumentSha256, str], Coroutine[None, None, None]]
"""Async callable: (document_sha256, summary) -> None. Persists summary to store."""

SUMMARY_EXCERPT_CHARS: int = 5_000
