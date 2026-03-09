# pyright: reportPrivateUsage=false
"""Tests for task resilience: retries, timeouts, cancellation, caching, and TaskHandle semantics."""

import asyncio

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------


class EdgeInputDoc(Document):
    """Input for edge-case tests."""


class EdgeOutputDoc(Document):
    """Output for edge-case tests."""


# ---------------------------------------------------------------------------
# CancelledError with retries > 0
# ---------------------------------------------------------------------------


class _CancelOnFirstAttempt(PipelineTask):
    retries = 2
    retry_delay_seconds = 0

    attempt_count = 0

    @classmethod
    async def run(cls, documents: tuple[EdgeInputDoc, ...]) -> tuple[EdgeOutputDoc, ...]:
        cls.attempt_count += 1
        raise asyncio.CancelledError("externally cancelled")


@pytest.mark.asyncio
async def test_cancelled_error_bypasses_retries() -> None:
    """CancelledError propagates immediately even with retries > 0."""
    _CancelOnFirstAttempt.attempt_count = 0
    doc = EdgeInputDoc.create_root(name="in.txt", content="x", reason="gap8")
    with pipeline_test_context():
        with pytest.raises(asyncio.CancelledError):
            await _CancelOnFirstAttempt.run((doc,))

    # CancelledError should not be retried
    assert _CancelOnFirstAttempt.attempt_count == 1


class _CancelDuringRetryDelay(PipelineTask):
    retries = 2
    retry_delay_seconds = 60  # long delay — cancel should interrupt it

    attempt_count = 0

    @classmethod
    async def run(cls, documents: tuple[EdgeInputDoc, ...]) -> tuple[EdgeOutputDoc, ...]:
        cls.attempt_count += 1
        raise RuntimeError("retriable failure")


@pytest.mark.asyncio
async def test_cancel_during_retry_sleep_propagates() -> None:
    """Cancellation during retry sleep propagates CancelledError."""
    _CancelDuringRetryDelay.attempt_count = 0
    doc = EdgeInputDoc.create_root(name="in.txt", content="x", reason="gap8b")

    with pipeline_test_context():
        handle = _CancelDuringRetryDelay.run((doc,))  # returns TaskHandle immediately
        # Give the task time to fail once and enter retry sleep
        await asyncio.sleep(0.2)
        handle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handle

    # First attempt ran but not the retry (cancelled during sleep)
    assert _CancelDuringRetryDelay.attempt_count == 1


# ---------------------------------------------------------------------------
# Timeout + retry interaction
# ---------------------------------------------------------------------------


class _TimeoutWithRetry(PipelineTask):
    retries = 1
    retry_delay_seconds = 0
    timeout_seconds = 1  # 1 second timeout

    attempt_count = 0

    @classmethod
    async def run(cls, documents: tuple[EdgeInputDoc, ...]) -> tuple[EdgeOutputDoc, ...]:
        cls.attempt_count += 1
        if cls.attempt_count <= 1:
            await asyncio.sleep(10)  # exceeds timeout
        return (EdgeOutputDoc.derive(from_documents=(documents[0],), name="out.txt", content="ok"),)


@pytest.mark.asyncio
async def test_timeout_triggers_retry_then_succeeds() -> None:
    """TimeoutError on first attempt triggers retry; second attempt succeeds."""
    _TimeoutWithRetry.attempt_count = 0
    doc = EdgeInputDoc.create_root(name="in.txt", content="x", reason="gap9")
    with pipeline_test_context():
        result = await _TimeoutWithRetry.run((doc,))

    assert _TimeoutWithRetry.attempt_count == 2
    assert len(result) == 1
    assert result[0].name == "out.txt"


class _TimeoutAllRetries(PipelineTask):
    retries = 1
    retry_delay_seconds = 0
    timeout_seconds = 1

    attempt_count = 0

    @classmethod
    async def run(cls, documents: tuple[EdgeInputDoc, ...]) -> tuple[EdgeOutputDoc, ...]:
        cls.attempt_count += 1
        await asyncio.sleep(10)
        return ()


@pytest.mark.asyncio
async def test_timeout_exhausts_retries() -> None:
    """Timeout on every attempt raises TimeoutError after all retries exhausted."""
    _TimeoutAllRetries.attempt_count = 0
    doc = EdgeInputDoc.create_root(name="in.txt", content="x", reason="gap9b")
    with pipeline_test_context():
        with pytest.raises(TimeoutError):
            await _TimeoutAllRetries.run((doc,))

    assert _TimeoutAllRetries.attempt_count == 2


# ---------------------------------------------------------------------------
# TaskHandle concurrent await / repeated await
# ---------------------------------------------------------------------------


class _InstantTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[EdgeInputDoc, ...]) -> tuple[EdgeOutputDoc, ...]:
        return (EdgeOutputDoc.derive(from_documents=(documents[0],), name="instant.txt", content="done"),)


@pytest.mark.asyncio
async def test_handle_can_be_awaited_twice() -> None:
    """Awaiting a completed TaskHandle a second time returns the same result."""
    doc = EdgeInputDoc.create_root(name="in.txt", content="x", reason="gap15")
    with pipeline_test_context():
        handle = _InstantTask.run((doc,))  # returns TaskHandle, not yet awaited
        result1 = await handle
        result2 = await handle  # second await on same handle
    assert result1[0].sha256 == result2[0].sha256


@pytest.mark.asyncio
async def test_handle_concurrent_await() -> None:
    """Multiple coroutines can await the same TaskHandle concurrently."""
    doc = EdgeInputDoc.create_root(name="in.txt", content="x", reason="gap15b")
    with pipeline_test_context():
        handle = _InstantTask.run((doc,))
        results = await asyncio.gather(handle.result(), handle.result())
    assert results[0][0].sha256 == results[1][0].sha256


# ---------------------------------------------------------------------------
# Forward references in Task/Flow annotations
# ---------------------------------------------------------------------------


def test_forward_reference_task_annotation() -> None:
    """PipelineTask with forward-reference Document type resolves correctly."""

    class LateDoc(Document):
        pass

    class ForwardRefTask(PipelineTask):
        @classmethod
        async def run(cls, documents: tuple[LateDoc, ...]) -> tuple[LateDoc, ...]:
            return documents

    assert ForwardRefTask._run_spec.input_document_types == (LateDoc,)
    assert ForwardRefTask._run_spec.output_document_types == (LateDoc,)
