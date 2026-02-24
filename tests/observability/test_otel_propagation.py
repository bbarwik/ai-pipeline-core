"""Tests for OpenTelemetry context propagation to background summary threads."""

import asyncio
import time
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest

clickhouse_connect = pytest.importorskip("clickhouse_connect")

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.sdk.trace import Tracer

from ai_pipeline_core.document_store._summary_worker import SummaryWorker, _SENTINEL, _SummaryItem
from ai_pipeline_core.documents.document import Document

type OtelEnv = tuple[Tracer, InMemorySpanExporter, TracerProvider]


class _TestDocument(Document):
    """Concrete Document subclass for testing."""


@pytest.fixture
def otel_env() -> Generator[OtelEnv]:
    """Isolated OTel TracerProvider with in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    yield tracer, exporter, provider  # type: ignore[misc]
    provider.shutdown()


class TestSummaryWorkerContextCapture:
    """Verify SummaryWorker.schedule() captures the active OTel context."""

    @pytest.mark.asyncio
    async def test_schedule_captures_context(self, otel_env: OtelEnv) -> None:
        tracer, _, _ = otel_env
        worker = SummaryWorker(generator=AsyncMock(return_value=""), update_fn=AsyncMock())
        worker._loop = asyncio.get_running_loop()
        worker._queue = asyncio.Queue()

        doc = _TestDocument(name="test.txt", content=b"x" * 2000)

        with tracer.start_as_current_span("save_task") as parent_span:
            parent_span_id = parent_span.get_span_context().span_id
            worker.schedule(doc)

        # Let the call_soon_threadsafe callback execute
        await asyncio.sleep(0)
        item = worker._queue.get_nowait()
        assert isinstance(item, _SummaryItem)
        assert item.parent_otel_context is not None
        span_from_ctx = trace.get_current_span(item.parent_otel_context)
        assert span_from_ctx.get_span_context().span_id == parent_span_id


class TestSummaryWorkerRunContext:
    """Verify SummaryWorker._run() attaches/detaches OTel context."""

    @pytest.mark.asyncio
    async def test_generator_sees_parent_span(self, otel_env: OtelEnv) -> None:
        tracer, _, _ = otel_env
        captured_trace_id = None

        async def capturing_gen(name: str, excerpt: str) -> str:
            nonlocal captured_trace_id
            captured_trace_id = trace.get_current_span().get_span_context().trace_id
            return "summary"

        worker = SummaryWorker(generator=capturing_gen, update_fn=AsyncMock())
        worker._queue = asyncio.Queue()

        with tracer.start_as_current_span("origin_task") as span:
            origin_trace_id = span.get_span_context().trace_id
            ctx = otel_context.get_current()

        item = _SummaryItem(sha256="h", name="n", excerpt="e", parent_otel_context=ctx)
        await worker._queue.put(item)
        await worker._queue.put(_SENTINEL)

        await worker._run()

        assert captured_trace_id == origin_trace_id

    @pytest.mark.asyncio
    async def test_context_not_leaked_between_jobs(self, otel_env: OtelEnv) -> None:
        tracer, _, _ = otel_env
        captured_trace_ids: list[int | None] = []

        async def capturing_gen(name: str, excerpt: str) -> str:
            ctx = trace.get_current_span().get_span_context()
            captured_trace_ids.append(ctx.trace_id if ctx.is_valid else None)
            return "summary"

        worker = SummaryWorker(generator=capturing_gen, update_fn=AsyncMock())
        worker._queue = asyncio.Queue()

        # Job 1: with context
        with tracer.start_as_current_span("task_A") as span_a:
            trace_id_a = span_a.get_span_context().trace_id
            ctx_a = otel_context.get_current()
        item_a = _SummaryItem(sha256="h1", name="a", excerpt="e", parent_otel_context=ctx_a)

        # Job 2: without context
        item_b = _SummaryItem(sha256="h2", name="b", excerpt="e", parent_otel_context=None)

        await worker._queue.put(item_a)
        await worker._queue.put(item_b)
        await worker._queue.put(_SENTINEL)

        await worker._run()

        assert len(captured_trace_ids) == 2
        assert captured_trace_ids[0] == trace_id_a
        assert captured_trace_ids[1] is None  # No leaked context from job 1

    @pytest.mark.asyncio
    async def test_context_restored_after_generator_failure(self, otel_env: OtelEnv) -> None:
        tracer, _, _ = otel_env

        async def failing_gen(name: str, excerpt: str) -> str:
            raise RuntimeError("boom")

        worker = SummaryWorker(generator=failing_gen, update_fn=AsyncMock())
        worker._queue = asyncio.Queue()

        with tracer.start_as_current_span("task"):
            ctx = otel_context.get_current()

        item = _SummaryItem(sha256="h", name="n", excerpt="e", parent_otel_context=ctx)
        await worker._queue.put(item)
        await worker._queue.put(_SENTINEL)

        span_before = trace.get_current_span()
        await worker._run()  # should not raise
        span_after = trace.get_current_span()

        assert span_before.get_span_context().span_id == span_after.get_span_context().span_id


class TestSummaryWorkerEndToEnd:
    """End-to-end: span created in worker thread has correct parent."""

    def test_child_span_has_correct_parent(self, otel_env: OtelEnv) -> None:
        tracer, exporter, provider = otel_env

        async def gen_with_span(name: str, excerpt: str) -> str:
            with tracer.start_as_current_span("doc_summary_llm"):
                pass
            return "summary"

        worker = SummaryWorker(generator=gen_with_span, update_fn=AsyncMock())
        worker.start()

        try:
            doc = _TestDocument(name="test.txt", content=b"x" * 2000)
            with tracer.start_as_current_span("pipeline_save") as parent_span:
                parent_trace_id = parent_span.get_span_context().trace_id
                parent_span_id = parent_span.get_span_context().span_id
                worker.schedule(doc)

            for _ in range(50):
                time.sleep(0.1)
                spans = exporter.get_finished_spans()
                if any(s.name == "doc_summary_llm" for s in spans):
                    break
        finally:
            worker.shutdown(timeout=5.0)

        spans = exporter.get_finished_spans()
        summary_spans = [s for s in spans if s.name == "doc_summary_llm"]
        assert len(summary_spans) == 1
        child = summary_spans[0]
        assert child.context is not None
        assert child.context.trace_id == parent_trace_id
        assert child.parent is not None
        assert child.parent.span_id == parent_span_id


class TestDataModelFieldConfig:
    """Verify parent_otel_context is excluded from hash and equality."""

    def test_summary_item_context_excluded_from_hash(self) -> None:
        item1 = _SummaryItem(sha256="h", name="n", excerpt="e", parent_otel_context=None)
        item2 = _SummaryItem(sha256="h", name="n", excerpt="e", parent_otel_context=otel_context.get_current())
        assert hash(item1) == hash(item2)

    def test_summary_item_context_excluded_from_equality(self) -> None:
        item1 = _SummaryItem(sha256="h", name="n", excerpt="e", parent_otel_context=None)
        item2 = _SummaryItem(sha256="h", name="n", excerpt="e", parent_otel_context=otel_context.get_current())
        assert item1 == item2
