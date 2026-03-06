"""Unit tests for ai_pipeline_core.deployment._cli — debug tracing and CLI wiring."""

# pyright: reportPrivateUsage=false

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_pipeline_core.deployment._cli import _init_debug_tracing
from ai_pipeline_core.deployment._types import ErrorCode, _NoopPublisher
from ai_pipeline_core.deployment.base import _classify_error, _create_publisher, _build_summary_generator, _validate_flow_chain
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import PipelineFlow
from ai_pipeline_core.documents._context import _suppress_document_registration
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError
from ai_pipeline_core.pipeline.options import FlowOptions
from ai_pipeline_core.settings import Settings


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield


# ---------------------------------------------------------------------------
# _classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_llm_error(self):
        assert _classify_error(LLMError("fail")) == ErrorCode.PROVIDER_ERROR

    def test_cancelled(self):
        assert _classify_error(asyncio.CancelledError()) == ErrorCode.CANCELLED

    def test_timeout(self):
        assert _classify_error(TimeoutError()) == ErrorCode.DURATION_EXCEEDED

    def test_value_error(self):
        assert _classify_error(ValueError("bad")) == ErrorCode.INVALID_INPUT

    def test_type_error(self):
        assert _classify_error(TypeError("wrong type")) == ErrorCode.INVALID_INPUT

    def test_pipeline_error(self):
        assert _classify_error(PipelineCoreError("pipe")) == ErrorCode.PIPELINE_ERROR

    def test_unknown(self):
        assert _classify_error(RuntimeError("unknown")) == ErrorCode.UNKNOWN


# ---------------------------------------------------------------------------
# _create_publisher
# ---------------------------------------------------------------------------


class TestCreatePublisher:
    def test_noop_when_no_pubsub(self):
        s = Settings(pubsub_project_id="", pubsub_topic_id="")
        publisher = _create_publisher(s, "research")
        assert isinstance(publisher, _NoopPublisher)

    def test_noop_when_no_service_type(self):
        s = Settings(pubsub_project_id="proj", pubsub_topic_id="topic")
        publisher = _create_publisher(s, "")
        assert isinstance(publisher, _NoopPublisher)

    def test_pubsub_without_clickhouse_creates_publisher(self):
        """Pub/Sub no longer requires ClickHouse — PubSubPublisher is pure event transport."""
        s = Settings(
            pubsub_project_id="proj",
            pubsub_topic_id="topic",
            clickhouse_host="",
        )
        with patch("ai_pipeline_core.deployment._pubsub.PubSubPublisher") as mock_cls:
            mock_cls.return_value = MagicMock()
            publisher = _create_publisher(s, "svc")
            mock_cls.assert_called_once()
            assert publisher is mock_cls.return_value


# ---------------------------------------------------------------------------
# _build_summary_generator
# ---------------------------------------------------------------------------


class TestBuildSummaryGenerator:
    @patch("ai_pipeline_core.deployment._helpers.settings")
    def test_disabled_returns_none(self, mock_settings):
        mock_settings.doc_summary_enabled = False
        result = _build_summary_generator()
        assert result is None

    @patch("ai_pipeline_core.deployment._helpers.settings")
    def test_enabled_returns_callable(self, mock_settings):
        mock_settings.doc_summary_enabled = True
        mock_settings.doc_summary_model = "gemini-3-flash"
        result = _build_summary_generator()
        assert result is not None
        assert callable(result)


# ---------------------------------------------------------------------------
# _init_debug_tracing
# ---------------------------------------------------------------------------


class TestInitDebugTracing:
    def test_creates_trace_dir(self, tmp_path: Path):
        with patch("ai_pipeline_core.deployment._cli.otel_trace") as mock_otel:
            provider = MagicMock()
            provider.add_span_processor = MagicMock()
            mock_otel.get_tracer_provider.return_value = provider
            processor = _init_debug_tracing(tmp_path)
        assert processor is not None
        assert (tmp_path / ".trace").is_dir()
        provider.add_span_processor.assert_called_once()

    def test_returns_none_on_os_error(self, tmp_path: Path):
        with patch("ai_pipeline_core.deployment._cli.TraceDebugConfig", side_effect=OSError("fail")):
            result = _init_debug_tracing(tmp_path)
        assert result is None

    def test_no_add_span_processor(self, tmp_path: Path):
        with patch("ai_pipeline_core.deployment._cli.otel_trace") as mock_otel:
            provider = MagicMock(spec=[])
            mock_otel.get_tracer_provider.return_value = provider
            processor = _init_debug_tracing(tmp_path)
        assert processor is not None


# ---------------------------------------------------------------------------
# _compute_run_scope
# ---------------------------------------------------------------------------

from ai_pipeline_core.deployment.base import _compute_run_scope, _heartbeat_loop


class ScopeDoc(Document):
    pass


class TestComputeRunScope:
    def test_no_documents(self):
        opts = FlowOptions()
        scope = _compute_run_scope("run1", [], opts)
        assert str(scope).startswith("run1:")

    def test_with_documents(self):
        doc = ScopeDoc.create_root(name="test.txt", content="hello", reason="test")
        scope = _compute_run_scope("run1", [doc], FlowOptions())
        assert str(scope).startswith("run1:")

    def test_different_docs_different_scope(self):
        doc1 = ScopeDoc.create_root(name="a.txt", content="aaa", reason="test")
        doc2 = ScopeDoc.create_root(name="b.txt", content="bbb", reason="test")
        scope1 = _compute_run_scope("run1", [doc1], FlowOptions())
        scope2 = _compute_run_scope("run1", [doc2], FlowOptions())
        assert scope1 != scope2

    def test_same_docs_same_scope(self):
        doc = ScopeDoc.create_root(name="a.txt", content="aaa", reason="test")
        opts = FlowOptions()
        scope1 = _compute_run_scope("run1", [doc], opts)
        scope2 = _compute_run_scope("run1", [doc], opts)
        assert scope1 == scope2


# ---------------------------------------------------------------------------
# _validate_flow_chain
# ---------------------------------------------------------------------------


class FlowInputDoc(Document):
    pass


class FlowOutputDoc(Document):
    pass


class FlowOutputDoc2(Document):
    pass


class _UnrelatedInputDoc(Document):
    pass


class _Flow1(PipelineFlow):
    async def run(self, run_id: str, documents: list[FlowInputDoc], options: FlowOptions) -> list[FlowOutputDoc]:
        return []


class _Flow2(PipelineFlow):
    async def run(self, run_id: str, documents: list[FlowOutputDoc], options: FlowOptions) -> list[FlowOutputDoc2]:
        return []


class _Flow2Bad(PipelineFlow):
    """Flow whose input is not produced by _Flow1 — chain validation must reject this."""

    async def run(self, run_id: str, documents: list[_UnrelatedInputDoc], options: FlowOptions) -> list[FlowOutputDoc2]:
        return []


class TestValidateFlowChain:
    def test_single_flow_ok(self):
        _validate_flow_chain("CliPipeline", [_Flow1()])

    def test_chained_flows_ok(self):
        _validate_flow_chain("CliPipeline", [_Flow1(), _Flow2()])

    def test_unsatisfied_input_raises(self):
        with pytest.raises(TypeError, match="requires input types"):
            _validate_flow_chain("CliPipeline", [_Flow1(), _Flow2Bad()])


# ---------------------------------------------------------------------------
# _heartbeat_loop
# ---------------------------------------------------------------------------


class TestHeartbeatLoop:
    async def test_heartbeat_publishes_and_can_cancel(self):
        publisher = MagicMock()
        publisher.publish_heartbeat = AsyncMock()
        with patch("ai_pipeline_core.deployment._helpers._HEARTBEAT_INTERVAL_SECONDS", 0.01):
            task = asyncio.create_task(_heartbeat_loop(publisher, "run-1"))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_heartbeat_swallows_exception(self):
        publisher = MagicMock()
        call_count = 0

        async def _failing_heartbeat(run_id: str) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("publish failed")

        publisher.publish_heartbeat = _failing_heartbeat

        with patch("ai_pipeline_core.deployment._helpers._HEARTBEAT_INTERVAL_SECONDS", 0.01):
            task = asyncio.create_task(_heartbeat_loop(publisher, "run-1"))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert call_count >= 1
