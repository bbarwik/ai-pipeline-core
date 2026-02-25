"""Unit tests for ai_pipeline_core.deployment._cli — debug tracing and CLI wiring."""

# pyright: reportPrivateUsage=false

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_pipeline_core.deployment._cli import _init_debug_tracing
from ai_pipeline_core.deployment.base import _classify_error, _create_publisher, _build_summary_generator
from ai_pipeline_core.deployment._publishers import NoopPublisher
from ai_pipeline_core.deployment._types import ErrorCode
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError
from ai_pipeline_core.settings import Settings


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
        assert isinstance(publisher, NoopPublisher)

    def test_noop_when_no_service_type(self):
        s = Settings(pubsub_project_id="proj", pubsub_topic_id="topic")
        publisher = _create_publisher(s, "")
        assert isinstance(publisher, NoopPublisher)

    def test_missing_clickhouse_raises(self):
        s = Settings(
            pubsub_project_id="proj",
            pubsub_topic_id="topic",
            clickhouse_host="",
        )
        with pytest.raises(ValueError, match="CLICKHOUSE_HOST"):
            _create_publisher(s, "svc")


# ---------------------------------------------------------------------------
# _build_summary_generator
# ---------------------------------------------------------------------------


class TestBuildSummaryGenerator:
    @patch("ai_pipeline_core.deployment.base.settings")
    def test_disabled_returns_none(self, mock_settings):
        mock_settings.doc_summary_enabled = False
        result = _build_summary_generator()
        assert result is None

    @patch("ai_pipeline_core.deployment.base.settings")
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

from unittest.mock import AsyncMock

from ai_pipeline_core.deployment.base import _compute_run_scope, _validate_flow_chain, _heartbeat_loop
from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline.options import FlowOptions


class ScopeDoc(Document):
    pass


class TestComputeRunScope:
    def test_no_documents(self):
        opts = FlowOptions()
        scope = _compute_run_scope("run1", [], opts)
        assert str(scope).startswith("run1:")

    def test_with_documents(self):
        doc = ScopeDoc(name="test.txt", content=b"hello")
        scope = _compute_run_scope("run1", [doc], FlowOptions())
        assert str(scope).startswith("run1:")

    def test_different_docs_different_scope(self):
        doc1 = ScopeDoc(name="a.txt", content=b"aaa")
        doc2 = ScopeDoc(name="b.txt", content=b"bbb")
        scope1 = _compute_run_scope("run1", [doc1], FlowOptions())
        scope2 = _compute_run_scope("run1", [doc2], FlowOptions())
        assert scope1 != scope2

    def test_same_docs_same_scope(self):
        doc = ScopeDoc(name="a.txt", content=b"aaa")
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


class TestValidateFlowChain:
    def test_single_flow_ok(self):
        flow_fn = MagicMock()
        flow_fn.input_document_types = [FlowInputDoc]
        flow_fn.output_document_types = [FlowOutputDoc]
        _validate_flow_chain("TestPipeline", [flow_fn])

    def test_chained_flows_ok(self):
        flow1 = MagicMock()
        flow1.input_document_types = [FlowInputDoc]
        flow1.output_document_types = [FlowOutputDoc]
        flow1.__name__ = "flow1"
        flow2 = MagicMock()
        flow2.input_document_types = [FlowOutputDoc]
        flow2.output_document_types = [FlowOutputDoc2]
        flow2.__name__ = "flow2"
        _validate_flow_chain("TestPipeline", [flow1, flow2])

    def test_unsatisfied_input_raises(self):
        flow1 = MagicMock()
        flow1.input_document_types = [FlowInputDoc]
        flow1.output_document_types = [FlowInputDoc]
        flow1.__name__ = "flow1"
        flow2 = MagicMock()
        flow2.input_document_types = [FlowOutputDoc2]
        flow2.output_document_types = []
        flow2.__name__ = "flow2"
        with pytest.raises(TypeError, match="requires input types"):
            _validate_flow_chain("TestPipeline", [flow1, flow2])


# ---------------------------------------------------------------------------
# _heartbeat_loop
# ---------------------------------------------------------------------------


class TestHeartbeatLoop:
    async def test_heartbeat_publishes_and_can_cancel(self):
        publisher = MagicMock()
        publisher.publish_heartbeat = AsyncMock()
        with patch("ai_pipeline_core.deployment.base._HEARTBEAT_INTERVAL_SECONDS", 0.01):
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

        with patch("ai_pipeline_core.deployment.base._HEARTBEAT_INTERVAL_SECONDS", 0.01):
            task = asyncio.create_task(_heartbeat_loop(publisher, "run-1"))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert call_count >= 1
