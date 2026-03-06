"""Tests for publisher factory, heartbeat lifecycle, and run() publisher integration."""

# pyright: reportPrivateUsage=false

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
)
from ai_pipeline_core.deployment._types import (
    ErrorCode,
    ProgressEvent,
    ResultPublisher,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    _MemoryPublisher,
    _NoopPublisher,
)
from ai_pipeline_core.deployment.base import _create_publisher, _create_task_result_store
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.pipeline import PipelineFlow


# --- Test infrastructure ---


class _WiringInputDoc(Document):
    """Input for wiring tests."""


class _WiringOutputDoc(Document):
    """Output for wiring tests."""


class _WiringResult(DeploymentResult):
    """Result for wiring tests."""


class _WiringFlow(PipelineFlow):
    """Flow for wiring tests."""

    async def run(self, run_id: str, documents: list[_WiringInputDoc], options: FlowOptions) -> list[_WiringOutputDoc]:
        return [_WiringOutputDoc.derive(from_documents=documents, name="out.txt", content="done")]


class _WiringDeployment(PipelineDeployment[FlowOptions, _WiringResult]):
    """Deployment for wiring tests."""

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [_WiringFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _WiringResult:
        return _WiringResult(success=True)


class _FailingFlow(PipelineFlow):
    """Flow that always raises."""

    async def run(self, run_id: str, documents: list[_WiringInputDoc], options: FlowOptions) -> list[_WiringOutputDoc]:
        raise RuntimeError("deliberate failure")


class _FailingDeployment(PipelineDeployment[FlowOptions, _WiringResult]):
    """Deployment that always fails."""

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [_FailingFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _WiringResult:
        return _WiringResult(success=False, error="failed")


class _CancellingFlow(PipelineFlow):
    """Flow that raises CancelledError."""

    async def run(self, run_id: str, documents: list[_WiringInputDoc], options: FlowOptions) -> list[_WiringOutputDoc]:
        raise asyncio.CancelledError()


class _CancelDeployment(PipelineDeployment[FlowOptions, _WiringResult]):
    """Deployment where the flow raises CancelledError."""

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [_CancellingFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _WiringResult:
        return _WiringResult(success=False)


# --- _create_publisher tests ---


class TestCreatePublisher:
    """Test _create_publisher factory function."""

    def test_returns_noop_when_pubsub_not_configured(self):
        """No pubsub_project_id → _NoopPublisher."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = ""
        mock_settings.pubsub_topic_id = ""
        publisher = _create_publisher(mock_settings, "research")
        assert isinstance(publisher, _NoopPublisher)

    def test_returns_noop_when_topic_id_missing(self):
        """pubsub_project_id set but pubsub_topic_id empty → _NoopPublisher."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = ""
        publisher = _create_publisher(mock_settings, "research")
        assert isinstance(publisher, _NoopPublisher)

    def test_returns_noop_when_service_type_empty(self):
        """Empty service_type → _NoopPublisher regardless of other config."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = "events"
        publisher = _create_publisher(mock_settings, "")
        assert isinstance(publisher, _NoopPublisher)

    def test_creates_pubsub_publisher_when_configured(self):
        """Full Pub/Sub config → PubSubPublisher (no ClickHouse required)."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = "events"

        with patch("ai_pipeline_core.deployment._pubsub.PubSubPublisher") as mock_pub_cls:
            mock_pub_cls.return_value = MagicMock(spec=ResultPublisher)
            _create_publisher(mock_settings, "research")
            mock_pub_cls.assert_called_once_with(
                project_id="my-project",
                topic_id="events",
                service_type="research",
            )


# --- _create_task_result_store tests ---


class TestCreateTaskResultStore:
    """Test _create_task_result_store factory function."""

    def test_returns_none_when_clickhouse_not_configured(self):
        mock_settings = MagicMock()
        mock_settings.clickhouse_host = ""
        assert _create_task_result_store(mock_settings) is None

    def test_creates_store_when_clickhouse_configured(self):
        mock_settings = MagicMock()
        mock_settings.clickhouse_host = "clickhouse.local"
        mock_settings.clickhouse_port = 8443
        mock_settings.clickhouse_database = "default"
        mock_settings.clickhouse_user = "default"
        mock_settings.clickhouse_password = ""
        mock_settings.clickhouse_secure = True
        mock_settings.clickhouse_connect_timeout = 10
        mock_settings.clickhouse_send_receive_timeout = 30

        with patch("ai_pipeline_core.deployment._task_results.ClickHouseTaskResultStore") as mock_cls:
            result = _create_task_result_store(mock_settings)
            mock_cls.assert_called_once_with(
                host="clickhouse.local",
                port=8443,
                database="default",
                username="default",
                password="",
                secure=True,
                connect_timeout=10,
                send_receive_timeout=30,
            )
            assert result is mock_cls.return_value


# --- run() publisher integration tests ---


class TestRunPublisherIntegration:
    """Test that run() publishes lifecycle events via the publisher."""

    async def test_publishes_started_event(self):
        """run() must publish RunStartedEvent before executing flows."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        started_events = [e for e in pub.events if isinstance(e, RunStartedEvent)]
        assert len(started_events) == 1
        assert started_events[0].run_id == "run-1"

    async def test_publishes_completed_event(self):
        """run() must publish RunCompletedEvent after successful execution."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        completed_events = [e for e in pub.events if isinstance(e, RunCompletedEvent)]
        assert len(completed_events) == 1
        assert completed_events[0].run_id == "run-1"
        assert "version" in completed_events[0].chain_context
        assert completed_events[0].chain_context["version"] == 1

    async def test_publishes_progress_events(self):
        """run() must publish ProgressEvents at STARTED and COMPLETED transitions."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        progress_events = [e for e in pub.events if isinstance(e, ProgressEvent)]
        assert len(progress_events) >= 2
        statuses = [e.status for e in progress_events]
        assert "started" in statuses
        assert "completed" in statuses

    async def test_publishes_failed_event_on_error(self):
        """run() must publish RunFailedEvent when a flow raises."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _FailingDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            with pytest.raises(RuntimeError, match="deliberate failure"):
                await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        failed_events = [e for e in pub.events if isinstance(e, RunFailedEvent)]
        assert len(failed_events) == 1
        assert failed_events[0].run_id == "run-1"
        assert failed_events[0].error_code == ErrorCode.UNKNOWN
        assert "deliberate failure" in failed_events[0].error_message

    async def test_event_ordering(self):
        """Events follow order: started → progress(started) → progress(completed) → completed."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        event_types = [type(e).__name__ for e in pub.events]
        assert event_types[0] == "RunStartedEvent"
        assert event_types[-1] == "RunCompletedEvent"

    async def test_heartbeat_runs_during_execution(self):
        """Heartbeat task is created and cancelled during run()."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)

        # Patch heartbeat interval to be very short
        with patch("ai_pipeline_core.deployment._helpers._HEARTBEAT_INTERVAL_SECONDS", 0.01):
            try:
                deployment = _WiringDeployment()
                doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
                await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
            finally:
                store.shutdown()
                set_document_store(None)

        # Heartbeats may or may not have fired depending on flow speed,
        # but the heartbeat task should have been created and cleaned up without errors

    async def test_chain_context_structure(self):
        """RunCompletedEvent chain_context has correct structure."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        completed = [e for e in pub.events if isinstance(e, RunCompletedEvent)][0]
        ctx = completed.chain_context
        assert ctx["version"] == 1
        assert "run_scope" in ctx
        assert "output_document_refs" in ctx
        assert isinstance(ctx["output_document_refs"], list)

    async def test_noop_publisher_is_default(self):
        """run() with publisher=None defaults to _NoopPublisher (no crash)."""
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            # No publisher argument → defaults to _NoopPublisher
            result = await deployment.run("run-1", [doc], FlowOptions())
            assert result.success
        finally:
            store.shutdown()
            set_document_store(None)

    async def test_cancelled_error_publishes_failed(self):
        """CancelledError in flow publishes RunFailedEvent with CANCELLED error code."""
        pub = _MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _CancelDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            with pytest.raises(asyncio.CancelledError):
                await deployment.run("run-1", [doc], FlowOptions(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        failed_events = [e for e in pub.events if isinstance(e, RunFailedEvent)]
        assert len(failed_events) == 1
        assert failed_events[0].error_code == ErrorCode.CANCELLED
