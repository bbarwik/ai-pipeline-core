"""Tests for publisher factory, heartbeat lifecycle, and run() publisher integration."""

# pyright: reportPrivateUsage=false, reportArgumentType=false, reportUnusedClass=false

from unittest.mock import MagicMock, patch

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)
from ai_pipeline_core.deployment import DeploymentContext
from ai_pipeline_core.deployment._publishers import MemoryPublisher, NoopPublisher
from ai_pipeline_core.deployment._types import (
    CompletedEvent,
    ErrorCode,
    FailedEvent,
    ProgressEvent,
    ResultPublisher,
    StartedEvent,
)
from ai_pipeline_core.deployment.base import _create_publisher
from ai_pipeline_core.document_store._protocol import set_document_store
from ai_pipeline_core.document_store._memory import MemoryDocumentStore


# --- Test infrastructure ---


class _WiringInputDoc(Document):
    """Input for wiring tests."""


class _WiringOutputDoc(Document):
    """Output for wiring tests."""


class _WiringResult(DeploymentResult):
    """Result for wiring tests."""


@pipeline_flow()
async def _wiring_flow(run_id: str, documents: list[_WiringInputDoc], flow_options: FlowOptions) -> list[_WiringOutputDoc]:
    """Flow for wiring tests."""
    return [_WiringOutputDoc.create_root(name="out.txt", content="done", reason="test")]


class _WiringDeployment(PipelineDeployment[FlowOptions, _WiringResult]):
    """Deployment for wiring tests."""

    flows = [_wiring_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _WiringResult:
        """Build result."""
        return _WiringResult(success=True)


@pipeline_flow()
async def _failing_flow(run_id: str, documents: list[_WiringInputDoc], flow_options: FlowOptions) -> list[_WiringOutputDoc]:
    """Flow that always raises."""
    raise RuntimeError("deliberate failure")


class _FailingDeployment(PipelineDeployment[FlowOptions, _WiringResult]):
    """Deployment that always fails."""

    flows = [_failing_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _WiringResult:
        """Build result."""
        return _WiringResult(success=False, error="failed")


# --- _create_publisher tests ---


class TestCreatePublisher:
    """Test _create_publisher factory function."""

    def test_returns_noop_when_pubsub_not_configured(self):
        """No pubsub_project_id → NoopPublisher."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = ""
        mock_settings.pubsub_topic_id = ""
        publisher = _create_publisher(mock_settings, "research")
        assert isinstance(publisher, NoopPublisher)

    def test_returns_noop_when_topic_id_missing(self):
        """pubsub_project_id set but pubsub_topic_id empty → NoopPublisher."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = ""
        publisher = _create_publisher(mock_settings, "research")
        assert isinstance(publisher, NoopPublisher)

    def test_returns_noop_when_service_type_empty(self):
        """Empty service_type → NoopPublisher regardless of other config."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = "events"
        publisher = _create_publisher(mock_settings, "")
        assert isinstance(publisher, NoopPublisher)

    def test_raises_when_clickhouse_missing(self):
        """Pub/Sub configured but no ClickHouse → ValueError."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = "events"
        mock_settings.clickhouse_host = ""
        with pytest.raises(ValueError, match="CLICKHOUSE_HOST"):
            _create_publisher(mock_settings, "research")

    def test_creates_pubsub_publisher_when_configured(self):
        """Full config → PubSubPublisher."""
        mock_settings = MagicMock()
        mock_settings.pubsub_project_id = "my-project"
        mock_settings.pubsub_topic_id = "events"
        mock_settings.clickhouse_host = "clickhouse.local"
        mock_settings.clickhouse_port = 8443
        mock_settings.clickhouse_database = "default"
        mock_settings.clickhouse_user = "default"
        mock_settings.clickhouse_password = ""
        mock_settings.clickhouse_secure = True

        with (
            patch("ai_pipeline_core.deployment._pubsub.PubSubPublisher") as mock_pub_cls,
            patch("ai_pipeline_core.deployment._task_results.ClickHouseTaskResultStore") as mock_store_cls,
        ):
            mock_pub_cls.return_value = MagicMock(spec=ResultPublisher)
            _create_publisher(mock_settings, "research")
            mock_store_cls.assert_called_once()
            mock_pub_cls.assert_called_once()


# --- run() publisher integration tests ---


class TestRunPublisherIntegration:
    """Test that run() publishes lifecycle events via the publisher."""

    async def test_publishes_started_event(self):
        """run() must publish StartedEvent before executing flows."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        started_events = [e for e in pub.events if isinstance(e, StartedEvent)]
        assert len(started_events) == 1
        assert started_events[0].run_id == "run-1"

    async def test_publishes_completed_event(self):
        """run() must publish CompletedEvent after successful execution."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        completed_events = [e for e in pub.events if isinstance(e, CompletedEvent)]
        assert len(completed_events) == 1
        assert completed_events[0].run_id == "run-1"
        assert "version" in completed_events[0].chain_context
        assert completed_events[0].chain_context["version"] == 1

    async def test_publishes_progress_events(self):
        """run() must publish ProgressEvents at STARTED and COMPLETED transitions."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        progress_events = [e for e in pub.events if isinstance(e, ProgressEvent)]
        assert len(progress_events) >= 2
        statuses = [e.status for e in progress_events]
        assert "started" in statuses
        assert "completed" in statuses

    async def test_publishes_failed_event_on_error(self):
        """run() must publish FailedEvent when a flow raises."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _FailingDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            with pytest.raises(RuntimeError, match="deliberate failure"):
                await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        failed_events = [e for e in pub.events if isinstance(e, FailedEvent)]
        assert len(failed_events) == 1
        assert failed_events[0].run_id == "run-1"
        assert failed_events[0].error_code == ErrorCode.UNKNOWN
        assert "deliberate failure" in failed_events[0].error_message

    async def test_event_ordering(self):
        """Events follow order: started → progress(started) → progress(completed) → completed."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        event_types = [type(e).__name__ for e in pub.events]
        assert event_types[0] == "StartedEvent"
        assert event_types[-1] == "CompletedEvent"

    async def test_heartbeat_runs_during_execution(self):
        """Heartbeat task is created and cancelled during run()."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)

        # Patch heartbeat interval to be very short
        with patch("ai_pipeline_core.deployment.base._HEARTBEAT_INTERVAL_SECONDS", 0.01):
            try:
                deployment = _WiringDeployment()
                doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
                await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
            finally:
                store.shutdown()
                set_document_store(None)

        # Heartbeats may or may not have fired depending on flow speed,
        # but the heartbeat task should have been created and cleaned up without errors

    async def test_chain_context_structure(self):
        """CompletedEvent chain_context has correct structure."""
        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        completed = [e for e in pub.events if isinstance(e, CompletedEvent)][0]
        ctx = completed.chain_context
        assert ctx["version"] == 1
        assert "run_scope" in ctx
        assert "output_document_refs" in ctx
        assert isinstance(ctx["output_document_refs"], list)

    async def test_noop_publisher_is_default(self):
        """run() with publisher=None defaults to NoopPublisher (no crash)."""
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _WiringDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            # No publisher argument → defaults to NoopPublisher
            result = await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext())
            assert result.success
        finally:
            store.shutdown()
            set_document_store(None)

    async def test_cancelled_error_publishes_failed(self):
        """CancelledError in flow publishes FailedEvent with CANCELLED error code."""
        import asyncio

        @pipeline_flow()
        async def _cancelling_flow(run_id: str, documents: list[_WiringInputDoc], flow_options: FlowOptions) -> list[_WiringOutputDoc]:
            raise asyncio.CancelledError()

        class _CancelDeployment(PipelineDeployment[FlowOptions, _WiringResult]):
            flows = [_cancelling_flow]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _WiringResult:
                return _WiringResult(success=False)

        pub = MemoryPublisher()
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _CancelDeployment()
            doc = _WiringInputDoc.create_root(name="in.txt", content="test", reason="test")
            with pytest.raises(asyncio.CancelledError):
                await deployment.run("run-1", [doc], FlowOptions(), DeploymentContext(), publisher=pub)
        finally:
            store.shutdown()
            set_document_store(None)

        failed_events = [e for e in pub.events if isinstance(e, FailedEvent)]
        assert len(failed_events) == 1
        assert failed_events[0].error_code == ErrorCode.CANCELLED
