"""Pub/Sub integration tests: full pipeline lifecycle event publishing.

Tests the core contract that consumers depend on — started, progress, completed/failed
events are published correctly through a real PubSubPublisher to the Pub/Sub emulator.
"""

# pyright: reportPrivateUsage=false, reportUnusedClass=false, reportArgumentType=false

import asyncio
from typing import Any, ClassVar

import pytest

from ai_pipeline_core import DeploymentResult, Document, FlowOptions, PipelineDeployment, pipeline_flow
from ai_pipeline_core.deployment._pubsub import PubSubPublisher, ResultTooLargeError
from ai_pipeline_core.deployment._types import EventType, ErrorCode
from ai_pipeline_core.deployment.contract import FlowStatus
from ai_pipeline_core.document_store._memory import MemoryDocumentStore

from .conftest import (
    PublisherWithStore,
    PubSubTestResources,
    PubsubInputDoc,
    PubsubOutputDoc,
    PubsubResult,
    TwoFlowDeployment,
    FailingSecondFlowDeployment,
    _flow_executions,
    assert_seq_monotonic,
    assert_valid_cloudevent,
    pull_events,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub


# ---------------------------------------------------------------------------
# Locally-defined deployments for specific test scenarios
# ---------------------------------------------------------------------------


@pipeline_flow(estimated_minutes=5)
async def _cancelling_flow(
    run_id: str,
    documents: list[PubsubInputDoc],
    flow_options: FlowOptions,
) -> list[PubsubOutputDoc]:
    """Flow that raises CancelledError."""
    _flow_executions.append("cancelling_flow")
    raise asyncio.CancelledError()


class CancellingDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """Deployment where the first flow raises CancelledError."""

    flows: ClassVar = [_cancelling_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        return PubsubResult(success=False, error="cancelled")


class BuildResultFailingDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """Deployment where build_result raises ValueError after all flows succeed."""

    flows: ClassVar = [TwoFlowDeployment.flows[0], TwoFlowDeployment.flows[1]]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        raise ValueError("extraction failed")


class OversizedResult(DeploymentResult):
    """Result with a very large field that exceeds the 8MB Pub/Sub limit."""

    huge_field: str = ""


class OversizedResultDeployment(PipelineDeployment[FlowOptions, OversizedResult]):
    """Deployment whose build_result returns an oversized result."""

    flows: ClassVar = [TwoFlowDeployment.flows[0], TwoFlowDeployment.flows[1]]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> OversizedResult:
        return OversizedResult(success=True, huge_field="x" * (9 * 1024 * 1024))


class FailingStartedPublisher(PubSubPublisher):
    """PubSubPublisher subclass that fails on publish_started."""

    async def publish_started(self, event: Any) -> None:
        raise RuntimeError("publish_started injection failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPubsubLifecycle:
    """Full pipeline lifecycle event publishing through real Pub/Sub emulator."""

    async def test_successful_pipeline_publishes_complete_event_sequence(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """2-flow success: started + 4 progress (2 flows x STARTED+COMPLETED) + completed = 6 events."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_topic, expected_count=6)

        # First event is started, last is completed
        assert events[0].event_type == EventType.STARTED
        assert events[-1].event_type == EventType.COMPLETED

        # Exactly 4 progress events (STARTED + COMPLETED per flow x 2 flows)
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress_events) == 4

        # No failed events
        failed_events = [e for e in events if e.event_type == EventType.FAILED]
        assert len(failed_events) == 0

        # Seq values strictly increasing
        assert_seq_monotonic(events)

        # Exactly 1 terminal event (completed)
        terminal = [e for e in events if e.event_type in (EventType.COMPLETED, EventType.FAILED)]
        assert len(terminal) == 1

        # All events are valid CloudEvents
        for event in events:
            assert_valid_cloudevent(event)

        # Completed event contains correct result data
        completed_data = events[-1].data
        assert completed_data["result"]["success"] is True
        assert completed_data["result"]["doc_count"] > 0
        assert "actual_cost" in completed_data

    async def test_failed_pipeline_publishes_started_and_failed(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Flow 2 raises RuntimeError: started + 3 progress + failed = 5 events."""
        deployment = FailingSecondFlowDeployment()
        with pytest.raises(RuntimeError, match="deliberate test failure"):
            await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        # started + flow1 STARTED + flow1 COMPLETED + failing_flow STARTED + failed = 5 events
        # (progress STARTED is published for failing_flow before it executes and raises)
        events = pull_events(pubsub_topic, expected_count=5)

        # started present
        started = [e for e in events if e.event_type == EventType.STARTED]
        assert len(started) == 1

        # progress events: flow1 STARTED + flow1 COMPLETED + failing_flow STARTED = 3
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress_events) == 3

        # failed present with correct error details
        failed = [e for e in events if e.event_type == EventType.FAILED]
        assert len(failed) == 1
        assert failed[0].data["error_code"] == ErrorCode.UNKNOWN
        assert "deliberate test failure" in failed[0].data["error_message"]

        # NO completed event
        completed = [e for e in events if e.event_type == EventType.COMPLETED]
        assert len(completed) == 0

        assert_seq_monotonic(events)

    async def test_cancelled_pipeline_publishes_cancelled_error_code(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """CancelledError in flow publishes task.failed with error_code=CANCELLED."""
        deployment = CancellingDeployment()
        with pytest.raises(asyncio.CancelledError):
            await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        # started + progress(STARTED for the flow) + failed = 3 events
        # (run() publishes progress STARTED before executing the flow)
        events = pull_events(pubsub_topic, expected_count=3)

        failed = [e for e in events if e.event_type == EventType.FAILED]
        assert len(failed) == 1
        assert failed[0].data["error_code"] == ErrorCode.CANCELLED

        assert_seq_monotonic(events)

    async def test_build_result_failure_publishes_failed_after_all_flows_complete(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """build_result() raises ValueError: all flow progress events are published, then failed."""
        deployment = BuildResultFailingDeployment()
        with pytest.raises(ValueError, match="extraction failed"):
            await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        # started + 4 progress (2 flows x STARTED+COMPLETED) + failed = 6 events
        events = pull_events(pubsub_topic, expected_count=6)

        # All flow progress events present (both flows completed successfully)
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress_events) == 4
        statuses = [e.data["status"] for e in progress_events]
        assert statuses.count(FlowStatus.STARTED) == 2
        assert statuses.count(FlowStatus.COMPLETED) == 2

        # Failed event with correct error
        failed = [e for e in events if e.event_type == EventType.FAILED]
        assert len(failed) == 1
        assert failed[0].data["error_code"] == ErrorCode.INVALID_INPUT
        assert "extraction failed" in failed[0].data["error_message"]

        # NO completed event
        completed = [e for e in events if e.event_type == EventType.COMPLETED]
        assert len(completed) == 0

        # Verify both flows actually executed (work not lost despite build_result failure)
        assert "flow_1" in _flow_executions
        assert "flow_2" in _flow_executions

    async def test_result_too_large_publishes_failed_event(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Oversized build_result output: failed with PIPELINE_ERROR, no result in store."""
        deployment = OversizedResultDeployment()
        with pytest.raises(ResultTooLargeError):
            await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        # started + 4 progress + failed = 6 events
        events = pull_events(pubsub_topic, expected_count=6)

        failed = [e for e in events if e.event_type == EventType.FAILED]
        assert len(failed) == 1
        assert failed[0].data["error_code"] == ErrorCode.PIPELINE_ERROR
        assert "byte" in failed[0].data["error_message"].lower()

        # NO completed event
        completed = [e for e in events if e.event_type == EventType.COMPLETED]
        assert len(completed) == 0

        # Result store should be empty — size guard fires before write_result
        record = await real_publisher.result_store.read_result("test-run")
        assert record is None

    async def test_publish_started_failure_aborts_pipeline(
        self,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """publish_started failure prevents any flow execution; original exception propagates."""
        from ai_pipeline_core.deployment._task_results import MemoryTaskResultStore

        result_store = MemoryTaskResultStore()
        topic_id = pubsub_topic.topic_path.split("/")[-1]
        failing_publisher = FailingStartedPublisher(
            project_id=pubsub_topic.project_id,
            topic_id=topic_id,
            service_type="test-service",
            result_store=result_store,
        )

        deployment = TwoFlowDeployment()
        with pytest.raises(RuntimeError, match="publish_started injection failure"):
            await run_pipeline(deployment, failing_publisher, pubsub_memory_store)

        # No flows should have executed
        assert len(_flow_executions) == 0

        # publish_failed was attempted (may succeed or fail). Since publish_started
        # failed before the real publisher could send, we check for 0 or 1 events.
        # The except block in run() tries publish_failed which uses the real
        # PubSubPublisher._publish_critical (inherited), so a failed event may arrive.
        try:
            events = pull_events(pubsub_topic, expected_count=1, timeout=3.0)
        except AssertionError:
            # No events arrived — publish_failed may also have failed, which is acceptable
            return

        # If we got an event, it must be task.failed (not started or completed)
        assert events[0].event_type == EventType.FAILED

    async def test_event_stream_forms_valid_state_machine(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Event stream sorted by seq forms a valid state machine for both success and failure.

        Successful: started -> progress* -> completed
        Failed: started -> progress* -> failed
        Both cases use the same topic — successful run first, then failed run.
        Events from both runs are pulled together and partitioned by run_id.
        """
        # --- Successful run ---
        deployment = TwoFlowDeployment()
        await run_pipeline(
            deployment,
            real_publisher.publisher,
            pubsub_memory_store,
            run_id="sm-success",
        )

        success_events = pull_events(pubsub_topic, expected_count=6)
        assert_seq_monotonic(success_events)
        assert success_events[0].event_type == EventType.STARTED
        assert success_events[-1].event_type == EventType.COMPLETED
        for event in success_events[1:-1]:
            assert event.event_type == EventType.PROGRESS

        # --- Failed run (same topic, events accumulate in subscription) ---
        failed_deployment = FailingSecondFlowDeployment()
        with pytest.raises(RuntimeError, match="deliberate test failure"):
            await run_pipeline(
                failed_deployment,
                real_publisher.publisher,
                pubsub_memory_store,
                run_id="sm-failure",
            )

        failed_events = pull_events(pubsub_topic, expected_count=5)
        assert_seq_monotonic(failed_events)
        assert failed_events[0].event_type == EventType.STARTED
        assert failed_events[-1].event_type == EventType.FAILED
        for event in failed_events[1:-1]:
            assert event.event_type == EventType.PROGRESS
