"""Tests for task_results durability — result written by PipelineDeployment.run().

Verifies that run() writes results to TaskResultStore when provided,
handles store failures gracefully, and supports reconciliation when
Pub/Sub publish fails after a successful result store write.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false, reportUnusedClass=false

import json

import pytest

from ai_pipeline_core.deployment._pubsub import PubSubPublisher
from ai_pipeline_core.deployment._task_results import MemoryTaskResultStore
from ai_pipeline_core.deployment._types import EventType

from .conftest import (
    PubsubTestResources,
    PublisherWithStore,
    TwoStageDeployment,
    pull_events,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub

# 2-flow success: 1 run.started + 2*(flow.started + task.started + task.completed + progress.STARTED + progress.COMPLETED + flow.completed) + 1 run.completed
TWO_FLOW_SUCCESS_EVENT_COUNT = 14


class TestTaskResults:
    """Task result store integration — result written by PipelineDeployment.run()."""

    async def test_result_store_contains_correct_result_and_chain_context(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store,
    ):
        """Result store contains the pipeline result and chain_context after a successful run."""
        run_id = "result-store-test"
        result_store = MemoryTaskResultStore()
        deployment = TwoStageDeployment()
        await run_pipeline(
            deployment,
            real_publisher.publisher,
            pubsub_memory_store,
            run_id=run_id,
            task_result_store=result_store,
        )

        # Drain events to confirm delivery
        pull_events(pubsub_test_resources, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)

        record = await result_store.read_result(run_id)
        assert record is not None, "task_result_store should contain a record after successful pipeline run"

        result = json.loads(record.result)
        assert result["success"] is True

        chain_context = json.loads(record.chain_context)
        assert chain_context["version"] == 1
        assert "run_scope" in chain_context
        assert isinstance(chain_context["output_document_refs"], list)
        assert len(chain_context["output_document_refs"]) > 0

    async def test_result_store_failure_does_not_block_completed_event(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store,
    ):
        """A failing result store does not prevent run.completed from being published."""

        class FailingResultStore(MemoryTaskResultStore):
            write_called = False

            async def write_result(self, run_id: str, result: str, chain_context: str) -> None:
                self.write_called = True
                raise RuntimeError("simulated ClickHouse failure")

        failing_store = FailingResultStore()

        deployment = TwoStageDeployment()
        await run_pipeline(
            deployment,
            real_publisher.publisher,
            pubsub_memory_store,
            run_id="failing-store-test",
            task_result_store=failing_store,
        )

        assert failing_store.write_called, "write_result must have been called before publish_run_completed"
        events = pull_events(pubsub_test_resources, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)
        completed_events = [e for e in events if e.event_type == EventType.RUN_COMPLETED]
        assert len(completed_events) == 1, "run.completed must still be published when result store fails"

    async def test_reconciliation_scenario_result_written_but_publish_failed(
        self,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store,
    ):
        """When _publish_critical fails for completed, write_result has already succeeded.

        run() writes to task_result_store BEFORE publisher.publish_run_completed(). So when
        _publish_critical fails on the completed event, the result is already in the
        store. The exception propagates to run()'s except block, which publishes a
        RunFailedEvent. The consumer sees 'failed', but the result is recoverable from
        the result store — enabling reconciliation.

        Call sequence:
        1. _publish_critical(run.started) -> succeeds (call_count=1)
        2. _publish_critical(run.completed) -> raises (call_count=2)
        3. run() except block -> publish_run_failed -> _publish_critical(run.failed) -> succeeds (call_count=3)
        """
        result_store = MemoryTaskResultStore()

        class FailOnCompletedPublisher(PubSubPublisher):
            """Publisher that fails _publish_critical on the second call (completed)."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._critical_call_count = 0

            async def _publish_critical(self, data: bytes, attributes: dict[str, str]) -> None:
                self._critical_call_count += 1
                if self._critical_call_count == 2:
                    raise RuntimeError("simulated Pub/Sub failure on completed event")
                await super()._publish_critical(data, attributes)

        topic_id = pubsub_test_resources.topic_path.split("/")[-1]
        publisher = FailOnCompletedPublisher(
            project_id=pubsub_test_resources.project_id,
            topic_id=topic_id,
            service_type="test-service",
        )

        run_id = "reconciliation-test"
        deployment = TwoStageDeployment()

        # The run() will raise because publish_run_completed fails, and run() re-raises after publishing failed
        with pytest.raises(RuntimeError, match="simulated Pub/Sub failure"):
            await run_pipeline(
                deployment,
                publisher,
                pubsub_memory_store,
                run_id=run_id,
                task_result_store=result_store,
            )

        # Pull events: run.started + flow events + progress events + run.failed (run.completed was never published)
        # 1 run.started + 2*(flow.started + progress.STARTED + progress.COMPLETED + flow.completed) + 1 run.failed = 10
        events = pull_events(pubsub_test_resources, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)

        event_types = [e.event_type for e in events]
        assert EventType.RUN_STARTED in event_types
        assert EventType.RUN_FAILED in event_types
        assert EventType.RUN_COMPLETED not in event_types, "run.completed must NOT be published when _publish_critical fails"

        # The key assertion: result IS in the store despite the consumer seeing 'failed'
        record = await result_store.read_result(run_id)
        assert record is not None, "write_result succeeded before publish_run_completed failed — result must be recoverable"

        result = json.loads(record.result)
        assert result["success"] is True
