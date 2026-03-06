"""Tests for progress events published via real Pub/Sub emulator.

Tests both flow-level progress events (STARTED/COMPLETED/CACHED) and intra-flow
progress_update() events. The latter are fire-and-forget (asyncio.create_task),
so tests yield the event loop after run() to let them flush before pulling.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false

import asyncio
from collections import defaultdict

import pytest

from ai_pipeline_core import (
    Document,
    FlowOptions,
    PipelineDeployment,
)
from ai_pipeline_core.deployment._types import EventType
from ai_pipeline_core.deployment.progress import progress_update
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.pipeline import PipelineFlow

from .conftest import (
    CollectedEvent,
    PubsubTestResources,
    PubsubInputDoc,
    PubsubMiddleDoc,
    PubsubOutputDoc,
    PubsubResult,
    PublisherWithStore,
    ThreeStageDeployment,
    make_input_doc,
    pull_events,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub

# 3-flow success: 1 run.started + 3*(flow.started + task.started + task.completed + progress.STARTED + progress.COMPLETED + flow.completed) + 1 run.completed
THREE_FLOW_EVENT_COUNT = 20


def _progress_events(events: list[CollectedEvent]) -> list[CollectedEvent]:
    """Filter to only progress events."""
    return [e for e in events if e.event_type == EventType.PROGRESS]


class TestProgressStepNumbers:
    """Verify step numbering in progress events."""

    async def test_progress_step_numbers_match_flow_count(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """3-flow deployment: all progress events have total_steps=3, steps 1-3 all present."""
        deployment = ThreeStageDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_test_resources, expected_count=THREE_FLOW_EVENT_COUNT)
        progress = _progress_events(events)

        # All progress events report total_steps=3
        for evt in progress:
            assert evt.data["total_steps"] == 3, f"total_steps mismatch on event: {evt.data}"

        # Steps 1, 2, 3 are all represented
        steps_seen = {evt.data["step"] for evt in progress}
        assert steps_seen == {1, 2, 3}


class TestProgressFlowNames:
    """Verify flow names in progress events match PipelineFlow class names."""

    async def test_progress_flow_names_match_class_names(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Each progress event's flow_name matches the actual PipelineFlow name."""
        deployment = ThreeStageDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_test_resources, expected_count=THREE_FLOW_EVENT_COUNT)
        progress = _progress_events(events)

        expected_names = {"chain_input_to_middle", "chain_middle_to_output", "chain_output_to_final"}
        seen_names = {evt.data["flow_name"] for evt in progress}
        assert seen_names == expected_names


class TestProgressStatusTransitions:
    """Verify STARTED -> COMPLETED ordering per flow."""

    async def test_progress_status_transitions_per_flow(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Each flow has a STARTED then COMPLETED progress event, in that seq order."""
        deployment = ThreeStageDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_test_resources, expected_count=THREE_FLOW_EVENT_COUNT)
        progress = _progress_events(events)

        # Group by flow_name, preserving seq order (events are already sorted by seq)
        by_flow: dict[str, list[CollectedEvent]] = defaultdict(list)
        for evt in progress:
            by_flow[evt.data["flow_name"]].append(evt)

        for flow_name, flow_events in by_flow.items():
            statuses = [evt.data["status"] for evt in flow_events]
            assert statuses == ["started", "completed"], f"Flow '{flow_name}' expected [started, completed] but got {statuses}"
            # Verify seq ordering: started comes before completed
            assert flow_events[0].seq < flow_events[1].seq


class TestPartialRun:
    """Verify partial runs (--start/--end) publish only executed steps."""

    async def test_partial_run_publishes_only_executed_steps(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """start_step=2, end_step=2: total_steps=3 but only step=2 progress events emitted."""
        deployment = ThreeStageDeployment()

        # Pre-populate store with flow_a's output so flow_b can load its inputs
        doc = make_input_doc()

        middle_doc = PubsubMiddleDoc.create_root(name="a_out.json", content={"a": 1}, reason="test")
        from ai_pipeline_core.deployment.base import _compute_run_scope

        run_scope = _compute_run_scope("test-run", [doc], FlowOptions())
        await pubsub_memory_store.save(doc, run_scope)
        await pubsub_memory_store.save(middle_doc, run_scope)

        await run_pipeline(
            deployment,
            real_publisher.publisher,
            pubsub_memory_store,
            start_step=2,
            end_step=2,
        )

        # Partial run: 1 run.started + flow/task/progress events for 1 flow + 1 run.completed = 8
        events = pull_events(pubsub_test_resources, expected_count=8)
        progress = _progress_events(events)

        assert len(progress) == 2
        for evt in progress:
            assert evt.data["step"] == 2
            assert evt.data["total_steps"] == 3

        # run.completed is still published
        completed = [e for e in events if e.event_type == EventType.RUN_COMPLETED]
        assert len(completed) == 1


# ---------------------------------------------------------------------------
# Intra-flow progress_update() — fire-and-forget publish via real Pub/Sub
# ---------------------------------------------------------------------------

FIRE_AND_FORGET_FLUSH_SECONDS = 2


class _ProgressReportingFlow(PipelineFlow):
    """Flow that reports intra-flow progress multiple times."""

    name = "progress_reporting_flow"
    estimated_minutes = 5

    async def run(
        self,
        run_id: str,
        documents: list[PubsubInputDoc],
        options: FlowOptions,
    ) -> list[PubsubOutputDoc]:
        await progress_update(0.25, "quarter done")
        await progress_update(0.50, "halfway there")
        await progress_update(0.75, "almost done")
        return [PubsubOutputDoc.derive(from_documents=(documents[0],), name="progress_out.json", content={"done": True})]


class _ProgressReportingDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """Single-flow deployment that reports intra-flow progress."""

    def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
        return [_ProgressReportingFlow()]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        return PubsubResult(success=True, doc_count=len(documents))


# Single flow with 3 progress_update calls:
# 1 run.started + flow.started + progress.STARTED + 3 PROGRESS + progress.COMPLETED + flow.completed + 1 run.completed = 9
PROGRESS_REPORTING_EVENT_COUNT = 9


class TestIntraFlowProgressPublishing:
    """Verify intra-flow progress_update() events are delivered to real Pub/Sub.

    progress_update() uses asyncio.create_task (fire-and-forget),
    so events are not awaited by run(). We yield the event loop after run() returns
    to let those tasks flush before blocking it with synchronous pull_events().
    """

    async def test_progress_update_events_arrive_on_pubsub(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """progress_update() calls inside a flow publish PROGRESS events to real Pub/Sub."""
        deployment = _ProgressReportingDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        # Yield event loop so fire-and-forget publish tasks complete before
        # we block with synchronous pull_events()
        await asyncio.sleep(FIRE_AND_FORGET_FLUSH_SECONDS)

        events = pull_events(pubsub_test_resources, expected_count=PROGRESS_REPORTING_EVENT_COUNT)
        progress = _progress_events(events)

        # Intra-flow progress events have status "progress"
        intra_flow = [e for e in progress if e.data["status"] == "progress"]
        assert len(intra_flow) == 3

        messages = [e.data["message"] for e in intra_flow]
        assert set(messages) == {"quarter done", "halfway there", "almost done"}

    async def test_progress_update_step_progress_values(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """progress_update() events carry correct step_progress fractions."""
        deployment = _ProgressReportingDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        await asyncio.sleep(FIRE_AND_FORGET_FLUSH_SECONDS)

        events = pull_events(pubsub_test_resources, expected_count=PROGRESS_REPORTING_EVENT_COUNT)
        progress = _progress_events(events)

        intra_flow = [e for e in progress if e.data["status"] == "progress"]
        step_progresses = sorted(e.data["step_progress"] for e in intra_flow)
        assert step_progresses == [0.25, 0.5, 0.75]

        # All report step=1 (single-flow deployment) and total_steps=1
        for evt in intra_flow:
            assert evt.data["step"] == 1
            assert evt.data["total_steps"] == 1

    async def test_progress_update_events_have_correct_flow_name(
        self,
        real_publisher: PublisherWithStore,
        pubsub_test_resources: PubsubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """progress_update() events carry the correct flow_name."""
        deployment = _ProgressReportingDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        await asyncio.sleep(FIRE_AND_FORGET_FLUSH_SECONDS)

        events = pull_events(pubsub_test_resources, expected_count=PROGRESS_REPORTING_EVENT_COUNT)
        progress = _progress_events(events)

        intra_flow = [e for e in progress if e.data["status"] == "progress"]
        for evt in intra_flow:
            assert evt.data["flow_name"] == "progress_reporting_flow"
