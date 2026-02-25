"""Tests for resume/cached flow behavior with real Pub/Sub emulator.

Verifies that resumed (cached) flows still publish correct progress events
with status=CACHED and proper step numbering.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false

import pytest
from google.api_core.exceptions import GoogleAPICallError

from ai_pipeline_core import FlowOptions
from ai_pipeline_core.deployment._publishers import NoopPublisher
from ai_pipeline_core.deployment._types import EventType
from ai_pipeline_core.deployment.contract import FlowStatus

from .conftest import (
    CollectedEvent,
    PubSubTestResources,
    PublisherWithStore,
    SingleFlowDeployment,
    ThreeFlowDeployment,
    TwoFlowDeployment,
    _flow_executions,
    make_input_doc,
    pull_events,
    run_pipeline,
)
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.deployment._pubsub import PubSubPublisher
from ai_pipeline_core.deployment._task_results import MemoryTaskResultStore

pytestmark = pytest.mark.pubsub


def _progress_events(events: list[CollectedEvent]) -> list[CollectedEvent]:
    """Filter to only progress events."""
    return [e for e in events if e.event_type == EventType.PROGRESS]


def _make_fresh_publisher(pubsub_topic: PubSubTestResources) -> PublisherWithStore:
    """Create a new PubSubPublisher + MemoryTaskResultStore for the second run."""
    result_store = MemoryTaskResultStore()
    topic_id = pubsub_topic.topic_path.split("/")[-1]
    publisher = PubSubPublisher(
        project_id=pubsub_topic.project_id,
        topic_id=topic_id,
        service_type="test-service",
        result_store=result_store,
    )
    return PublisherWithStore(publisher=publisher, result_store=result_store)


class TestResumedFlowCachedStatus:
    """Verify that resumed flows publish progress events with status=CACHED."""

    async def test_resumed_flow_publishes_cached_status(
        self,
        pubsub_emulator: str,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Run TwoFlowDeployment twice; second run's progress events all have status=CACHED."""
        deployment = TwoFlowDeployment()
        doc = make_input_doc()

        # First run: use NoopPublisher (we don't care about its events)
        await run_pipeline(deployment, NoopPublisher(), pubsub_memory_store, docs=[doc])
        assert len(_flow_executions) == 2  # both flows executed

        # Create a fresh topic + subscription for the second run
        from .conftest import PubSubTestResources
        from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
        from uuid import uuid4

        pub_client = PublisherClient()
        sub_client = SubscriberClient()
        topic_id = f"test-resume-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path("test-project", topic_id)
        sub_id = f"test-resume-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path("test-project", sub_id)
        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        resources = PubSubTestResources(
            project_id="test-project",
            topic_path=topic_path,
            subscription_path=sub_path,
            publisher_client=pub_client,
            subscriber_client=sub_client,
        )

        try:
            second_pub = _make_fresh_publisher(resources)

            # Second run: same run_id, same docs, same store (flows should be cached)
            _flow_executions.clear()
            await run_pipeline(deployment, second_pub.publisher, pubsub_memory_store, docs=[doc])
            assert len(_flow_executions) == 0  # no flows executed (all cached)

            # 2 cached flows: 1 started + 2 progress (CACHED x2) + 1 completed = 4
            events = pull_events(resources, expected_count=4)
            progress = _progress_events(events)

            assert len(progress) == 2
            for evt in progress:
                assert evt.data["status"] == FlowStatus.CACHED

            # Step numbering is correct
            steps = sorted(evt.data["step"] for evt in progress)
            assert steps == [1, 2]
        finally:
            try:
                sub_client.delete_subscription(subscription=sub_path)
            except (OSError, GoogleAPICallError):
                pass
            try:
                pub_client.delete_topic(topic=topic_path)
            except (OSError, GoogleAPICallError):
                pass


class TestResumedPipelineLifecycle:
    """Verify that fully cached pipelines still publish started + completed."""

    async def test_resumed_pipeline_still_publishes_started_and_completed(
        self,
        pubsub_emulator: str,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """SingleFlowDeployment run twice: second run still has task.started and task.completed."""
        deployment = SingleFlowDeployment()
        doc = make_input_doc()

        # First run: use NoopPublisher
        await run_pipeline(deployment, NoopPublisher(), pubsub_memory_store, docs=[doc])
        assert len(_flow_executions) == 1

        # Create fresh topic for second run
        from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
        from uuid import uuid4

        pub_client = PublisherClient()
        sub_client = SubscriberClient()
        topic_id = f"test-resume-single-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path("test-project", topic_id)
        sub_id = f"test-resume-single-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path("test-project", sub_id)
        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        resources = PubSubTestResources(
            project_id="test-project",
            topic_path=topic_path,
            subscription_path=sub_path,
            publisher_client=pub_client,
            subscriber_client=sub_client,
        )

        try:
            second_pub = _make_fresh_publisher(resources)

            _flow_executions.clear()
            await run_pipeline(deployment, second_pub.publisher, pubsub_memory_store, docs=[doc])
            assert len(_flow_executions) == 0  # cached

            # 1 cached flow: 1 started + 1 progress (CACHED) + 1 completed = 3
            events = pull_events(resources, expected_count=3)

            started = [e for e in events if e.event_type == EventType.STARTED]
            completed = [e for e in events if e.event_type == EventType.COMPLETED]
            progress = _progress_events(events)

            assert len(started) == 1
            assert len(completed) == 1
            assert len(progress) == 1
            assert progress[0].data["status"] == FlowStatus.CACHED
        finally:
            try:
                sub_client.delete_subscription(subscription=sub_path)
            except (OSError, GoogleAPICallError):
                pass
            try:
                pub_client.delete_topic(topic=topic_path)
            except (OSError, GoogleAPICallError):
                pass


class TestPartialResumeMix:
    """Verify mixed cached + executed flows publish correct statuses."""

    async def test_partial_resume_mix_of_cached_and_executed(
        self,
        pubsub_emulator: str,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """3-flow deployment: flows A and B have pre-saved FlowCompletion records,
        flow C executes. Assert A+B progress events are CACHED, C has STARTED -> COMPLETED.
        """
        deployment = ThreeFlowDeployment()
        doc = make_input_doc()

        # Pre-save FlowCompletion records for flow_a and flow_b (same pattern as test_resume_logic.py)
        # First, run a full pipeline to populate the store with documents and completions
        await run_pipeline(deployment, NoopPublisher(), pubsub_memory_store, docs=[doc])
        assert set(_flow_executions) == {"flow_a", "flow_b", "flow_c"}

        # Delete flow_c's completion record so it re-executes on second run
        from ai_pipeline_core.deployment.base import _compute_run_scope

        run_scope = _compute_run_scope("test-run", [doc], FlowOptions())
        del pubsub_memory_store._flow_completions[run_scope, "pubsub_flow_c"]

        # Create fresh topic for second run
        from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
        from uuid import uuid4

        pub_client = PublisherClient()
        sub_client = SubscriberClient()
        topic_id = f"test-partial-resume-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path("test-project", topic_id)
        sub_id = f"test-partial-resume-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path("test-project", sub_id)
        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        resources = PubSubTestResources(
            project_id="test-project",
            topic_path=topic_path,
            subscription_path=sub_path,
            publisher_client=pub_client,
            subscriber_client=sub_client,
        )

        try:
            second_pub = _make_fresh_publisher(resources)

            _flow_executions.clear()
            await run_pipeline(deployment, second_pub.publisher, pubsub_memory_store, docs=[doc])

            # Only flow_c should have re-executed
            assert _flow_executions == ["flow_c"]

            # 2 cached + 1 executed: 1 started + 4 progress (2 CACHED + STARTED + COMPLETED) + 1 completed = 6
            events = pull_events(resources, expected_count=6)
            progress = _progress_events(events)

            assert len(progress) == 4

            # Group by flow_name
            by_flow: dict[str, list[CollectedEvent]] = {}
            for evt in progress:
                by_flow.setdefault(evt.data["flow_name"], []).append(evt)

            # Flows A and B: single CACHED event each
            for name in ("pubsub_flow_a", "pubsub_flow_b"):
                flow_events = by_flow[name]
                assert len(flow_events) == 1
                assert flow_events[0].data["status"] == FlowStatus.CACHED

            # Flow C: STARTED then COMPLETED
            flow_c_events = by_flow["pubsub_flow_c"]
            assert len(flow_c_events) == 2
            statuses = [evt.data["status"] for evt in flow_c_events]
            assert statuses == [FlowStatus.STARTED, FlowStatus.COMPLETED]
        finally:
            try:
                sub_client.delete_subscription(subscription=sub_path)
            except (OSError, GoogleAPICallError):
                pass
            try:
                pub_client.delete_topic(topic=topic_path)
            except (OSError, GoogleAPICallError):
                pass
