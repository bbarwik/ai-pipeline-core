"""Tests for resume/cached flow behavior with real Pub/Sub emulator.

Verifies that resumed (cached) flows still publish correct progress events
with status=CACHED and proper step numbering.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false

import pytest
from google.api_core.exceptions import GoogleAPICallError

from ai_pipeline_core import FlowOptions
from ai_pipeline_core.deployment._pubsub import PubSubPublisher
from ai_pipeline_core.deployment._types import EventType, _NoopPublisher
from ai_pipeline_core.deployment._contract import FlowStatus
from ai_pipeline_core.document_store._memory import MemoryDocumentStore

from .conftest import (
    CollectedEvent,
    PubsubTestResources,
    PublisherWithStore,
    SingleStageDeployment,
    ThreeStageDeployment,
    TwoStageDeployment,
    _flow_executions,
    make_input_doc,
    pull_events,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub


def _progress_events(events: list[CollectedEvent]) -> list[CollectedEvent]:
    """Filter to only progress events."""
    return [e for e in events if e.event_type == EventType.PROGRESS]


def _make_fresh_publisher(pubsub_test_resources: PubsubTestResources) -> PublisherWithStore:
    """Create a new PubSubPublisher for the second run."""
    topic_id = pubsub_test_resources.topic_path.split("/")[-1]
    publisher = PubSubPublisher(
        project_id=pubsub_test_resources.project_id,
        topic_id=topic_id,
        service_type="test-service",
    )
    return PublisherWithStore(publisher=publisher)


class TestResumedFlowCachedStatus:
    """Verify that resumed flows publish progress events with status=CACHED."""

    async def test_resumed_flow_publishes_cached_status(
        self,
        pubsub_emulator: str,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Run TwoStageDeployment twice; second run's progress events all have status=CACHED."""
        deployment = TwoStageDeployment()
        doc = make_input_doc()

        # First run: use _NoopPublisher (we don't care about its events)
        await run_pipeline(deployment, _NoopPublisher(), pubsub_memory_store, docs=[doc])
        assert len(_flow_executions) == 2  # both flows executed

        # Create a fresh topic + subscription for the second run
        from uuid import uuid4

        from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

        pub_client = PublisherClient()
        sub_client = SubscriberClient()
        topic_id = f"test-resume-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path("test-project", topic_id)
        sub_id = f"test-resume-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path("test-project", sub_id)
        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        resources = PubsubTestResources(
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

            # 2 cached flows: 1 run.started + 2*(flow.skipped + progress.CACHED) + 1 run.completed = 6
            events = pull_events(resources, expected_count=6)
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
        """SingleStageDeployment run twice: second run still has run.started and run.completed."""
        deployment = SingleStageDeployment()
        doc = make_input_doc()

        # First run: use _NoopPublisher
        await run_pipeline(deployment, _NoopPublisher(), pubsub_memory_store, docs=[doc])
        assert len(_flow_executions) == 1

        # Create fresh topic for second run
        from uuid import uuid4

        from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

        pub_client = PublisherClient()
        sub_client = SubscriberClient()
        topic_id = f"test-resume-single-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path("test-project", topic_id)
        sub_id = f"test-resume-single-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path("test-project", sub_id)
        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        resources = PubsubTestResources(
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

            # 1 cached flow: 1 run.started + (flow.skipped + progress.CACHED) + 1 run.completed = 4
            events = pull_events(resources, expected_count=4)

            started = [e for e in events if e.event_type == EventType.RUN_STARTED]
            completed = [e for e in events if e.event_type == EventType.RUN_COMPLETED]
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
        deployment = ThreeStageDeployment()
        doc = make_input_doc()

        # First run to populate the store with documents and completions
        await run_pipeline(deployment, _NoopPublisher(), pubsub_memory_store, docs=[doc])
        assert set(_flow_executions) == {"flow_a", "flow_b", "flow_c"}

        # Delete flow_c's completion record so it re-executes on second run
        # completion_name format: f"{flow_name}:{step}" where flow_c is step 3
        from ai_pipeline_core.deployment.base import _compute_run_scope

        run_scope = _compute_run_scope("test-run", [doc], FlowOptions())
        del pubsub_memory_store._flow_completions[run_scope, "chain_output_to_final:3"]

        # Create fresh topic for second run
        from uuid import uuid4

        from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

        pub_client = PublisherClient()
        sub_client = SubscriberClient()
        topic_id = f"test-partial-resume-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path("test-project", topic_id)
        sub_id = f"test-partial-resume-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path("test-project", sub_id)
        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        resources = PubsubTestResources(
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

            # 2 cached + 1 executed:
            # 1 run.started + 2*(flow.skipped + progress.CACHED) + (flow.started + progress.STARTED
            # + progress.COMPLETED + flow.completed) + 1 run.completed = 10
            events = pull_events(resources, expected_count=10)
            progress = _progress_events(events)

            assert len(progress) == 4

            # Group by flow_name
            by_flow: dict[str, list[CollectedEvent]] = {}
            for evt in progress:
                by_flow.setdefault(evt.data["flow_name"], []).append(evt)

            # Flows A and B: single CACHED event each
            for name in ("chain_input_to_middle", "chain_middle_to_output"):
                flow_events = by_flow[name]
                assert len(flow_events) == 1
                assert flow_events[0].data["status"] == FlowStatus.CACHED

            # Flow C: STARTED then COMPLETED
            flow_c_events = by_flow["chain_output_to_final"]
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
