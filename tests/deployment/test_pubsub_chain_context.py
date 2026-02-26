"""Chain context integration tests via real Pub/Sub emulator.

Verifies chain_context structure, output_document_refs correctness,
and chain_context integrity after full and partial resume.
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false

from typing import Any
from uuid import uuid4

import pytest
from google.api_core.exceptions import GoogleAPICallError

from ai_pipeline_core.deployment._pubsub import PubSubPublisher

from ai_pipeline_core.document_store._memory import MemoryDocumentStore

from .conftest import (
    CollectedEvent,
    PubSubTestResources,
    PubsubFinalDoc,
    PubsubMiddleDoc,
    PubsubOutputDoc,
    PublisherWithStore,
    ThreeFlowDeployment,
    TwoFlowDeployment,
    make_input_doc,
    pull_events,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub

# 2-flow success: 1 started + 4 progress (2 per flow: STARTED+COMPLETED) + 1 completed
TWO_FLOW_EVENT_COUNT = 6

# 2-flow full resume: 1 started + 2 CACHED progress + 1 completed
TWO_FLOW_RESUME_EVENT_COUNT = 4

# 3-flow success: 1 started + 6 progress + 1 completed
THREE_FLOW_EVENT_COUNT = 8

# 3-flow full resume: 1 started + 3 CACHED progress + 1 completed
THREE_FLOW_RESUME_EVENT_COUNT = 5


def _get_completed_event(events: list[CollectedEvent]) -> CollectedEvent:
    """Return the task.completed event (highest seq)."""
    completed = [e for e in events if e.event_type == "task.completed"]
    assert len(completed) == 1, f"Expected 1 completed event, got {len(completed)}"
    return completed[0]


def _chain_context(event: CollectedEvent) -> dict[str, Any]:
    """Extract chain_context from a completed event's data."""
    ctx = event.data.get("chain_context")
    assert ctx is not None, "chain_context missing from completed event data"
    return ctx


def _make_second_publisher(pubsub_topic: PubSubTestResources) -> tuple[PubSubTestResources, PublisherWithStore]:
    """Create a second topic+subscription and publisher for collecting only a second run's events."""
    from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

    pub_client = PublisherClient()
    sub_client = SubscriberClient()

    topic_id = f"test-events-{uuid4().hex[:8]}"
    topic_path = pub_client.topic_path(pubsub_topic.project_id, topic_id)
    sub_id = f"test-sub-{uuid4().hex[:8]}"
    sub_path = sub_client.subscription_path(pubsub_topic.project_id, sub_id)

    pub_client.create_topic(name=topic_path)
    sub_client.create_subscription(name=sub_path, topic=topic_path)

    resources = PubSubTestResources(
        project_id=pubsub_topic.project_id,
        topic_path=topic_path,
        subscription_path=sub_path,
        publisher_client=pub_client,
        subscriber_client=sub_client,
    )

    publisher = PubSubPublisher(
        project_id=pubsub_topic.project_id,
        topic_id=topic_id,
        service_type="test-service",
    )

    return resources, PublisherWithStore(publisher=publisher)


class TestChainContext:
    """Chain context structure and correctness tests."""

    async def test_chain_context_contains_required_fields(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Completed event chain_context has version, run_scope, and output_document_refs."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_EVENT_COUNT)
        completed = _get_completed_event(events)
        ctx = _chain_context(completed)

        assert ctx["version"] == 1
        assert isinstance(ctx["run_scope"], str)
        assert len(ctx["run_scope"]) > 0
        assert isinstance(ctx["output_document_refs"], list)

    async def test_output_document_refs_point_to_last_flow_outputs(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """output_document_refs contains SHA256s of last flow's outputs (PubsubOutputDoc), not intermediate docs."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_EVENT_COUNT)
        ctx = _chain_context(_get_completed_event(events))
        output_refs = ctx["output_document_refs"]

        assert len(output_refs) > 0, "output_document_refs should not be empty"

        # Load all documents from the store by type to cross-reference
        run_scope = ctx["run_scope"]
        middle_docs = await pubsub_memory_store.load(run_scope, [PubsubMiddleDoc])
        output_docs = await pubsub_memory_store.load(run_scope, [PubsubOutputDoc])

        middle_sha256s = {d.sha256 for d in middle_docs}
        output_sha256s = {d.sha256 for d in output_docs}

        # All refs should be from PubsubOutputDoc (last flow)
        for ref in output_refs:
            assert ref in output_sha256s, f"SHA256 {ref} not in PubsubOutputDoc set"
            assert ref not in middle_sha256s, f"SHA256 {ref} is a PubsubMiddleDoc, should only be PubsubOutputDoc"

    async def test_chain_context_document_refs_exist_in_store(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """Every SHA256 in output_document_refs exists in the document store."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store)

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_EVENT_COUNT)
        ctx = _chain_context(_get_completed_event(events))
        output_refs = ctx["output_document_refs"]

        assert len(output_refs) > 0

        existing = await pubsub_memory_store.check_existing(output_refs)
        for ref in output_refs:
            assert ref in existing, f"SHA256 {ref} from output_document_refs not found in document store"

    async def test_chain_context_correct_after_full_resume(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """After full resume (all flows cached), chain_context matches first run's last flow outputs."""
        deployment = TwoFlowDeployment()
        input_doc = make_input_doc()

        # First run — all flows execute
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, docs=[input_doc])
        first_events = pull_events(pubsub_topic, expected_count=TWO_FLOW_EVENT_COUNT)
        first_ctx = _chain_context(_get_completed_event(first_events))
        first_refs = first_ctx["output_document_refs"]
        first_run_scope = first_ctx["run_scope"]

        # Second run — same store (has flow completions), new topic/publisher for clean event collection
        second_resources, second_pub = _make_second_publisher(pubsub_topic)
        try:
            await run_pipeline(deployment, second_pub.publisher, pubsub_memory_store, docs=[input_doc])
            second_events = pull_events(second_resources, expected_count=TWO_FLOW_RESUME_EVENT_COUNT)
            second_ctx = _chain_context(_get_completed_event(second_events))

            # Same inputs + options → same run_scope
            assert second_ctx["run_scope"] == first_run_scope

            # output_document_refs should match: cached flows read the same completion records
            assert sorted(second_ctx["output_document_refs"]) == sorted(first_refs)
        finally:
            try:
                second_resources.subscriber_client.delete_subscription(subscription=second_resources.subscription_path)
            except (OSError, GoogleAPICallError):
                pass
            try:
                second_resources.publisher_client.delete_topic(topic=second_resources.topic_path)
            except (OSError, GoogleAPICallError):
                pass

    async def test_chain_context_correct_after_full_resume_3flow(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
    ):
        """After full resume on 3-flow pipeline (all cached), output_document_refs points to flow C's outputs."""
        deployment = ThreeFlowDeployment()
        input_doc = make_input_doc()

        # First run — all 3 flows execute
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, docs=[input_doc])
        pull_events(pubsub_topic, expected_count=THREE_FLOW_EVENT_COUNT)

        # Second run — all 3 flows cached, new topic for clean collection
        second_resources, second_pub = _make_second_publisher(pubsub_topic)
        try:
            await run_pipeline(deployment, second_pub.publisher, pubsub_memory_store, docs=[input_doc])
            second_events = pull_events(second_resources, expected_count=THREE_FLOW_RESUME_EVENT_COUNT)
            ctx = _chain_context(_get_completed_event(second_events))
            output_refs = ctx["output_document_refs"]

            assert len(output_refs) > 0

            # Verify refs point to PubsubFinalDoc (flow C output), not intermediate types
            run_scope = ctx["run_scope"]
            final_docs = await pubsub_memory_store.load(run_scope, [PubsubFinalDoc])
            output_docs = await pubsub_memory_store.load(run_scope, [PubsubOutputDoc])
            middle_docs = await pubsub_memory_store.load(run_scope, [PubsubMiddleDoc])

            final_sha256s = {d.sha256 for d in final_docs}
            output_sha256s = {d.sha256 for d in output_docs}
            middle_sha256s = {d.sha256 for d in middle_docs}

            for ref in output_refs:
                assert ref in final_sha256s, f"SHA256 {ref} not in PubsubFinalDoc set"
                assert ref not in output_sha256s, f"SHA256 {ref} is a PubsubOutputDoc, expected PubsubFinalDoc"
                assert ref not in middle_sha256s, f"SHA256 {ref} is a PubsubMiddleDoc, expected PubsubFinalDoc"
        finally:
            try:
                second_resources.subscriber_client.delete_subscription(subscription=second_resources.subscription_path)
            except (OSError, GoogleAPICallError):
                pass
            try:
                second_resources.publisher_client.delete_topic(topic=second_resources.topic_path)
            except (OSError, GoogleAPICallError):
                pass
