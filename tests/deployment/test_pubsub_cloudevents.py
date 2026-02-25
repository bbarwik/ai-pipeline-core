"""Tests for CloudEvents 1.0 envelope format in real Pub/Sub messages.

Validates that messages published to Pub/Sub conform to the CloudEvents spec,
have correct attributes, unique IDs, monotonic sequences, parseable timestamps,
and handle non-primitive type serialization via json.dumps(default=str).
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false, reportUnusedClass=false

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import ClassVar
from uuid import UUID

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)
from ai_pipeline_core.deployment._types import EventType

from .conftest import (
    PubSubTestResources,
    PublisherWithStore,
    PubsubInputDoc,
    PubsubOutputDoc,
    TwoFlowDeployment,
    assert_seq_monotonic,
    assert_valid_cloudevent,
    make_input_doc,
    pull_events,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub

# 2-flow success: 1 started + 4 progress (STARTED/COMPLETED x 2 flows) + 1 completed = 6
TWO_FLOW_SUCCESS_EVENT_COUNT = 6


class TestCloudEventsFormat:
    """CloudEvents envelope format validation on real Pub/Sub messages."""

    async def test_all_events_have_valid_cloudevents_fields(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store,
    ):
        """Every event has required CloudEvents 1.0 fields."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, run_id="ce-fields-test")

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)
        for event in events:
            assert_valid_cloudevent(event)

    async def test_message_attributes_match_envelope(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store,
    ):
        """Pub/Sub message attributes match corresponding CloudEvents envelope fields."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, run_id="ce-attrs-test")

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)
        for event in events:
            assert event.service_type == "test-service", f"service_type attribute should be 'test-service', got '{event.service_type}'"
            assert event.event_type == event.envelope["type"], (
                f"event_type attribute '{event.event_type}' should match envelope type '{event.envelope['type']}'"
            )
            assert event.run_id == event.envelope["subject"], f"run_id attribute '{event.run_id}' should match envelope subject '{event.envelope['subject']}'"

    async def test_event_ids_are_unique_uuids(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store,
    ):
        """All CloudEvents id fields are unique valid UUIDs."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, run_id="ce-ids-test")

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)
        ids = set()
        for event in events:
            event_id = event.envelope["id"]
            # Verify it parses as a valid UUID
            parsed = UUID(event_id)
            assert str(parsed) == event_id
            ids.add(event_id)

        assert len(ids) == TWO_FLOW_SUCCESS_EVENT_COUNT, f"Expected {TWO_FLOW_SUCCESS_EVENT_COUNT} unique event IDs, got {len(ids)}"

    async def test_seq_is_monotonically_increasing(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store,
    ):
        """All events have strictly increasing seq values when sorted by seq."""
        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, run_id="ce-seq-test")

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)
        assert_seq_monotonic(events)

    async def test_timestamps_are_parseable_iso8601(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store,
    ):
        """All event timestamps are valid ISO 8601 and fall within the test execution window."""
        margin = timedelta(seconds=30)
        test_start = datetime.now(UTC) - margin

        deployment = TwoFlowDeployment()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, run_id="ce-time-test")

        test_end = datetime.now(UTC) + margin

        events = pull_events(pubsub_topic, expected_count=TWO_FLOW_SUCCESS_EVENT_COUNT)
        for event in events:
            time_str = event.envelope["time"]
            parsed = datetime.fromisoformat(time_str)
            assert parsed >= test_start, f"Event time {time_str} is before test start {test_start.isoformat()}"
            assert parsed <= test_end, f"Event time {time_str} is after test end {test_end.isoformat()}"


# ---------------------------------------------------------------------------
# Non-primitive type serialization test
# ---------------------------------------------------------------------------


class _Priority(StrEnum):
    HIGH = "high"
    LOW = "low"


class _SerializationResult(DeploymentResult):
    """Result with non-primitive fields to test default=str serialization."""

    created_at: datetime = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
    task_uuid: UUID = UUID("12345678-1234-5678-1234-567812345678")
    priority: _Priority = _Priority.HIGH


@pipeline_flow(estimated_minutes=1)
async def _serialization_flow(
    run_id: str,
    documents: list[PubsubInputDoc],
    flow_options: FlowOptions,
) -> list[PubsubOutputDoc]:
    """Single flow for serialization test."""
    return [PubsubOutputDoc.create_root(name="ser_out.json", content={"serialized": True}, reason="test")]


class _SerializationDeployment(PipelineDeployment[FlowOptions, _SerializationResult]):
    """Deployment returning a result with non-primitive types."""

    flows: ClassVar = [_serialization_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _SerializationResult:
        return _SerializationResult(success=True)


class TestNonPrimitiveTypeSerialization:
    """Verify json.dumps(default=str) handles non-primitive types in result."""

    async def test_default_str_serialization_of_nonprimitive_types(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store,
    ):
        """datetime, UUID, and Enum fields in DeploymentResult are stringified in the completed event."""
        # Single-flow: 1 started + 2 progress (STARTED/COMPLETED) + 1 completed = 4
        expected_count = 4

        deployment = _SerializationDeployment()
        doc = make_input_doc()
        await run_pipeline(deployment, real_publisher.publisher, pubsub_memory_store, run_id="ser-test", docs=[doc])

        events = pull_events(pubsub_topic, expected_count=expected_count)
        completed_events = [e for e in events if e.event_type == EventType.COMPLETED]
        assert len(completed_events) == 1

        result = completed_events[0].data["result"]
        assert result["success"] is True

        # model_dump(mode='python') returns Python objects (datetime, UUID, Enum).
        # json.dumps(default=str) in _build_envelope stringifies them for transport.
        created_at_str = str(result["created_at"])
        datetime.fromisoformat(created_at_str)  # Must be parseable as ISO 8601
        assert "2025-06-15" in created_at_str, f"datetime should contain expected date, got: {created_at_str}"

        task_uuid_str = str(result["task_uuid"])
        UUID(task_uuid_str)  # Must be parseable as UUID
        assert task_uuid_str == "12345678-1234-5678-1234-567812345678", f"UUID should roundtrip exactly, got: {task_uuid_str}"

        priority_str = str(result["priority"])
        assert priority_str == "high", f"StrEnum should serialize to its value, got: {priority_str}"
