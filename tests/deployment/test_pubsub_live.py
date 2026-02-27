"""Tests against real GCP Pub/Sub (not emulator).

Skipped when PUBSUB_PROJECT_ID / PUBSUB_TOPIC_ID env vars are not set.
Runs only via: pytest -m pubsub_live
"""

# pyright: reportPrivateUsage=false, reportArgumentType=false

import json
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import uuid4

from google.api_core.exceptions import GoogleAPICallError

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)

from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store

pubsub_v1 = pytest.importorskip("google.cloud.pubsub_v1")
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

from ai_pipeline_core.deployment._pubsub import PubSubPublisher

pytestmark = pytest.mark.pubsub_live


# ---------------------------------------------------------------------------
# Live test resources
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LivePubSubResources:
    """Real GCP Pub/Sub topic and per-test subscription."""

    project_id: str
    topic_id: str
    topic_path: str
    subscription_path: str
    publisher_client: PublisherClient
    subscriber_client: SubscriberClient


@dataclass(frozen=True, slots=True)
class LiveCollectedEvent:
    """A decoded CloudEvents message from real Pub/Sub."""

    envelope: dict[str, Any]
    data: dict[str, Any]
    event_type: str
    run_id: str
    service_type: str
    seq: int


@pytest.fixture
def live_pubsub_topic():
    """Create a subscription on a real GCP Pub/Sub topic. Skip if not configured."""
    project_id = os.environ.get("PUBSUB_PROJECT_ID")
    topic_id = os.environ.get("PUBSUB_TOPIC_ID")
    if not project_id or not topic_id:
        pytest.skip("PUBSUB_PROJECT_ID and PUBSUB_TOPIC_ID not configured")

    pub_client = PublisherClient()
    sub_client = SubscriberClient()
    topic_path = pub_client.topic_path(project_id, topic_id)
    sub_id = f"test-live-{uuid4().hex[:8]}"
    sub_path = sub_client.subscription_path(project_id, sub_id)
    sub_client.create_subscription(name=sub_path, topic=topic_path)

    yield LivePubSubResources(
        project_id=project_id,
        topic_id=topic_id,
        topic_path=topic_path,
        subscription_path=sub_path,
        publisher_client=pub_client,
        subscriber_client=sub_client,
    )

    try:
        sub_client.delete_subscription(subscription=sub_path)
    except (OSError, GoogleAPICallError):
        pass


def _pull_live_events(
    resources: LivePubSubResources,
    *,
    expected_count: int,
    run_id: str,
    timeout: float = 30.0,
) -> list[LiveCollectedEvent]:
    """Pull events from a real GCP subscription, filtering by run_id.

    On a shared topic, other publishers may be active. Only events matching
    the given run_id are collected; others are acknowledged and discarded.
    """
    collected: list[LiveCollectedEvent] = []
    deadline = time.monotonic() + timeout

    while len(collected) < expected_count and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            response = resources.subscriber_client.pull(
                subscription=resources.subscription_path,
                max_messages=expected_count * 2,
                timeout=min(remaining, 10.0),
            )
        except (OSError, GoogleAPICallError):
            time.sleep(0.5)
            continue

        ack_ids = []
        for msg in response.received_messages:
            envelope = json.loads(msg.message.data.decode())
            attrs = msg.message.attributes
            ack_ids.append(msg.ack_id)
            msg_run_id = attrs.get("run_id", "")
            if msg_run_id != run_id:
                continue
            collected.append(
                LiveCollectedEvent(
                    envelope=envelope,
                    data=envelope["data"],
                    event_type=attrs.get("event_type", envelope.get("type", "")),
                    run_id=msg_run_id,
                    service_type=attrs.get("service_type", ""),
                    seq=envelope["data"].get("seq", 0),
                )
            )
        if ack_ids:
            resources.subscriber_client.acknowledge(
                subscription=resources.subscription_path,
                ack_ids=ack_ids,
            )
        if len(collected) < expected_count:
            time.sleep(0.5)

    if len(collected) < expected_count:
        event_types = [e.event_type for e in collected]
        raise AssertionError(
            f"Expected {expected_count} events for run_id={run_id}, got {len(collected)} within {timeout}s. Event types received: {event_types}"
        )

    return sorted(collected, key=lambda e: e.seq)


# ---------------------------------------------------------------------------
# Live test documents and deployment
# ---------------------------------------------------------------------------


class _LiveInputDoc(Document):
    """Input for live pubsub tests."""


class _LiveOutputDoc(Document):
    """Output for live pubsub tests."""


class _LiveResult(DeploymentResult):
    """Result for live pubsub tests."""

    item_count: int = 0


@pipeline_flow(estimated_minutes=1)
async def _live_flow(
    run_id: str,
    documents: list[_LiveInputDoc],
    flow_options: FlowOptions,
) -> list[_LiveOutputDoc]:
    """Single flow for live tests."""
    return [_LiveOutputDoc.create_root(name="live_out.json", content={"live": True}, reason="live test")]


class _LiveDeployment(PipelineDeployment[FlowOptions, _LiveResult]):
    """Single-flow deployment for live GCP tests."""

    flows: ClassVar = [_live_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _LiveResult:
        return _LiveResult(success=True, item_count=len(documents))


def _make_live_publisher(resources: LivePubSubResources) -> PubSubPublisher:
    """Create a PubSubPublisher pointed at real GCP (not emulator)."""
    # Ensure emulator env var is NOT set so we hit real GCP
    saved = os.environ.pop("PUBSUB_EMULATOR_HOST", None)
    try:
        publisher = PubSubPublisher(
            project_id=resources.project_id,
            topic_id=resources.topic_id,
            service_type="test-live-service",
        )
    finally:
        if saved is not None:
            os.environ["PUBSUB_EMULATOR_HOST"] = saved
    return publisher


async def _run_live_pipeline(
    resources: LivePubSubResources,
) -> list[LiveCollectedEvent]:
    """Run the live deployment and return collected events."""
    run_id = f"live-run-{uuid4().hex[:8]}"
    store = MemoryDocumentStore()
    set_document_store(store)
    publisher = _make_live_publisher(resources)
    try:
        deployment = _LiveDeployment()
        doc = _LiveInputDoc.create_root(name="live_input.txt", content="live data", reason="live test")
        await deployment.run(
            run_id,
            [doc],
            FlowOptions(),
            publisher=publisher,
        )
    finally:
        store.shutdown()
        set_document_store(None)

    # A single-flow deployment produces:
    # 1 started + 1 progress(started) + 1 progress(completed) + 1 completed = 4 events
    return _pull_live_events(resources, expected_count=4, run_id=run_id, timeout=30.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiveLifecycle:
    """Full lifecycle against real GCP Pub/Sub."""

    async def test_live_full_lifecycle(self, live_pubsub_topic: LivePubSubResources):
        """Successful pipeline produces started -> progress -> completed via real GCP."""
        events = await _run_live_pipeline(live_pubsub_topic)

        event_types = [e.event_type for e in events]
        assert events[0].event_type == "task.started"
        assert events[-1].event_type == "task.completed"
        assert "task.progress" in event_types
        assert "task.failed" not in event_types

        # Seq values are strictly increasing
        for i in range(1, len(events)):
            assert events[i].seq > events[i - 1].seq


class TestLiveCloudEventsFormat:
    """CloudEvents envelope validation against real GCP Pub/Sub."""

    async def test_live_cloudevents_format(self, live_pubsub_topic: LivePubSubResources):
        """All events have valid CloudEvents 1.0 fields when delivered via real GCP."""
        events = await _run_live_pipeline(live_pubsub_topic)

        for event in events:
            env = event.envelope
            assert env["specversion"] == "1.0", f"specversion={env.get('specversion')}"
            for field_name in ("id", "type", "source", "time", "subject", "datacontenttype"):
                assert field_name in env, f"Missing CloudEvents field: {field_name}"
            assert env["datacontenttype"] == "application/json"
            assert isinstance(env["data"], dict)
            assert env["data"]["run_id"] == env["subject"]


class TestLiveChainContext:
    """Chain context structure validation against real GCP Pub/Sub."""

    async def test_live_chain_context(self, live_pubsub_topic: LivePubSubResources):
        """Completed event chain_context has required fields via real GCP."""
        events = await _run_live_pipeline(live_pubsub_topic)

        completed = [e for e in events if e.event_type == "task.completed"]
        assert len(completed) == 1, f"Expected 1 completed event, got {len(completed)}"

        chain_ctx = completed[0].data["chain_context"]
        assert chain_ctx["version"] == 1
        assert "run_scope" in chain_ctx
        assert isinstance(chain_ctx["run_scope"], str)
        assert "output_document_refs" in chain_ctx
        assert isinstance(chain_ctx["output_document_refs"], list)
        assert len(chain_ctx["output_document_refs"]) > 0
