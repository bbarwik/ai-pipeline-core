"""Deployment test fixtures.

Prefect test harness and logging suppression are provided by tests/conftest.py.
Controllable test deployments and Pub/Sub emulator fixtures for integration tests.

Pub/Sub emulator fixtures (pubsub_emulator, pubsub_topic, real_publisher) require
testcontainers (dev dependency) and are conditionally registered.
"""

# pyright: reportPrivateUsage=false, reportUnusedClass=false, reportArgumentType=false

import asyncio
import json
import os
import time
from dataclasses import dataclass
from collections.abc import Generator
from typing import Any, ClassVar
from uuid import uuid4

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)

from ai_pipeline_core.deployment import DeploymentContext
from ai_pipeline_core.document_store._memory import MemoryDocumentStore
from ai_pipeline_core.document_store._protocol import set_document_store


# ---------------------------------------------------------------------------
# Controllable test deployments (no external deps — always available)
# ---------------------------------------------------------------------------

# Gate for flow control in heartbeat tests
_flow_1_gate = asyncio.Event()
_flow_1_gate.set()  # Open by default

# Execution tracker
_flow_executions: list[str] = []


class PubsubInputDoc(Document):
    """Input document for pubsub integration tests."""


class PubsubMiddleDoc(Document):
    """Intermediate document for pubsub integration tests."""


class PubsubOutputDoc(Document):
    """Output document for pubsub integration tests."""


class PubsubResult(DeploymentResult):
    """Result for pubsub integration tests."""

    doc_count: int = 0


@pipeline_flow(estimated_minutes=10)
async def pubsub_flow_1(
    run_id: str,
    documents: list[PubsubInputDoc],
    flow_options: FlowOptions,
) -> list[PubsubMiddleDoc]:
    """First flow — controllable via _flow_1_gate."""
    _flow_executions.append("flow_1")
    await _flow_1_gate.wait()
    return [PubsubMiddleDoc.create_root(name="middle.json", content={"step": 1}, reason="test")]


@pipeline_flow(estimated_minutes=20)
async def pubsub_flow_2(
    run_id: str,
    documents: list[PubsubMiddleDoc],
    flow_options: FlowOptions,
) -> list[PubsubOutputDoc]:
    """Second flow — straightforward."""
    _flow_executions.append("flow_2")
    return [PubsubOutputDoc.create_root(name="output.json", content={"step": 2}, reason="test")]


class TwoFlowDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """Standard 2-flow deployment for most integration tests."""

    flows: ClassVar = [pubsub_flow_1, pubsub_flow_2]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        """Build result."""
        return PubsubResult(success=True, doc_count=len(documents))


# --- Failing variants ---


@pipeline_flow(estimated_minutes=5)
async def pubsub_failing_flow(
    run_id: str,
    documents: list[PubsubMiddleDoc],
    flow_options: FlowOptions,
) -> list[PubsubOutputDoc]:
    """Flow that always raises RuntimeError."""
    _flow_executions.append("failing_flow")
    raise RuntimeError("deliberate test failure")


class FailingSecondFlowDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """Deployment where second flow fails."""

    flows: ClassVar = [pubsub_flow_1, pubsub_failing_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        """Build result."""
        return PubsubResult(success=False, error="failed")


# --- Single flow for simple resume tests ---


@pipeline_flow(estimated_minutes=5)
async def pubsub_single_flow(
    run_id: str,
    documents: list[PubsubInputDoc],
    flow_options: FlowOptions,
) -> list[PubsubOutputDoc]:
    """Single flow for resume tests."""
    _flow_executions.append("single_flow")
    return [PubsubOutputDoc.create_root(name="single_out.json", content={"single": True}, reason="test")]


class SingleFlowDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """Single-flow deployment for simple resume tests."""

    flows: ClassVar = [pubsub_single_flow]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        """Build result."""
        return PubsubResult(success=True, doc_count=len(documents))


# --- 3-flow for progress and resume tests ---


class PubsubFinalDoc(Document):
    """Final document for 3-flow pipeline."""


@pipeline_flow(estimated_minutes=10)
async def pubsub_flow_a(
    run_id: str,
    documents: list[PubsubInputDoc],
    flow_options: FlowOptions,
) -> list[PubsubMiddleDoc]:
    """Flow A in 3-flow pipeline."""
    _flow_executions.append("flow_a")
    return [PubsubMiddleDoc.create_root(name="a_out.json", content={"a": 1}, reason="test")]


@pipeline_flow(estimated_minutes=20)
async def pubsub_flow_b(
    run_id: str,
    documents: list[PubsubMiddleDoc],
    flow_options: FlowOptions,
) -> list[PubsubOutputDoc]:
    """Flow B in 3-flow pipeline."""
    _flow_executions.append("flow_b")
    return [PubsubOutputDoc.create_root(name="b_out.json", content={"b": 2}, reason="test")]


@pipeline_flow(estimated_minutes=30)
async def pubsub_flow_c(
    run_id: str,
    documents: list[PubsubOutputDoc],
    flow_options: FlowOptions,
) -> list[PubsubFinalDoc]:
    """Flow C in 3-flow pipeline."""
    _flow_executions.append("flow_c")
    return [PubsubFinalDoc.create_root(name="c_out.json", content={"c": 3}, reason="test")]


class ThreeFlowDeployment(PipelineDeployment[FlowOptions, PubsubResult]):
    """3-flow deployment for progress and resume tests."""

    flows: ClassVar = [pubsub_flow_a, pubsub_flow_b, pubsub_flow_c]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> PubsubResult:
        """Build result."""
        return PubsubResult(success=True, doc_count=len(documents))


# ---------------------------------------------------------------------------
# Common fixtures (no external deps)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_flow_state() -> Generator[None, None, None]:  # pyright: ignore[reportUnusedFunction]
    """Clear flow execution tracker and recreate gate before each test.

    asyncio.Event binds to the event loop on first .wait() call. Since
    pytest-asyncio creates a new loop per test, we must recreate the Event.
    """
    global _flow_1_gate
    _flow_executions.clear()
    _flow_1_gate = asyncio.Event()
    _flow_1_gate.set()
    yield
    _flow_1_gate.set()


@pytest.fixture
def flow_gate() -> asyncio.Event:
    """Return the current flow_1 gate Event (fresh per test)."""
    return _flow_1_gate


@pytest.fixture
def pubsub_memory_store() -> Generator[MemoryDocumentStore, None, None]:
    """Provide a MemoryDocumentStore and set it as global singleton. Cleans up after."""
    store = MemoryDocumentStore()
    set_document_store(store)
    yield store
    store.shutdown()
    set_document_store(None)


def make_input_doc(name: str = "input.txt", content: str = "test data") -> PubsubInputDoc:
    """Create a standard input document for tests."""
    return PubsubInputDoc.create_root(name=name, content=content, reason="test input")


async def run_pipeline(
    deployment: PipelineDeployment[FlowOptions, PubsubResult],
    publisher: Any,
    store: MemoryDocumentStore,
    *,
    run_id: str = "test-run",
    docs: list[Document] | None = None,
    start_step: int = 1,
    end_step: int | None = None,
    task_result_store: Any = None,
) -> PubsubResult:
    """Run a pipeline with the given publisher and store."""
    if docs is None:
        docs = [make_input_doc()]
    kwargs: dict[str, Any] = {"start_step": start_step}
    if end_step is not None:
        kwargs["end_step"] = end_step
    if task_result_store is not None:
        kwargs["task_result_store"] = task_result_store
    return await deployment.run(run_id, docs, FlowOptions(), DeploymentContext(), publisher=publisher, **kwargs)


# ---------------------------------------------------------------------------
# Pub/Sub emulator fixtures — testcontainers is optional (dev dependency)
# ---------------------------------------------------------------------------

from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

from ai_pipeline_core.deployment._pubsub import PubSubPublisher

DockerContainer: type | None = None
wait_for_logs: Any = None

_testcontainers_available = False
try:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    _testcontainers_available = True
except ImportError:
    pass


# Data types used by test files (always importable, no external deps)


@dataclass(frozen=True, slots=True)
class CollectedEvent:
    """A decoded CloudEvents message from Pub/Sub."""

    envelope: dict[str, Any]
    data: dict[str, Any]
    event_type: str
    run_id: str
    service_type: str
    seq: int


@dataclass(frozen=True, slots=True)
class PubSubTestResources:
    """Topic and subscription created for a single test."""

    project_id: str
    topic_path: str
    subscription_path: str
    publisher_client: PublisherClient
    subscriber_client: SubscriberClient


@dataclass
class PublisherWithStore:
    """PubSubPublisher for test assertions."""

    publisher: PubSubPublisher


def _decode_message(message: Any) -> CollectedEvent:
    """Decode a Pub/Sub message into a CollectedEvent."""
    envelope = json.loads(message.data.decode())
    data = envelope["data"]
    attrs = message.attributes
    return CollectedEvent(
        envelope=envelope,
        data=data,
        event_type=attrs.get("event_type", envelope.get("type", "")),
        run_id=attrs.get("run_id", data.get("run_id", "")),
        service_type=attrs.get("service_type", ""),
        seq=data.get("seq", 0),
    )


def pull_events(
    resources: PubSubTestResources,
    *,
    expected_count: int,
    timeout: float = 15.0,
) -> list[CollectedEvent]:
    """Pull exactly expected_count events from the subscription.

    Returns events sorted by seq.
    """
    collected: list[CollectedEvent] = []
    deadline = time.monotonic() + timeout

    while len(collected) < expected_count and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        response = resources.subscriber_client.pull(
            subscription=resources.subscription_path,
            max_messages=expected_count - len(collected),
            timeout=min(remaining, 5.0),
        )
        ack_ids = []
        for msg in response.received_messages:
            collected.append(_decode_message(msg.message))
            ack_ids.append(msg.ack_id)
        if ack_ids:
            resources.subscriber_client.acknowledge(
                subscription=resources.subscription_path,
                ack_ids=ack_ids,
            )
        if len(collected) < expected_count:
            time.sleep(0.1)

    if len(collected) < expected_count:
        event_types = [e.event_type for e in collected]
        raise AssertionError(f"Expected {expected_count} events, got {len(collected)} within {timeout}s. Event types received: {event_types}")

    return sorted(collected, key=lambda e: e.seq)


def assert_valid_cloudevent(event: CollectedEvent) -> None:
    """Validate a CollectedEvent has required CloudEvents 1.0 fields."""
    env = event.envelope
    assert env["specversion"] == "1.0", f"specversion={env.get('specversion')}"
    for field_name in ("id", "type", "source", "time", "subject", "datacontenttype"):
        assert field_name in env, f"Missing CloudEvents field: {field_name}"
    assert env["datacontenttype"] == "application/json"
    assert isinstance(env["data"], dict)
    assert env["data"]["run_id"] == env["subject"]


def assert_seq_monotonic(events: list[CollectedEvent]) -> None:
    """Assert that seq values are strictly increasing."""
    for i in range(1, len(events)):
        assert events[i].seq > events[i - 1].seq, f"seq not monotonic: events[{i - 1}].seq={events[i - 1].seq} >= events[{i}].seq={events[i].seq}"


# Emulator fixtures — only registered when testcontainers is available

EMULATOR_PORT = 8085
EMULATOR_PROJECT = "test-project"

if _testcontainers_available:
    assert DockerContainer is not None
    assert wait_for_logs is not None

    class PubSubEmulatorContainer(DockerContainer):  # pyright: ignore[reportUntypedBaseClass]
        """GCP Pub/Sub emulator via google/cloud-sdk."""

        def __init__(self) -> None:
            super().__init__("google/cloud-sdk:emulators")
            self.with_exposed_ports(EMULATOR_PORT)
            self.with_command(f"gcloud beta emulators pubsub start --host-port=0.0.0.0:{EMULATOR_PORT}")

    @pytest.fixture(scope="session")
    def pubsub_emulator() -> Generator[str, None, None]:
        """Start Pub/Sub emulator container for the test session."""
        container = PubSubEmulatorContainer()
        container.start()
        wait_for_logs(container, "Server started", timeout=30)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(EMULATOR_PORT)
        emulator_host = f"{host}:{port}"
        os.environ["PUBSUB_EMULATOR_HOST"] = emulator_host

        # Warm up: ensure emulator is responsive
        client = PublisherClient()
        warmup_topic = client.create_topic(name=client.topic_path(EMULATOR_PROJECT, "warmup"))
        client.delete_topic(topic=warmup_topic.name)

        yield emulator_host

        del os.environ["PUBSUB_EMULATOR_HOST"]
        container.stop()

    @pytest.fixture
    def pubsub_topic(pubsub_emulator: str) -> Generator[PubSubTestResources, None, None]:
        """Create a unique topic + subscription per test, clean up after."""
        pub_client = PublisherClient()
        sub_client = SubscriberClient()

        topic_id = f"test-events-{uuid4().hex[:8]}"
        topic_path = pub_client.topic_path(EMULATOR_PROJECT, topic_id)
        sub_id = f"test-sub-{uuid4().hex[:8]}"
        sub_path = sub_client.subscription_path(EMULATOR_PROJECT, sub_id)

        pub_client.create_topic(name=topic_path)
        sub_client.create_subscription(name=sub_path, topic=topic_path)

        yield PubSubTestResources(
            project_id=EMULATOR_PROJECT,
            topic_path=topic_path,
            subscription_path=sub_path,
            publisher_client=pub_client,
            subscriber_client=sub_client,
        )

        try:
            sub_client.delete_subscription(subscription=sub_path)
        except (OSError, GoogleAPICallError):
            pass
        try:
            pub_client.delete_topic(topic=topic_path)
        except (OSError, GoogleAPICallError):
            pass

    @pytest.fixture
    def real_publisher(pubsub_topic: PubSubTestResources) -> PublisherWithStore:
        """Create a PubSubPublisher pointed at the emulator topic."""
        topic_id = pubsub_topic.topic_path.split("/")[-1]
        publisher = PubSubPublisher(
            project_id=pubsub_topic.project_id,
            topic_id=topic_id,
            service_type="test-service",
        )
        return PublisherWithStore(publisher=publisher)
