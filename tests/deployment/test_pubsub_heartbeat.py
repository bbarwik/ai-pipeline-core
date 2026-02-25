"""Tests for heartbeat publishing during pipeline execution via real Pub/Sub emulator."""

# pyright: reportPrivateUsage=false, reportArgumentType=false

import asyncio
import json
from datetime import datetime
from unittest.mock import patch

import pytest
from google.api_core.exceptions import GoogleAPICallError

from ai_pipeline_core.document_store._memory import MemoryDocumentStore

from .conftest import (
    CollectedEvent,
    PubSubTestResources,
    PublisherWithStore,
    TwoFlowDeployment,
    run_pipeline,
)

pytestmark = pytest.mark.pubsub


def _pull_sync(pubsub_topic: PubSubTestResources, max_messages: int = 10, timeout: float = 1.0) -> list[CollectedEvent]:
    """Synchronous pull — meant to run in a thread via run_in_executor."""
    events: list[CollectedEvent] = []
    try:
        response = pubsub_topic.subscriber_client.pull(
            subscription=pubsub_topic.subscription_path,
            max_messages=max_messages,
            timeout=timeout,
        )
        ack_ids = []
        for msg in response.received_messages:
            envelope = json.loads(msg.message.data.decode())
            attrs = msg.message.attributes
            event_type = attrs.get("event_type", envelope.get("type", ""))
            events.append(
                CollectedEvent(
                    envelope=envelope,
                    data=envelope["data"],
                    event_type=event_type,
                    run_id=attrs.get("run_id", ""),
                    service_type=attrs.get("service_type", ""),
                    seq=envelope["data"].get("seq", 0),
                )
            )
            ack_ids.append(msg.ack_id)
        if ack_ids:
            pubsub_topic.subscriber_client.acknowledge(
                subscription=pubsub_topic.subscription_path,
                ack_ids=ack_ids,
            )
    except (OSError, GoogleAPICallError):
        pass
    return events


class TestHeartbeat:
    """Heartbeat publishing during long-running flows."""

    async def test_heartbeat_published_during_long_running_flow(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
        flow_gate: asyncio.Event,
    ):
        """Heartbeat events arrive on subscription while a flow is blocked."""
        flow_gate.clear()

        with patch("ai_pipeline_core.deployment.base._HEARTBEAT_INTERVAL_SECONDS", 0.3):
            run_task = asyncio.create_task(run_pipeline(TwoFlowDeployment(), real_publisher.publisher, pubsub_memory_store))

            # Let the pipeline task start and reach the gate
            await asyncio.sleep(0.1)

            # Poll for heartbeat events (up to 5 seconds)
            # Use run_in_executor so the blocking pull doesn't starve the event loop
            loop = asyncio.get_event_loop()
            heartbeats: list[CollectedEvent] = []
            deadline = loop.time() + 5.0
            while not heartbeats and loop.time() < deadline:
                batch = await loop.run_in_executor(None, _pull_sync, pubsub_topic)
                heartbeats.extend(evt for evt in batch if evt.event_type == "task.heartbeat")
                if not heartbeats:
                    await asyncio.sleep(0.2)

            # Release the gate so the pipeline can finish
            flow_gate.set()
            await run_task

        assert len(heartbeats) >= 1, f"Expected at least 1 heartbeat, got {len(heartbeats)}"

        # Validate heartbeat data has parseable ISO 8601 timestamp
        for hb in heartbeats:
            ts_str = hb.data["timestamp"]
            parsed = datetime.fromisoformat(ts_str)
            assert parsed.tzinfo is not None, f"Heartbeat timestamp should be timezone-aware: {ts_str}"

    async def test_heartbeat_stops_after_completion(
        self,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
        pubsub_memory_store: MemoryDocumentStore,
        flow_gate: asyncio.Event,
    ):
        """No additional heartbeats arrive after pipeline completes.

        Uses a short heartbeat interval and a gate to hold flow_1 long enough
        for heartbeats to accumulate. After releasing and awaiting completion,
        drains all events and then verifies no further heartbeats arrive.
        """
        flow_gate.clear()
        loop = asyncio.get_running_loop()

        with patch("ai_pipeline_core.deployment.base._HEARTBEAT_INTERVAL_SECONDS", 0.3):
            run_task = asyncio.create_task(run_pipeline(TwoFlowDeployment(), real_publisher.publisher, pubsub_memory_store))

            # Hold the gate for 2 seconds — enough for ~6 heartbeats at 0.3s interval
            await asyncio.sleep(2.0)

            # Release the gate and let the pipeline complete
            flow_gate.set()
            await run_task

        # Drain all events (lifecycle + heartbeats accumulated during execution)
        all_events: list[CollectedEvent] = []
        for _ in range(10):
            batch = await loop.run_in_executor(None, _pull_sync, pubsub_topic, 20, 1.0)
            all_events.extend(batch)
            if not batch:
                break

        pre_heartbeats = [e for e in all_events if e.event_type == "task.heartbeat"]
        assert len(pre_heartbeats) >= 1, (
            f"Expected at least 1 heartbeat during blocked execution, got {len(pre_heartbeats)}. All event types: {[e.event_type for e in all_events]}"
        )

        # Wait 2 seconds after completion — with 0.3s interval, ~6 heartbeats
        # would arrive if the loop were still running
        await asyncio.sleep(2.0)
        post_events = await loop.run_in_executor(None, _pull_sync, pubsub_topic, 10, 2.0)
        post_heartbeats = [e for e in post_events if e.event_type == "task.heartbeat"]
        assert len(post_heartbeats) == 0, f"Expected 0 heartbeats after completion, got {len(post_heartbeats)}"
