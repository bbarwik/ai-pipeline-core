"""Tests for NoopPublisher and MemoryPublisher."""

# pyright: reportPrivateUsage=false

from ai_pipeline_core.deployment._publishers import MemoryPublisher, NoopPublisher
from ai_pipeline_core.deployment._types import (
    CompletedEvent,
    ErrorCode,
    FailedEvent,
    ProgressEvent,
    ResultPublisher,
    StartedEvent,
)
from ai_pipeline_core.deployment.contract import FlowStatus


def _started_event() -> StartedEvent:
    return StartedEvent(run_id="run-1", flow_run_id="fr-1", run_scope="run-1:abc")


def _progress_event() -> ProgressEvent:
    return ProgressEvent(
        run_id="run-1",
        flow_run_id="fr-1",
        flow_name="extract",
        step=1,
        total_steps=3,
        progress=0.33,
        step_progress=0.5,
        status=FlowStatus.STARTED,
        message="halfway",
    )


def _completed_event() -> CompletedEvent:
    return CompletedEvent(
        run_id="run-1",
        flow_run_id="fr-1",
        result={"success": True},
        chain_context={"version": 1, "run_scope": "run-1:abc", "output_document_refs": []},
        actual_cost=0.0,
    )


def _failed_event() -> FailedEvent:
    return FailedEvent(
        run_id="run-1",
        flow_run_id="fr-1",
        error_code=ErrorCode.PIPELINE_ERROR,
        error_message="something failed",
    )


class TestNoopPublisher:
    """Test NoopPublisher silently discards all events."""

    def test_satisfies_protocol(self):
        """NoopPublisher must be a valid ResultPublisher."""
        assert isinstance(NoopPublisher(), ResultPublisher)

    async def test_publish_started(self):
        """publish_started does not raise."""
        await NoopPublisher().publish_started(_started_event())

    async def test_publish_progress(self):
        """publish_progress does not raise."""
        await NoopPublisher().publish_progress(_progress_event())

    async def test_publish_heartbeat(self):
        """publish_heartbeat does not raise."""
        await NoopPublisher().publish_heartbeat("run-1")

    async def test_publish_completed(self):
        """publish_completed does not raise."""
        await NoopPublisher().publish_completed(_completed_event())

    async def test_publish_failed(self):
        """publish_failed does not raise."""
        await NoopPublisher().publish_failed(_failed_event())


class TestMemoryPublisher:
    """Test MemoryPublisher records events for test assertions."""

    def test_satisfies_protocol(self):
        """MemoryPublisher must be a valid ResultPublisher."""
        assert isinstance(MemoryPublisher(), ResultPublisher)

    def test_starts_empty(self):
        """New MemoryPublisher has no events or heartbeats."""
        pub = MemoryPublisher()
        assert pub.events == []
        assert pub.heartbeats == []

    async def test_records_started(self):
        """publish_started appends to events list."""
        pub = MemoryPublisher()
        event = _started_event()
        await pub.publish_started(event)
        assert pub.events == [event]

    async def test_records_progress(self):
        """publish_progress appends to events list."""
        pub = MemoryPublisher()
        event = _progress_event()
        await pub.publish_progress(event)
        assert pub.events == [event]

    async def test_records_completed(self):
        """publish_completed appends to events list."""
        pub = MemoryPublisher()
        event = _completed_event()
        await pub.publish_completed(event)
        assert pub.events == [event]

    async def test_records_failed(self):
        """publish_failed appends to events list."""
        pub = MemoryPublisher()
        event = _failed_event()
        await pub.publish_failed(event)
        assert pub.events == [event]

    async def test_records_heartbeat_separately(self):
        """Heartbeats are stored in heartbeats list, not events."""
        pub = MemoryPublisher()
        await pub.publish_heartbeat("run-1")
        await pub.publish_heartbeat("run-2")
        assert pub.heartbeats == ["run-1", "run-2"]
        assert pub.events == []

    async def test_event_ordering(self):
        """Events are stored in publish order."""
        pub = MemoryPublisher()
        started = _started_event()
        progress = _progress_event()
        completed = _completed_event()
        await pub.publish_started(started)
        await pub.publish_progress(progress)
        await pub.publish_completed(completed)
        assert pub.events == [started, progress, completed]

    async def test_mixed_events_and_heartbeats(self):
        """Events and heartbeats are tracked independently."""
        pub = MemoryPublisher()
        await pub.publish_started(_started_event())
        await pub.publish_heartbeat("run-1")
        await pub.publish_progress(_progress_event())
        assert len(pub.events) == 2
        assert len(pub.heartbeats) == 1
