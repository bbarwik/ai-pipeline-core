"""In-process ResultPublisher implementations.

NoopPublisher silently discards all events (default for CLI and run_local).
MemoryPublisher records events in-memory for test assertions.
"""

from ._types import (
    CompletedEvent,
    FailedEvent,
    ProgressEvent,
    StartedEvent,
)


class NoopPublisher:
    """Discards all lifecycle events. Default publisher for CLI and run_local."""

    async def publish_started(self, event: StartedEvent) -> None:
        """Accept and discard a started event."""

    async def publish_progress(self, event: ProgressEvent) -> None:
        """Accept and discard a progress event."""

    async def publish_heartbeat(self, run_id: str) -> None:
        """Accept and discard a heartbeat."""

    async def publish_completed(self, event: CompletedEvent) -> None:
        """Accept and discard a completed event."""

    async def publish_failed(self, event: FailedEvent) -> None:
        """Accept and discard a failed event."""

    async def close(self) -> None:
        """No resources to release."""


class MemoryPublisher:
    """Records all lifecycle events in-memory for test assertions."""

    def __init__(self) -> None:
        self.events: list[StartedEvent | ProgressEvent | CompletedEvent | FailedEvent] = []
        self.heartbeats: list[str] = []

    async def publish_started(self, event: StartedEvent) -> None:
        """Record a started event."""
        self.events.append(event)

    async def publish_progress(self, event: ProgressEvent) -> None:
        """Record a progress event."""
        self.events.append(event)

    async def publish_heartbeat(self, run_id: str) -> None:
        """Record a heartbeat."""
        self.heartbeats.append(run_id)

    async def publish_completed(self, event: CompletedEvent) -> None:
        """Record a completed event."""
        self.events.append(event)

    async def publish_failed(self, event: FailedEvent) -> None:
        """Record a failed event."""
        self.events.append(event)

    async def close(self) -> None:
        """No resources to release."""


__all__ = [
    "MemoryPublisher",
    "NoopPublisher",
]
