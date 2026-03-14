"""Shared serialization for lifecycle event dataclasses."""

from dataclasses import asdict
from enum import StrEnum
from typing import Any

from ai_pipeline_core._lifecycle_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent

from ._types import (
    FlowCompletedEvent,
    FlowFailedEvent,
    FlowSkippedEvent,
    FlowStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
)

LifecycleEvent = (
    RunStartedEvent
    | RunCompletedEvent
    | RunFailedEvent
    | FlowStartedEvent
    | FlowCompletedEvent
    | FlowFailedEvent
    | FlowSkippedEvent
    | TaskStartedEvent
    | TaskCompletedEvent
    | TaskFailedEvent
)

__all__ = [
    "LifecycleEvent",
    "event_to_payload",
]


def event_to_payload(event: LifecycleEvent) -> dict[str, Any]:
    """Convert a lifecycle event dataclass to a JSON-safe payload dict.

    All values in the returned dict are JSON-primitive types (str, int, float,
    bool, None, list, dict). No enum instances or tuples remain.
    """
    payload = asdict(event)
    for key, value in payload.items():
        if isinstance(value, StrEnum):
            payload[key] = str(value)
    return payload
