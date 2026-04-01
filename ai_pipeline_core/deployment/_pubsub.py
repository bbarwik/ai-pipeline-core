"""Google Cloud Pub/Sub ResultPublisher with CloudEvents 1.0 envelope.

Publishes pipeline lifecycle events directly from workers to a Pub/Sub topic.
All events use simple retry: 3 attempts with backoff (1s, 2s, 4s),
fire-and-forget on final failure.
"""

import asyncio
import json
import logging
from concurrent.futures import Future
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid7

from google.cloud.pubsub_v1 import PublisherClient  # pyright: ignore[reportMissingTypeStubs]

from ai_pipeline_core.exceptions import PipelineCoreError

from ._types import (
    DocumentRef,
    EventType,
    FlowCompletedEvent,
    FlowFailedEvent,
    FlowSkippedEvent,
    FlowStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)

logger = logging.getLogger(__name__)

MAX_PUBSUB_MESSAGE_BYTES = 8_388_608
PUBLISH_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1
CLOUDEVENTS_SPEC_VERSION = "1.0"


class ResultTooLargeError(PipelineCoreError):
    """Raised when a Pub/Sub message exceeds the size limit."""


class PubSubPublisher:
    """Publishes pipeline lifecycle events to Google Cloud Pub/Sub."""

    def __init__(
        self,
        project_id: str,
        topic_id: str,
        service_type: str,
    ) -> None:
        self._client = PublisherClient()
        self._topic_path: str = self._client.topic_path(project_id, topic_id)
        self._service_type = service_type
        self._seq = 0

    def _build_envelope(self, event_type: EventType, run_id: str, data: dict[str, Any]) -> bytes:
        """Build a CloudEvents 1.0 JSON envelope."""
        self._seq += 1
        envelope = {
            "id": str(uuid7()),
            "source": f"ai-{self._service_type}-worker",
            "type": event_type,
            "specversion": CLOUDEVENTS_SPEC_VERSION,
            "time": datetime.now(UTC).isoformat(),
            "subject": run_id,
            "datacontenttype": "application/json",
            "data": {"run_id": run_id, "seq": self._seq, **data},
        }
        return json.dumps(envelope, default=str).encode()

    async def _publish(self, data: bytes, attributes: dict[str, str]) -> None:
        """Publish to Pub/Sub with 3 retries and exponential backoff. Fire-and-forget on final failure."""
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                future = cast(Future[str], self._client.publish(self._topic_path, data, **attributes))  # pyright: ignore[reportUnknownMemberType] — google-cloud-pubsub stubs incomplete
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=PUBLISH_TIMEOUT_SECONDS,
                )
                return
            except Exception as e:
                last_error = e
                delay = BACKOFF_BASE_SECONDS * (2**attempt)
                logger.warning("Pub/Sub publish attempt %d failed: %s (retry in %ds)", attempt + 1, e, delay, exc_info=e)
                await asyncio.sleep(delay)
        logger.warning("Pub/Sub publish failed after %d attempts: %s", MAX_RETRIES, last_error, exc_info=last_error)

    def _make_attributes(self, event_type: EventType, run_id: str) -> dict[str, str]:
        """Build Pub/Sub message attributes."""
        return {
            "service_type": self._service_type,
            "event_type": str(event_type),
            "run_id": run_id,
        }

    async def _publish_event(self, event_type: EventType, run_id: str, payload: dict[str, Any]) -> None:
        """Build and publish one CloudEvents payload for a lifecycle event."""
        data = self._build_envelope(event_type, run_id, payload)
        if event_type == EventType.RUN_COMPLETED and len(data) > MAX_PUBSUB_MESSAGE_BYTES:
            raise ResultTooLargeError(f"Completed event ({len(data)} bytes) exceeds {MAX_PUBSUB_MESSAGE_BYTES} byte Pub/Sub limit")
        attrs = self._make_attributes(event_type, run_id)
        root_id = payload.get("root_deployment_id", "")
        if root_id:
            attrs["root_deployment_id"] = str(root_id)
        await self._publish(data, attrs)

    @staticmethod
    def _doc_payloads(document_refs: tuple[DocumentRef, ...] | list[DocumentRef]) -> list[dict[str, Any]]:
        """Serialize document references into JSON-safe dicts."""
        return [asdict(doc) for doc in document_refs]

    async def publish_run_started(self, event: RunStartedEvent) -> None:
        """Publish run.started event."""
        await self._publish_event(
            EventType.RUN_STARTED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "deployment_name": event.deployment_name,
                "deployment_class": event.deployment_class,
                "input_fingerprint": event.input_fingerprint,
                "status": event.status,
                "flow_plan": event.flow_plan,
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_heartbeat(self, run_id: str, *, root_deployment_id: str = "", span_id: str = "") -> None:
        """Publish run.heartbeat event."""
        await self._publish_event(
            EventType.RUN_HEARTBEAT,
            run_id,
            {
                "root_deployment_id": root_deployment_id,
                "span_id": span_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    async def publish_run_completed(self, event: RunCompletedEvent) -> None:
        """Publish run.completed event."""
        await self._publish_event(
            EventType.RUN_COMPLETED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "deployment_name": event.deployment_name,
                "deployment_class": event.deployment_class,
                "status": event.status,
                "result": event.result,
                "duration_ms": event.duration_ms,
                "output_document_sha256s": list(event.output_document_sha256s),
            },
        )

    async def publish_run_failed(self, event: RunFailedEvent) -> None:
        """Publish run.failed event."""
        await self._publish_event(
            EventType.RUN_FAILED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "deployment_name": event.deployment_name,
                "deployment_class": event.deployment_class,
                "status": event.status,
                "error_code": str(event.error_code),
                "error_message": event.error_message,
                "duration_ms": event.duration_ms,
            },
        )

    async def publish_flow_started(self, event: FlowStartedEvent) -> None:
        """Publish flow.started event."""
        await self._publish_event(
            EventType.FLOW_STARTED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "flow_class": event.flow_class,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "expected_tasks": event.expected_tasks,
                "flow_params": event.flow_params,
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_flow_completed(self, event: FlowCompletedEvent) -> None:
        """Publish flow.completed event."""
        await self._publish_event(
            EventType.FLOW_COMPLETED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "flow_class": event.flow_class,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "duration_ms": event.duration_ms,
                "output_documents": self._doc_payloads(event.output_documents),
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_flow_failed(self, event: FlowFailedEvent) -> None:
        """Publish flow.failed event."""
        await self._publish_event(
            EventType.FLOW_FAILED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "flow_class": event.flow_class,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "error_message": event.error_message,
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_flow_skipped(self, event: FlowSkippedEvent) -> None:
        """Publish flow.skipped event."""
        await self._publish_event(
            EventType.FLOW_SKIPPED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "flow_class": event.flow_class,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "reason": event.reason,
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_task_started(self, event: TaskStartedEvent) -> None:
        """Publish task.started event."""
        await self._publish_event(
            EventType.TASK_STARTED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "task_name": event.task_name,
                "task_class": event.task_class,
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_task_completed(self, event: TaskCompletedEvent) -> None:
        """Publish task.completed event."""
        await self._publish_event(
            EventType.TASK_COMPLETED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "task_name": event.task_name,
                "task_class": event.task_class,
                "duration_ms": event.duration_ms,
                "output_documents": self._doc_payloads(event.output_documents),
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def publish_task_failed(self, event: TaskFailedEvent) -> None:
        """Publish task.failed event."""
        await self._publish_event(
            EventType.TASK_FAILED,
            event.run_id,
            {
                "span_id": event.span_id,
                "root_deployment_id": event.root_deployment_id,
                "parent_deployment_task_id": event.parent_deployment_task_id,
                "parent_span_id": event.parent_span_id,
                "flow_name": event.flow_name,
                "step": event.step,
                "total_steps": event.total_steps,
                "status": event.status,
                "task_name": event.task_name,
                "task_class": event.task_class,
                "error_message": event.error_message,
                "input_document_sha256s": list(event.input_document_sha256s),
            },
        )

    async def close(self) -> None:
        """Close the Pub/Sub client."""
        try:
            self._client.stop()
        except Exception as e:
            logger.warning("Pub/Sub client stop failed: %s", e)


__all__ = [
    "PubSubPublisher",
    "ResultTooLargeError",
]
