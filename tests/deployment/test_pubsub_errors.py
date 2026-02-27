"""Tests for error classification in published Pub/Sub events via real emulator."""

# pyright: reportPrivateUsage=false, reportArgumentType=false, reportUnusedClass=false

from typing import ClassVar

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
from ai_pipeline_core.exceptions import LLMError, PipelineCoreError

from .conftest import (
    PubSubTestResources,
    PublisherWithStore,
    pull_events,
)

pytestmark = pytest.mark.pubsub


# ---------------------------------------------------------------------------
# Factory for dynamically creating failing deployments per exception type
# ---------------------------------------------------------------------------


class _ErrorInputDoc(Document):
    """Input doc for error classification tests."""


class _ErrorOutputDoc(Document):
    """Output doc for error classification tests."""


class _ErrorResult(DeploymentResult):
    """Result for error classification tests."""


# We need a unique flow per parametrize invocation to avoid Prefect name collisions.
# Use a factory that creates a fresh flow + deployment class for each exception.

_flow_counter = 0


def _make_failing_deployment(exc: Exception) -> PipelineDeployment[FlowOptions, _ErrorResult]:
    """Create a deployment whose single flow raises the given exception."""
    global _flow_counter
    _flow_counter += 1
    suffix = _flow_counter

    # The exception must be captured in a closure
    captured_exc = exc

    @pipeline_flow(estimated_minutes=1)
    async def error_flow(
        run_id: str,
        documents: list[_ErrorInputDoc],
        flow_options: FlowOptions,
    ) -> list[_ErrorOutputDoc]:
        raise captured_exc

    # Give the flow a unique name to avoid Prefect registration collisions
    error_flow.__name__ = f"_error_flow_{suffix}"
    error_flow.__qualname__ = f"_error_flow_{suffix}"
    if hasattr(error_flow, "name"):
        error_flow.name = f"_error_flow_{suffix}"

    class FailDeploy(PipelineDeployment[FlowOptions, _ErrorResult]):
        flows: ClassVar = [error_flow]

        @staticmethod
        def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> _ErrorResult:
            return _ErrorResult(success=False, error="failed")

    return FailDeploy()


# ---------------------------------------------------------------------------
# Parametrized error classification test
# ---------------------------------------------------------------------------


class TestErrorClassification:
    """Verify _classify_error maps exceptions to correct ErrorCode in real Pub/Sub events."""

    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (LLMError("fail"), "provider_error"),
            (TimeoutError("timed out"), "duration_exceeded"),
            (ValueError("bad"), "invalid_input"),
            (TypeError("wrong"), "invalid_input"),
            (PipelineCoreError("pipe"), "pipeline_error"),
            (OSError("os"), "unknown"),
            (RuntimeError("rt"), "unknown"),
        ],
        ids=["LLMError", "TimeoutError", "ValueError", "TypeError", "PipelineCoreError", "OSError", "RuntimeError"],
    )
    async def test_error_classification_in_published_events(
        self,
        exception: Exception,
        expected_code: str,
        real_publisher: PublisherWithStore,
        pubsub_topic: PubSubTestResources,
    ):
        """Each exception type maps to the correct error_code in the task.failed event."""
        store = MemoryDocumentStore()
        set_document_store(store)
        try:
            deployment = _make_failing_deployment(exception)
            doc = _ErrorInputDoc.create_root(name="err_input.txt", content="error test", reason="test")

            with pytest.raises(type(exception)):
                await deployment.run(
                    "error-run",
                    [doc],
                    FlowOptions(),
                    publisher=real_publisher.publisher,
                )
        finally:
            store.shutdown()
            set_document_store(None)

        # A single-flow deployment that immediately raises produces:
        # 1 started + 1 progress(STARTED for the flow) + 1 failed = 3 events
        events = pull_events(resources=pubsub_topic, expected_count=3, timeout=10.0)

        failed_events = [e for e in events if e.event_type == "task.failed"]
        assert len(failed_events) == 1, f"Expected exactly 1 task.failed event, got {len(failed_events)}. All event types: {[e.event_type for e in events]}"

        failed = failed_events[0]
        assert failed.data["error_code"] == expected_code, (
            f"Expected error_code='{expected_code}', got '{failed.data['error_code']}' for {type(exception).__name__}"
        )
        assert str(exception) in failed.data["error_message"], f"Expected error_message to contain '{exception}', got '{failed.data['error_message']}'"
