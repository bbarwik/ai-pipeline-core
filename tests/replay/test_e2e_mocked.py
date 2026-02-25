"""End-to-end replay round-trip with mocked LLM.

Exercises the full capture -> serialize -> deserialize -> execute cycle:

1. Define Document types, a @pipeline_task, and a @pipeline_flow.
2. Run the flow with mocked LLM and capture replay payloads via
   Laminar.set_span_attributes.
3. Verify all three payload types (conversation, pipeline_task, pipeline_flow)
   are captured.
4. Serialize each to YAML, deserialize back, and verify equality.
5. Execute each replayed payload (with mocked LLM) and verify results match.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import Conversation, ModelOptions
from ai_pipeline_core.pipeline import FlowOptions, pipeline_flow, pipeline_task
from ai_pipeline_core.replay import ConversationReplay, FlowReplay, TaskReplay
from tests.support.helpers import create_test_model_response


# ---------------------------------------------------------------------------
# Document and option types for the mocked pipeline
# ---------------------------------------------------------------------------


class MockInputDocument(Document):
    """Input document for end-to-end replay test."""


class MockOutputDocument(Document):
    """Output document for end-to-end replay test."""


class MockFlowOptions(FlowOptions):
    """Flow options for end-to-end replay test."""

    analysis_mode: str = "standard"


# ---------------------------------------------------------------------------
# Laminar capture helpers
# ---------------------------------------------------------------------------


class _CaptureSpan:
    """Minimal span that records replay.payload values."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads

    def __enter__(self) -> "_CaptureSpan":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def set_attribute(self, key: str, value: object) -> None:
        if key == "replay.payload" and isinstance(value, str):
            self._payloads.append(json.loads(value))

    def set_attributes(self, attrs: dict[str, object]) -> None:
        payload = attrs.get("replay.payload")
        if isinstance(payload, str):
            self._payloads.append(json.loads(payload))


class _CaptureLaminar:
    """Drop-in replacement for Laminar that collects replay payloads."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def start_as_current_span(self, *args: object, **kwargs: object) -> _CaptureSpan:
        return _CaptureSpan(self.payloads)

    def set_span_output(self, output: object) -> None:
        pass

    def set_span_attributes(self, attrs: dict[str, Any]) -> None:
        payload = attrs.get("replay.payload")
        if isinstance(payload, str):
            self.payloads.append(json.loads(payload))


# ---------------------------------------------------------------------------
# Pipeline definitions (task + flow)
# ---------------------------------------------------------------------------


@pipeline_task(estimated_minutes=1)
async def mock_analysis_task(source: MockInputDocument, *, label: str) -> MockOutputDocument:
    """Task that uses a mocked Conversation to produce output."""
    conv = Conversation(
        model="test-model",
        model_options=ModelOptions(max_completion_tokens=100),
    )
    conv = conv.with_context(source)
    conv = await conv.send(f"Analyze document with label={label}")
    return MockOutputDocument(
        name="analysis_result.txt",
        content=conv.content.encode(),
        description=f"Analysis: {label}",
        derived_from=(source.sha256,),
    )


@pipeline_flow(estimated_minutes=1)
async def mock_analysis_flow(
    run_id: str,
    documents: list[MockInputDocument],
    flow_options: MockFlowOptions,
) -> list[MockOutputDocument]:
    """Flow that calls the analysis task for each input document."""
    results: list[MockOutputDocument] = []
    for doc in documents:
        result = await mock_analysis_task(doc, label=flow_options.analysis_mode)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestEndToEndMockedReplay:
    """Full pipeline capture -> YAML round-trip -> replay execution."""

    @pytest.mark.asyncio
    async def test_full_replay_round_trip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Run a mocked pipeline, capture payloads, round-trip through YAML, re-execute."""

        # -- 1. Set up mocked LLM -----------------------------------------

        async def fake_generate(messages: Any, **kwargs: Any) -> Any:
            return create_test_model_response(content="Mocked LLM analysis output.")

        monkeypatch.setattr("ai_pipeline_core.llm.conversation.core_generate", fake_generate)

        # -- 2. Set up Laminar capture -------------------------------------

        conv_capture = _CaptureLaminar()
        task_capture = _CaptureLaminar()
        flow_capture = _CaptureLaminar()

        # Conversation-level Laminar (for send())
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", conv_capture)
        # Pipeline decorator-level Laminar (for set_span_attributes in task/flow wrappers)
        monkeypatch.setattr("ai_pipeline_core.pipeline.decorators.Laminar", task_capture)

        # -- 3. Prepare input data -----------------------------------------

        input_doc = MockInputDocument(
            name="input.txt",
            content=b"Sample content for end-to-end replay testing.",
            description="Test input document",
        )

        # Persist document for later replay resolution
        store_base = tmp_path / "output"
        from ai_pipeline_core.document_store._local import LocalDocumentStore
        from ai_pipeline_core.documents.types import RunScope

        store = LocalDocumentStore(base_path=store_base)
        await store.save(input_doc, RunScope("e2e/test"))

        flow_options = MockFlowOptions(analysis_mode="deep")

        # -- 4. Run the pipeline -------------------------------------------

        # Direct call (bypass Prefect orchestration in tests)
        result_docs = await mock_analysis_flow.fn(
            "run-e2e-001",
            [input_doc],
            flow_options,
        )

        assert len(result_docs) == 1
        assert isinstance(result_docs[0], MockOutputDocument)
        assert result_docs[0].content == b"Mocked LLM analysis output."

        # -- 5. Collect all captured payloads ------------------------------

        all_payloads = conv_capture.payloads + task_capture.payloads + flow_capture.payloads

        conversation_payloads = [p for p in all_payloads if p.get("payload_type") == "conversation"]
        task_payloads = [p for p in all_payloads if p.get("payload_type") == "pipeline_task"]
        flow_payloads = [p for p in all_payloads if p.get("payload_type") == "pipeline_flow"]

        assert len(conversation_payloads) >= 1, (
            f"Expected at least 1 conversation payload, got {len(conversation_payloads)}. All payloads: {[p.get('payload_type') for p in all_payloads]}"
        )
        assert len(task_payloads) >= 1, (
            f"Expected at least 1 task payload, got {len(task_payloads)}. All payloads: {[p.get('payload_type') for p in all_payloads]}"
        )
        assert len(flow_payloads) >= 1, (
            f"Expected at least 1 flow payload, got {len(flow_payloads)}. All payloads: {[p.get('payload_type') for p in all_payloads]}"
        )

        # -- 6. YAML round-trip for each payload type ---------------------

        conv_payload = conversation_payloads[0]
        conv_replay = ConversationReplay.model_validate(conv_payload)
        conv_yaml = conv_replay.to_yaml()
        conv_restored = ConversationReplay.from_yaml(conv_yaml)
        assert conv_restored.model == conv_replay.model
        assert conv_restored.prompt == conv_replay.prompt
        assert conv_restored.payload_type == "conversation"

        task_payload = task_payloads[0]
        task_replay = TaskReplay.model_validate(task_payload)
        task_yaml = task_replay.to_yaml()
        task_restored = TaskReplay.from_yaml(task_yaml)
        assert task_restored.function_path == task_replay.function_path
        assert task_restored.payload_type == "pipeline_task"

        flow_payload = flow_payloads[0]
        flow_replay = FlowReplay.model_validate(flow_payload)
        flow_yaml = flow_replay.to_yaml()
        flow_restored = FlowReplay.from_yaml(flow_yaml)
        assert flow_restored.function_path == flow_replay.function_path
        assert flow_restored.run_id == flow_replay.run_id
        assert flow_restored.payload_type == "pipeline_flow"

        # -- 7. Re-execute each replay with mocked LLM --------------------

        # Conversation replay
        with patch("ai_pipeline_core.llm.conversation.Laminar", MagicMock()):
            conv_result = await conv_restored.execute(store_base)

        assert isinstance(conv_result, Conversation)
        assert conv_result.content == "Mocked LLM analysis output."

        # Task replay
        task_result = await task_restored.execute(store_base)
        assert isinstance(task_result, MockOutputDocument)
        assert task_result.content == b"Mocked LLM analysis output."

        # Flow replay
        flow_result = await flow_restored.execute(store_base)
        assert isinstance(flow_result, list)
        assert len(flow_result) == 1
        assert isinstance(flow_result[0], MockOutputDocument)
