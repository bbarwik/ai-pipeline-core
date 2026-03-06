"""Integration tests for the replay system with real LLM calls."""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import Conversation, ModelOptions
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context
from ai_pipeline_core.replay.types import TaskReplay
from ai_pipeline_core.settings import settings

pytestmark = pytest.mark.integration
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)
DEFAULT_MODEL = "gemini-3-flash"
MAX_COMPLETION_TOKENS = 1000


# -- Structured output models ------------------------------------------------


class ReplayStructuredResponse(BaseModel):
    prompt_summary: str
    requested_token: str
    validation_passed: bool


class ReplayMultiTurnResponse(BaseModel):
    prior_context_summary: str
    codename: str
    is_context_consistent: bool


class ReplayTaskExtraction(BaseModel):
    source_summary: str
    extracted_code: str


# -- Document types -----------------------------------------------------------


class ReplayContextDocument(Document):
    """Context document for replay integration tests."""


class ReplayTaskInputDocument(Document):
    """Input document for task replay integration tests."""


class ReplayTaskOutputDocument(Document):
    """Output document for task replay integration tests."""


# -- Laminar capture helpers --------------------------------------------------


class _CaptureSpan:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = payloads

    def __enter__(self) -> "_CaptureSpan":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def set_attribute(self, key: str, value: object) -> None:
        if key == "replay.payload" and isinstance(value, str):
            self._payloads.append(value)

    def set_attributes(self, attrs: dict[str, object]) -> None:
        payload = attrs.get("replay.payload")
        if isinstance(payload, str):
            self._payloads.append(payload)


class _CaptureLaminar:
    def __init__(self) -> None:
        self.payloads: list[str] = []

    def start_as_current_span(self, *args: object, **kwargs: object) -> _CaptureSpan:
        return _CaptureSpan(self.payloads)

    def set_span_output(self, output: object) -> None:
        pass


class _TaskCaptureLaminar:
    """Capture for PipelineTask spans."""

    def __init__(self) -> None:
        self.payloads: list[str] = []

    def set_span_attributes(self, attrs: dict[str, Any]) -> None:
        payload = attrs.get("replay.payload")
        if isinstance(payload, str):
            self.payloads.append(payload)


# -- Pipeline task for replay test -------------------------------------------


class ReplayIntegrationTask(PipelineTask):
    @classmethod
    async def run(cls, source: ReplayTaskInputDocument, model_name: str) -> ReplayTaskOutputDocument:
        conv = Conversation(model=model_name, model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"))
        conv = conv.with_context(source)
        conv = await conv.send_structured(
            "Extract source_summary and extracted_code from context.",
            response_format=ReplayTaskExtraction,
            purpose="replay_task",
        )
        parsed = conv.parsed
        assert parsed is not None, "Structured output returned None"
        return ReplayTaskOutputDocument.derive(
            from_documents=(source,),
            name=f"result_{source.id}.json",
            content=parsed.model_dump_json(),
            description="Task result",
        )


# -- Helpers ------------------------------------------------------------------


def _write_document_to_local_store(doc: Document, store_base: Path) -> None:
    """Write a document to disk in LocalDocumentStore layout."""
    class_name = doc.__class__.__name__
    type_dir = store_base / class_name
    type_dir.mkdir(parents=True, exist_ok=True)
    sha_prefix = doc.sha256[:6]
    safe_name = f"{Path(doc.name).stem}_{sha_prefix}{Path(doc.name).suffix}"
    (type_dir / safe_name).write_bytes(doc.content)
    (type_dir / f"{safe_name}.meta.json").write_text(
        json.dumps({
            "name": doc.name,
            "document_sha256": doc.sha256,
            "content_sha256": doc.sha256,
            "class_name": class_name,
            "description": doc.description,
            "derived_from": list(doc.derived_from),
            "triggered_by": list(doc.triggered_by),
            "mime_type": "text/plain",
            "attachments": [],
        }),
        encoding="utf-8",
    )


# -- Unit test (always runs) -------------------------------------------------


def test_task_replay_yaml_roundtrip() -> None:
    payload = TaskReplay(function_path="pkg.tasks:NormalizeTask", arguments={"documents": []})
    restored = TaskReplay.from_yaml(payload.to_yaml())
    assert restored.function_path == payload.function_path
    assert restored.payload_type == "pipeline_task"


# -- Integration tests (require API keys) ------------------------------------


@pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured in settings or .env file")
class TestReplayIntegration:
    """Integration tests exercising the replay lifecycle with real LLM calls."""

    @pytest.mark.asyncio
    async def test_end_to_end_conversation_replay(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture ConversationReplay, YAML round-trip, re-execute, verify token in response."""
        from ai_pipeline_core.replay import ConversationReplay

        token = uuid.uuid4().hex[:10].upper()
        capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", capture)

        doc = ReplayContextDocument.create_root(
            name="context.txt",
            content=f"The replay code is {token}.",
            reason="test input",
            description="Replay context",
        )
        _write_document_to_local_store(doc, tmp_path)

        conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            context=(doc,),
            enable_substitutor=False,
        )
        conv = await conv.send(f"Reply with only the replay code from the document. The code is {token}.")
        assert conv.content
        assert len(capture.payloads) > 0, "No replay payload captured"

        replay = ConversationReplay.from_yaml(capture.payloads[-1])
        yaml_text = replay.to_yaml()
        assert DEFAULT_MODEL in yaml_text

        restored = ConversationReplay.from_yaml(yaml_text)
        result = await restored.execute(store_base=tmp_path)
        assert token in result.content
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_replay_with_changed_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture with gemini-3-flash, replay with grok-4.1-fast, verify structured output."""
        from ai_pipeline_core.replay import ConversationReplay

        token = uuid.uuid4().hex[:10].upper()
        capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", capture)

        conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await conv.send_structured(
            f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
            response_format=ReplayStructuredResponse,
        )
        assert conv.parsed is not None
        assert conv.parsed.requested_token == token

        assert len(capture.payloads) > 0
        replay = ConversationReplay.from_yaml(capture.payloads[-1])
        modified = replay.model_copy(update={"model": "grok-4.1-fast"})
        assert modified.model == "grok-4.1-fast"

        result = await modified.execute(store_base=tmp_path)
        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayStructuredResponse)
        assert result.parsed.requested_token == token

    @pytest.mark.asyncio
    async def test_replay_with_changed_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture with token_a, replay with token_b, verify only token_b appears."""
        from ai_pipeline_core.replay import ConversationReplay

        token_a = uuid.uuid4().hex[:10].upper()
        token_b = uuid.uuid4().hex[:10].upper()
        capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", capture)

        conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await conv.send(f"Reply with only this token: {token_a}")

        assert len(capture.payloads) > 0
        replay = ConversationReplay.from_yaml(capture.payloads[-1])
        modified = replay.model_copy(update={"prompt": f"Reply with only this token: {token_b}"})

        result = await modified.execute(store_base=tmp_path)
        assert token_b in result.content
        assert token_a not in result.content

    @pytest.mark.asyncio
    async def test_replay_with_changed_model_options(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture with reasoning_effort=low, replay with reasoning_effort=high."""
        from ai_pipeline_core.replay import ConversationReplay

        token = uuid.uuid4().hex[:10].upper()
        capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", capture)

        conv = Conversation(
            model=DEFAULT_MODEL,
            enable_substitutor=False,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
        )
        conv = await conv.send_structured(
            f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
            response_format=ReplayStructuredResponse,
        )
        assert conv.parsed is not None

        assert len(capture.payloads) > 0
        replay = ConversationReplay.from_yaml(capture.payloads[-1])
        modified = replay.model_copy(update={"model_options": {**replay.model_options, "reasoning_effort": "high", "max_completion_tokens": 2000}})

        result = await modified.execute(store_base=tmp_path)
        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayStructuredResponse)
        assert result.parsed.requested_token == token

    @pytest.mark.asyncio
    async def test_structured_output_replay(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify response_format round-trips through YAML and produces typed .parsed."""
        from ai_pipeline_core.replay import ConversationReplay

        token = uuid.uuid4().hex[:10].upper()
        capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", capture)

        conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await conv.send_structured(
            f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
            response_format=ReplayStructuredResponse,
        )
        assert conv.parsed is not None

        assert len(capture.payloads) > 0
        yaml_text = capture.payloads[-1]
        response_format_path = f"{ReplayStructuredResponse.__module__}:{ReplayStructuredResponse.__qualname__}"
        assert response_format_path in yaml_text or "ReplayStructuredResponse" in yaml_text

        replay = ConversationReplay.from_yaml(yaml_text)
        assert replay.response_format is not None

        result = await replay.execute(store_base=tmp_path)
        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayStructuredResponse)
        assert result.parsed.requested_token == token

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_replay(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replay turn 2 of a multi-turn conversation with history from turn 1."""
        from ai_pipeline_core.replay import ConversationReplay

        token = uuid.uuid4().hex[:10].upper()
        capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", capture)

        conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await conv.send(f"Remember codename: {token}")
        assert conv.content

        capture.payloads.clear()  # capture only turn 2
        conv = await conv.send_structured("Return the codename I told you.", response_format=ReplayMultiTurnResponse)
        assert conv.parsed is not None
        assert token in conv.parsed.codename.upper()

        assert len(capture.payloads) > 0
        replay = ConversationReplay.from_yaml(capture.payloads[-1])

        has_user = any(e.type == "user_text" for e in replay.history)
        has_response = any(e.type == "response" for e in replay.history)
        assert has_user, "History must contain user_text entry from turn 1"
        assert has_response, "History must contain response entry from turn 1"

        yaml_text = replay.to_yaml()
        restored = ConversationReplay.from_yaml(yaml_text)
        result = await restored.execute(store_base=tmp_path)

        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayMultiTurnResponse)
        assert token in result.parsed.codename.upper()

    @pytest.mark.asyncio
    async def test_replay_with_document_context(self, tmp_path: Path) -> None:
        """Manually build ConversationReplay with $doc_ref, execute, verify LLM uses document."""
        from ai_pipeline_core.replay import ConversationReplay, DocumentRef

        doc = ReplayContextDocument.create_root(
            name="facts.txt",
            content="The Eiffel Tower is 330 meters tall and was built in 1889.",
            reason="test input",
            description="Facts about the Eiffel Tower",
        )
        _write_document_to_local_store(doc, tmp_path)

        replay = ConversationReplay(
            model=DEFAULT_MODEL,
            model_options={"max_completion_tokens": MAX_COMPLETION_TOKENS},
            prompt="How tall is the Eiffel Tower according to the document? Answer in one sentence.",
            context=(
                DocumentRef.model_validate({
                    "$doc_ref": doc.sha256,
                    "class_name": "ReplayContextDocument",
                    "name": "facts.txt",
                }),
            ),
            enable_substitutor=False,
        )

        result = await replay.execute(store_base=tmp_path)
        assert result.content
        assert "330" in result.content
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_pipeline_task_replay(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Capture TaskReplay from a PipelineTask run, YAML round-trip, re-execute."""
        from ai_pipeline_core.document_store._memory import MemoryDocumentStore
        from ai_pipeline_core.replay import TaskReplay as TaskReplayType

        token = uuid.uuid4().hex[:10].upper()
        task_capture = _TaskCaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.pipeline._task.Laminar", task_capture)
        conv_capture = _CaptureLaminar()
        monkeypatch.setattr("ai_pipeline_core.llm.conversation.Laminar", conv_capture)

        source_doc = ReplayTaskInputDocument.create_root(
            name="source.txt",
            content=f"Project code: {token}. Extract the code from this document.",
            reason="test input",
            description="Source document for task replay",
        )
        _write_document_to_local_store(source_doc, tmp_path)

        store = MemoryDocumentStore()
        with pipeline_test_context(store=store):
            result_docs: list[Any] = await ReplayIntegrationTask.run(source_doc, model_name=DEFAULT_MODEL)

        assert len(result_docs) == 1
        result_doc = result_docs[0]
        assert isinstance(result_doc, ReplayTaskOutputDocument)
        parsed = ReplayTaskExtraction.model_validate_json(result_doc.content)
        assert token in parsed.extracted_code.upper() or token in parsed.source_summary.upper()

        assert len(task_capture.payloads) > 0, "No task replay payload captured"
        task_replay = TaskReplayType.from_yaml(task_capture.payloads[-1])

        yaml_text = task_replay.to_yaml()
        restored = TaskReplayType.from_yaml(yaml_text)
        replay_result = await restored.execute(store_base=tmp_path)

        assert isinstance(replay_result, list)
        assert len(replay_result) == 1
        assert isinstance(replay_result[0], ReplayTaskOutputDocument)
        replay_parsed = ReplayTaskExtraction.model_validate_json(replay_result[0].content)
        assert token in replay_parsed.extracted_code.upper() or token in replay_parsed.source_summary.upper()
