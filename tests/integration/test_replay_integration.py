"""Integration tests for the replay system with real LLM calls."""

import os
import asyncio
from datetime import UTC, datetime
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from ai_pipeline_core._llm_core.model_response import ModelResponse
from ai_pipeline_core.database import ExecutionNode, NodeKind, NodeStatus
from ai_pipeline_core.database._download import download_deployment
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import Conversation, ModelOptions
from ai_pipeline_core.pipeline import PipelineTask, pipeline_test_context
from ai_pipeline_core.replay.cli import main
from ai_pipeline_core.replay.types import TaskReplay
from ai_pipeline_core.settings import settings
from tests.replay.conftest import store_document_in_database

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


class ReplayDownloadOutputDocument(Document):
    """Output document returned by download/replay integration tests."""


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


class ReplayDownloadedBundleTask(PipelineTask):
    @classmethod
    async def run(cls, source: ReplayTaskInputDocument, label: str) -> tuple[ReplayDownloadOutputDocument, ...]:
        _ = cls
        return (
            ReplayDownloadOutputDocument(
                name="downloaded-result.txt",
                content=source.content,
                description=label,
            ),
        )


async def _create_replay_database(base_path: Path, *documents: Document) -> FilesystemDatabase:
    """Persist replay documents into a FilesystemDatabase snapshot."""
    database = FilesystemDatabase(base_path)
    for document in documents:
        await store_document_in_database(database, document)
    return database


# -- Unit test (always runs) -------------------------------------------------


def test_task_replay_yaml_roundtrip() -> None:
    payload = TaskReplay(function_path="pkg.tasks:NormalizeTask", arguments={"documents": []})
    restored = TaskReplay.from_yaml(payload.to_yaml())
    assert restored.function_path == payload.function_path
    assert restored.payload_type == "pipeline_task"


@pytest.mark.clickhouse
@pytest.mark.skipif(not os.environ.get("CLICKHOUSE_HOST"), reason="ClickHouse not available (CLICKHOUSE_HOST not set)")
def test_clickhouse_downloaded_bundle_can_replay_cross_deployment_document_refs(tmp_path: Path) -> None:
    from ai_pipeline_core.database._clickhouse import ClickHouseDatabase
    from ai_pipeline_core.settings import Settings

    async def seed_and_download() -> tuple[Path, uuid.UUID]:
        database = ClickHouseDatabase(Settings())
        deployment_id = uuid.uuid4()
        external_deployment_id = uuid.uuid4()
        task_id = uuid.uuid4()
        try:
            source_doc = ReplayTaskInputDocument.create_root(
                name="clickhouse-download.txt",
                content="downloaded bundle replay input",
                reason="clickhouse integration input",
            )
            await store_document_in_database(database, source_doc, deployment_id=external_deployment_id)

            deployment_node = ExecutionNode(
                node_id=deployment_id,
                node_kind=NodeKind.DEPLOYMENT,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                run_id="clickhouse-download-run",
                run_scope="clickhouse-download-run/scope",
                deployment_name="clickhouse-download",
                name="clickhouse-download",
                sequence_no=0,
                status=NodeStatus.COMPLETED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={"flow_plan": [], "options": {}, "parent_execution_id": ""},
            )
            flow_node = ExecutionNode(
                node_id=uuid.uuid4(),
                node_kind=NodeKind.FLOW,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                parent_node_id=deployment_id,
                run_id="clickhouse-download-run",
                run_scope="clickhouse-download-run/scope",
                deployment_name="clickhouse-download",
                name="ClickHouseReplayFlow",
                sequence_no=1,
                status=NodeStatus.COMPLETED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={},
            )
            task_node = ExecutionNode(
                node_id=task_id,
                node_kind=NodeKind.TASK,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                parent_node_id=flow_node.node_id,
                run_id="clickhouse-download-run",
                run_scope="clickhouse-download-run/scope",
                deployment_name="clickhouse-download",
                name="ClickHouseReplayTask",
                sequence_no=1,
                flow_id=flow_node.node_id,
                status=NodeStatus.COMPLETED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={
                    "replay_payload": {
                        "version": 1,
                        "payload_type": "pipeline_task",
                        "function_path": f"{ReplayDownloadedBundleTask.__module__}:{ReplayDownloadedBundleTask.__qualname__}",
                        "arguments": {
                            "source": {
                                "$doc_ref": source_doc.sha256,
                                "class_name": type(source_doc).__name__,
                                "name": source_doc.name,
                            },
                            "label": "clickhouse-replayed",
                        },
                        "original": {},
                    }
                },
            )

            await database.insert_node(deployment_node)
            await database.insert_node(flow_node)
            await database.insert_node(task_node)

            download_dir = tmp_path / "clickhouse-downloaded"
            await download_deployment(database, deployment_id, download_dir)
            return download_dir, task_id
        finally:
            await database.shutdown()

    download_dir, task_id = asyncio.run(seed_and_download())
    replay_output_dir = tmp_path / "clickhouse-replay-output"
    exit_code = main([
        "run",
        "--from-db",
        str(task_id),
        "--db-path",
        str(download_dir),
        "--output-dir",
        str(replay_output_dir),
    ])

    assert exit_code == 0


# -- Integration tests (require API keys) ------------------------------------


@pytest.mark.skipif(not HAS_API_KEYS, reason="OpenAI API keys not configured in settings or .env file")
class TestReplayIntegration:
    """Integration tests exercising the replay lifecycle with real LLM calls."""

    @pytest.mark.asyncio
    async def test_end_to_end_conversation_replay(self, tmp_path: Path) -> None:
        """Capture ConversationReplay, YAML round-trip, re-execute, verify token in response."""
        from ai_pipeline_core.replay import ConversationReplay

        token = uuid.uuid4().hex[:10].upper()

        doc = ReplayContextDocument.create_root(
            name="context.txt",
            content=f"The replay code is {token}.",
            reason="test input",
            description="Replay context",
        )
        replay_database = await _create_replay_database(tmp_path, doc)

        base_conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            context=(doc,),
            enable_substitutor=False,
        )
        conv = await base_conv.send(f"Reply with only the replay code from the document. The code is {token}.")
        assert conv.content
        response = cast(ModelResponse[Any], conv.messages[-1])
        replay = ConversationReplay.model_validate(
            base_conv._build_replay_payload(
                f"Reply with only the replay code from the document. The code is {token}.",
                None,
                None,
                response,
            )
        )
        yaml_text = replay.to_yaml()
        assert DEFAULT_MODEL in yaml_text

        restored = ConversationReplay.from_yaml(yaml_text)
        result = await restored.execute(replay_database)
        assert token in result.content
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_replay_with_changed_model(self, tmp_path: Path) -> None:
        """Capture with gemini-3-flash, replay with grok-4.1-fast, verify structured output."""
        from ai_pipeline_core.replay import ConversationReplay

        replay_database = FilesystemDatabase(tmp_path)
        token = uuid.uuid4().hex[:10].upper()
        base_conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await base_conv.send_structured(
            f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
            response_format=ReplayStructuredResponse,
        )
        assert conv.parsed is not None
        assert conv.parsed.requested_token == token

        response = cast(ModelResponse[Any], conv.messages[-1])
        replay = ConversationReplay.model_validate(
            base_conv._build_replay_payload(
                f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
                ReplayStructuredResponse,
                None,
                response,
            )
        )
        modified = replay.model_copy(update={"model": "grok-4.1-fast"})
        assert modified.model == "grok-4.1-fast"

        result = await modified.execute(replay_database)
        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayStructuredResponse)
        assert result.parsed.requested_token == token

    @pytest.mark.asyncio
    async def test_replay_with_changed_prompt(self, tmp_path: Path) -> None:
        """Capture with token_a, replay with token_b, verify only token_b appears."""
        from ai_pipeline_core.replay import ConversationReplay

        replay_database = FilesystemDatabase(tmp_path)
        token_a = uuid.uuid4().hex[:10].upper()
        token_b = uuid.uuid4().hex[:10].upper()
        base_conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await base_conv.send(f"Reply with only this token: {token_a}")

        response = cast(ModelResponse[Any], conv.messages[-1])
        replay = ConversationReplay.model_validate(base_conv._build_replay_payload(f"Reply with only this token: {token_a}", None, None, response))
        modified = replay.model_copy(update={"prompt": f"Reply with only this token: {token_b}"})

        result = await modified.execute(replay_database)
        assert token_b in result.content
        assert token_a not in result.content

    @pytest.mark.asyncio
    async def test_replay_with_changed_model_options(self, tmp_path: Path) -> None:
        """Capture with reasoning_effort=low, replay with reasoning_effort=high."""
        from ai_pipeline_core.replay import ConversationReplay

        replay_database = FilesystemDatabase(tmp_path)
        token = uuid.uuid4().hex[:10].upper()
        base_conv = Conversation(
            model=DEFAULT_MODEL,
            enable_substitutor=False,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
        )
        conv = await base_conv.send_structured(
            f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
            response_format=ReplayStructuredResponse,
        )
        assert conv.parsed is not None

        response = cast(ModelResponse[Any], conv.messages[-1])
        replay = ConversationReplay.model_validate(
            base_conv._build_replay_payload(
                f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
                ReplayStructuredResponse,
                None,
                response,
            )
        )
        modified = replay.model_copy(update={"model_options": {**replay.model_options, "reasoning_effort": "high", "max_completion_tokens": 2000}})

        result = await modified.execute(replay_database)
        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayStructuredResponse)
        assert result.parsed.requested_token == token

    @pytest.mark.asyncio
    async def test_structured_output_replay(self, tmp_path: Path) -> None:
        """Verify response_format round-trips through YAML and produces typed .parsed."""
        from ai_pipeline_core.replay import ConversationReplay

        replay_database = FilesystemDatabase(tmp_path)
        token = uuid.uuid4().hex[:10].upper()
        base_conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await base_conv.send_structured(
            f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
            response_format=ReplayStructuredResponse,
        )
        assert conv.parsed is not None

        response = cast(ModelResponse[Any], conv.messages[-1])
        replay = ConversationReplay.model_validate(
            base_conv._build_replay_payload(
                f"Summarize the prompt, return the requested_token as '{token}', set validation_passed=true.",
                ReplayStructuredResponse,
                None,
                response,
            )
        )
        yaml_text = replay.to_yaml()
        response_format_path = f"{ReplayStructuredResponse.__module__}:{ReplayStructuredResponse.__qualname__}"
        assert response_format_path in yaml_text or "ReplayStructuredResponse" in yaml_text

        restored_replay = ConversationReplay.from_yaml(yaml_text)
        assert restored_replay.response_format is not None

        result = await restored_replay.execute(replay_database)
        assert result.parsed is not None
        assert isinstance(result.parsed, ReplayStructuredResponse)
        assert result.parsed.requested_token == token

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_replay(self, tmp_path: Path) -> None:
        """Replay turn 2 of a multi-turn conversation with history from turn 1."""
        from ai_pipeline_core.replay import ConversationReplay

        replay_database = FilesystemDatabase(tmp_path)
        token = uuid.uuid4().hex[:10].upper()

        conv = Conversation(
            model=DEFAULT_MODEL,
            model_options=ModelOptions(max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort="low"),
            enable_substitutor=False,
        )
        conv = await conv.send(f"Remember codename: {token}")
        assert conv.content

        conv_with_history = conv
        conv = await conv.send_structured("Return the codename I told you.", response_format=ReplayMultiTurnResponse)
        assert conv.parsed is not None
        assert token in conv.parsed.codename.upper()

        response = cast(ModelResponse[Any], conv.messages[-1])
        replay = ConversationReplay.model_validate(
            conv_with_history._build_replay_payload("Return the codename I told you.", ReplayMultiTurnResponse, None, response)
        )

        has_user = any(e.type == "user_text" for e in replay.history)
        has_response = any(e.type == "response" for e in replay.history)
        assert has_user, "History must contain user_text entry from turn 1"
        assert has_response, "History must contain response entry from turn 1"

        yaml_text = replay.to_yaml()
        restored = ConversationReplay.from_yaml(yaml_text)
        result = await restored.execute(replay_database)

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
        replay_database = await _create_replay_database(tmp_path, doc)

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

        result = await replay.execute(replay_database)
        assert result.content
        assert "330" in result.content
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_pipeline_task_replay(
        self,
        tmp_path: Path,
    ) -> None:
        """Task replay payload persisted on the task node round-trips through YAML and re-executes."""
        from uuid import uuid4

        from ai_pipeline_core.replay import TaskReplay as TaskReplayType
        from ai_pipeline_core.database import MemoryDatabase, NodeKind

        token = uuid.uuid4().hex[:10].upper()
        database = MemoryDatabase()

        source_doc = ReplayTaskInputDocument.create_root(
            name="source.txt",
            content=f"Project code: {token}. Extract the code from this document.",
            reason="test input",
            description="Source document for task replay",
        )
        replay_database = await _create_replay_database(tmp_path, source_doc)

        with pipeline_test_context() as ctx:
            deployment_id = uuid4()
            ctx.database = database
            ctx.deployment_id = deployment_id
            ctx.root_deployment_id = deployment_id
            ctx.deployment_name = "replay-integration"
            result_docs: list[Any] = await ReplayIntegrationTask.run(source_doc, model_name=DEFAULT_MODEL)

        assert len(result_docs) == 1
        result_doc = result_docs[0]
        assert isinstance(result_doc, ReplayTaskOutputDocument)
        parsed = ReplayTaskExtraction.model_validate_json(result_doc.content)
        assert token in parsed.extracted_code.upper() or token in parsed.source_summary.upper()

        task_nodes = [node for node in database._nodes.values() if node.node_kind == NodeKind.TASK]
        assert len(task_nodes) == 1
        task_replay = TaskReplayType.model_validate(task_nodes[0].payload["replay_payload"])

        yaml_text = task_replay.to_yaml()
        restored = TaskReplayType.from_yaml(yaml_text)
        replay_result = await restored.execute(replay_database)

        assert isinstance(replay_result, tuple)
        assert len(replay_result) == 1
        assert isinstance(replay_result[0], ReplayTaskOutputDocument)
        replay_parsed = ReplayTaskExtraction.model_validate_json(replay_result[0].content)
        assert token in replay_parsed.extracted_code.upper() or token in replay_parsed.source_summary.upper()
