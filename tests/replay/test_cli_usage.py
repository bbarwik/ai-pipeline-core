"""CLI usage examples for the replay module."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

from ai_pipeline_core import Document, PipelineTask
from ai_pipeline_core.database import ExecutionNode, MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.database._download import download_deployment
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.replay import ConversationReplay
from ai_pipeline_core.replay.cli import _apply_overrides, main
from ai_pipeline_core.replay.types import _infer_db_path
from tests.replay.conftest import store_document_in_database


class _DownloadedBundleInputDocument(Document):
    """Input document used to verify replay from a downloaded bundle."""


class _DownloadedBundleOutputDocument(Document):
    """Output document returned by the downloaded-bundle replay task."""


class _DownloadedBundleReplayTask(PipelineTask):
    @classmethod
    async def run(cls, source: _DownloadedBundleInputDocument, label: str) -> tuple[_DownloadedBundleOutputDocument, ...]:
        _ = cls
        return (
            _DownloadedBundleOutputDocument(
                name="bundle-result.txt",
                content=source.content,
                description=label,
            ),
        )


def _conversation_yaml(tmp_path: Path, **overrides: object) -> Path:
    data: dict[str, object] = {
        "version": 1,
        "payload_type": "conversation",
        "model": "gemini-3-flash",
        "prompt": "Summarize the document.",
        "context": [],
        "history": [],
    }
    data.update(overrides)
    path = tmp_path / "conversation.yaml"
    path.write_text(yaml.dump(data))
    return path


class _MockResult:
    def __init__(self, content: str = "LLM response") -> None:
        self.content = content
        self.usage = SimpleNamespace(total_tokens=150, model_dump=lambda: {"total_tokens": 150})
        self.cost = 0.003
        self.parsed = None


@pytest.mark.ai_docs
def test_main_show_conversation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-pro")

    exit_code = main(["show", str(replay_file)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "ConversationReplay" in output
    assert "gemini-3-pro" in output


@pytest.mark.ai_docs
def test_main_run_with_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_file = _conversation_yaml(tmp_path)
    db_path = tmp_path / "bundle"
    FilesystemDatabase(db_path)

    async def fake_execute(payload: object, database: object) -> _MockResult:
        _ = (payload, database)
        return _MockResult()

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)

    exit_code = main(["run", str(replay_file), "--db-path", str(db_path)])

    assert exit_code == 0
    output_dir = tmp_path / "conversation_replay"
    assert (output_dir / "output.yaml").exists()


@pytest.mark.ai_docs
def test_main_run_with_set_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-flash")
    db_path = tmp_path / "bundle"
    FilesystemDatabase(db_path)

    captured_models: list[str] = []

    async def fake_execute(payload: object, database: object) -> _MockResult:
        _ = database
        captured_models.append(payload.model)
        return _MockResult()

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)

    exit_code = main(["run", str(replay_file), "--db-path", str(db_path), "--set", "model=grok-4.1-fast"])

    assert exit_code == 0
    assert captured_models == ["grok-4.1-fast"]


@pytest.mark.ai_docs
def test_main_run_with_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_file = _conversation_yaml(tmp_path)
    db_path = tmp_path / "bundle"
    FilesystemDatabase(db_path)
    custom_dir = tmp_path / "my_output"

    async def fake_execute(payload: object, database: object) -> _MockResult:
        _ = (payload, database)
        return _MockResult()

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)

    exit_code = main(["run", str(replay_file), "--db-path", str(db_path), "--output-dir", str(custom_dir)])

    assert exit_code == 0
    assert (custom_dir / "output.yaml").exists()


@pytest.mark.ai_docs
def test_main_run_from_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.TASK,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="ReplayTask",
                sequence_no=0,
                started_at=datetime.now(UTC),
                payload={
                    "replay_payload": {
                        "version": 1,
                        "payload_type": "conversation",
                        "model": "gemini-3-flash",
                        "prompt": "Replay from db",
                        "context": [],
                        "history": [],
                    }
                },
            )
        )

    asyncio.run(seed())

    async def fake_execute(payload: object, database_obj: object) -> _MockResult:
        _ = (payload, database_obj)
        return _MockResult("DB replay")

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 0
    assert (tmp_path / f"node_{str(node_id)[:8]}_replay" / "output.yaml").exists()


def test_main_run_from_db_rejects_conversation_parent_node(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.CONVERSATION,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="ConversationParent",
                sequence_no=0,
                started_at=datetime.now(UTC),
                payload={"turn_count": 1},
            )
        )

    asyncio.run(seed())

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 1
    assert "conversation_turn, task, and flow" in capsys.readouterr().err


def test_apply_overrides_coerces_boolean_integer_and_mapping_types() -> None:
    payload = ConversationReplay(
        model="gemini-3-flash",
        prompt="hello",
        context=(),
        history=(),
        enable_substitutor=True,
    )

    updated = _apply_overrides(
        payload,
        [
            "enable_substitutor=false",
            "version=2",
            "model_options={timeout: 5}",
        ],
    )

    assert updated.enable_substitutor is False
    assert updated.version == 2
    assert updated.model_options == {"timeout": 5}


def test_main_run_from_db_rejects_flow_without_replay_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.FLOW,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                parent_node_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="CachedFlow",
                sequence_no=1,
                status=NodeStatus.CACHED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={"skip_reason": "cached result"},
            )
        )

    asyncio.run(seed())

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 1
    assert "has no replay payload and cannot be replayed" in capsys.readouterr().err


def test_main_run_from_db_missing_payload_message_does_not_claim_failed_nodes_must_succeed_first(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.FLOW,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                parent_node_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="FailedFlow",
                sequence_no=1,
                status=NodeStatus.FAILED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={"error_message": "boom"},
            )
        )

    asyncio.run(seed())

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 1
    assert "executed normally" not in capsys.readouterr().err


@pytest.mark.ai_docs
def test_ConversationReplay_from_yaml_and_override() -> None:
    yaml_text = yaml.dump({
        "version": 1,
        "payload_type": "conversation",
        "model": "gemini-3-flash",
        "prompt": "Analyze the report.",
        "context": [],
        "history": [],
    })

    replay = ConversationReplay.from_yaml(yaml_text)
    assert replay.model == "gemini-3-flash"

    modified = replay.model_copy(update={"model": "grok-4.1-fast"})
    assert modified.model == "grok-4.1-fast"
    assert modified.prompt == replay.prompt
    assert "grok-4.1-fast" in modified.to_yaml()


@pytest.mark.ai_docs
def test__infer_db_path_from_filesystem_database_root(tmp_path: Path) -> None:
    store_dir = tmp_path / "pipeline_output"
    replay_dir = store_dir / "runs" / "20260308_pipeline_abcd1234" / "flows" / "01_flow_1234"
    replay_dir.mkdir(parents=True)
    (store_dir / "blobs").mkdir(parents=True)
    replay_file = replay_dir / "conversation.yaml"
    replay_file.write_text("dummy")

    result = _infer_db_path(replay_file)
    assert result == store_dir


def test__infer_db_path_missing_root_raises(tmp_path: Path) -> None:
    replay_file = tmp_path / "conversation.yaml"
    replay_file.write_text("dummy")

    with pytest.raises(FileNotFoundError):
        _infer_db_path(replay_file)


def test_downloaded_bundle_can_be_replayed_via_from_db_cli(tmp_path: Path) -> None:
    source_database = MemoryDatabase()
    deployment_id = uuid4()
    flow_id = uuid4()
    task_id = uuid4()

    async def seed_and_download() -> tuple[ExecutionNode | None, object | None]:
        source_doc = _DownloadedBundleInputDocument(
            name="bundle-input.txt",
            content=b"downloaded bundle replay input",
            description="Seed input for downloaded bundle replay",
        )
        await store_document_in_database(source_database, source_doc, deployment_id=deployment_id)

        deployment_node = ExecutionNode(
            node_id=deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            deployment_id=deployment_id,
            root_deployment_id=deployment_id,
            run_id="bundle-run",
            run_scope="bundle-run/scope",
            deployment_name="bundle-test",
            name="bundle-test",
            sequence_no=0,
            status=NodeStatus.COMPLETED,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            payload={"flow_plan": [], "options": {}, "parent_execution_id": ""},
        )
        flow_node = ExecutionNode(
            node_id=flow_id,
            node_kind=NodeKind.FLOW,
            deployment_id=deployment_id,
            root_deployment_id=deployment_id,
            parent_node_id=deployment_id,
            run_id="bundle-run",
            run_scope="bundle-run/scope",
            deployment_name="bundle-test",
            name="BundleFlow",
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
            parent_node_id=flow_id,
            run_id="bundle-run",
            run_scope="bundle-run/scope",
            deployment_name="bundle-test",
            name="BundleReplayTask",
            sequence_no=1,
            flow_id=flow_id,
            status=NodeStatus.COMPLETED,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            input_document_shas=(source_doc.sha256,),
            payload={
                "replay_payload": {
                    "version": 1,
                    "payload_type": "pipeline_task",
                    "function_path": f"{_DownloadedBundleReplayTask.__module__}:{_DownloadedBundleReplayTask.__qualname__}",
                    "arguments": {
                        "source": {
                            "$doc_ref": source_doc.sha256,
                            "class_name": type(source_doc).__name__,
                            "name": source_doc.name,
                        },
                        "label": "replayed-from-download",
                    },
                    "original": {},
                }
            },
        )

        await source_database.insert_node(deployment_node)
        await source_database.insert_node(flow_node)
        await source_database.insert_node(task_node)

        download_dir = tmp_path / "downloaded"
        await download_deployment(source_database, deployment_id, download_dir)

        downloaded_database = FilesystemDatabase(download_dir)
        downloaded_task = await downloaded_database.get_node(task_id)
        downloaded_document = await downloaded_database.get_document(source_doc.sha256)
        return downloaded_task, downloaded_document

    downloaded_task, downloaded_document = asyncio.run(seed_and_download())

    assert downloaded_task is not None
    assert downloaded_document is not None

    replay_output_dir = tmp_path / "downloaded-replay-output"
    exit_code = main([
        "run",
        "--from-db",
        str(task_id),
        "--db-path",
        str(tmp_path / "downloaded"),
        "--output-dir",
        str(replay_output_dir),
    ])

    assert exit_code == 0
    output = yaml.safe_load((replay_output_dir / "output.yaml").read_text())
    assert output["type"] == "document_list"
    assert output["documents"][0]["name"] == "bundle-result.txt"
