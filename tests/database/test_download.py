"""Tests for download_deployment snapshot export."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from ai_pipeline_core.database import ExecutionNode, MemoryDatabase, NodeKind, NodeStatus
from ai_pipeline_core.database._download import download_deployment
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.replay import DocumentRef
from ai_pipeline_core.replay._resolve import resolve_document_ref
from tests.replay.conftest import store_document_in_database


class _DownloadReplayDocument(Document):
    """Document type used for replay-payload export tests."""


def _doc_ref(document: Document) -> dict[str, str]:
    return {
        "$doc_ref": document.sha256,
        "class_name": type(document).__name__,
        "name": document.name,
    }


def _make_node(**kwargs: object) -> ExecutionNode:
    deployment_id = kwargs.pop("_deployment_id", None) or uuid4()
    root_deployment_id = kwargs.pop("_root_deployment_id", None) or deployment_id
    defaults: dict[str, object] = {
        "node_id": uuid4(),
        "node_kind": NodeKind.TASK,
        "deployment_id": deployment_id,
        "root_deployment_id": root_deployment_id,
        "run_id": f"download-{uuid4().hex[:8]}",
        "run_scope": "download-test/scope",
        "deployment_name": "download-test",
        "name": "DownloadTask",
        "sequence_no": 0,
        "status": NodeStatus.COMPLETED,
        "started_at": datetime.now(UTC),
        "ended_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return ExecutionNode(**defaults)  # type: ignore[arg-type]


class TestDownloadDeployment:
    @pytest.mark.asyncio
    async def test_download_preserves_nested_task_parent_order(self, tmp_path: Path) -> None:
        source = MemoryDatabase()
        deployment_id = uuid4()
        deployment = _make_node(
            node_id=deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            run_id="download-parent-order",
            name="download-parent-order",
            sequence_no=0,
        )
        flow = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            parent_node_id=deployment.node_id,
            run_id=deployment.run_id,
            name="DownloadFlow",
            sequence_no=1,
        )
        parent_task = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            parent_node_id=flow.node_id,
            flow_id=flow.node_id,
            run_id=deployment.run_id,
            name="ParentTask",
            sequence_no=3,
        )
        child_task = _make_node(
            node_kind=NodeKind.TASK,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            parent_node_id=parent_task.node_id,
            flow_id=flow.node_id,
            run_id=deployment.run_id,
            name="ChildTask",
            sequence_no=0,
        )

        for node in (deployment, flow, parent_task, child_task):
            await source.insert_node(node)

        download_dir = tmp_path / "downloaded"
        await download_deployment(source, deployment_id, download_dir)

        snapshot = FilesystemDatabase(download_dir)
        parent_path = snapshot._node_index[parent_task.node_id]
        child_path = snapshot._node_index[child_task.node_id]
        assert parent_path.parent in child_path.parents

    @pytest.mark.asyncio
    async def test_download_exports_replay_payload_document_records_and_blobs(self, tmp_path: Path) -> None:
        source = MemoryDatabase()
        current_deployment_id = uuid4()
        external_deployment_id = uuid4()
        prompt_document = _DownloadReplayDocument.create_root(
            name="prompt-bundle.md",
            content=b"# Prompt bundle\ncontent",
            reason="download replay payload test",
            attachments=(
                Attachment(
                    name="details.txt",
                    content=b"attachment details",
                    description="text attachment",
                ),
            ),
        )
        history_document = _DownloadReplayDocument.create_root(
            name="history.txt",
            content=b"history-only replay reference",
            reason="download replay payload test",
        )
        await store_document_in_database(source, prompt_document, deployment_id=external_deployment_id)
        await store_document_in_database(source, history_document, deployment_id=external_deployment_id)

        deployment = _make_node(
            node_id=current_deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=current_deployment_id,
            _root_deployment_id=current_deployment_id,
            run_id="download-replay-payload",
            name="download-replay-payload",
            sequence_no=0,
        )
        flow = _make_node(
            node_kind=NodeKind.FLOW,
            _deployment_id=current_deployment_id,
            _root_deployment_id=current_deployment_id,
            parent_node_id=current_deployment_id,
            run_id=deployment.run_id,
            name="ReplayFlow",
            sequence_no=1,
        )
        conversation_turn = _make_node(
            node_kind=NodeKind.CONVERSATION_TURN,
            _deployment_id=current_deployment_id,
            _root_deployment_id=current_deployment_id,
            parent_node_id=flow.node_id,
            flow_id=flow.node_id,
            run_id=deployment.run_id,
            name="turn_0",
            sequence_no=0,
            payload={
                "replay_payload": {
                    "version": 1,
                    "payload_type": "conversation",
                    "model": "gemini-3-flash",
                    "prompt": "",
                    "prompt_documents": [_doc_ref(prompt_document)],
                    "context": [],
                    "history": [
                        {
                            "type": "document",
                            **_doc_ref(history_document),
                        }
                    ],
                }
            },
        )

        for node in (deployment, flow, conversation_turn):
            await source.insert_node(node)

        download_dir = tmp_path / "downloaded"
        await download_deployment(source, current_deployment_id, download_dir)

        snapshot = FilesystemDatabase(download_dir)
        resolved = await resolve_document_ref(DocumentRef.model_validate(_doc_ref(prompt_document)), snapshot)
        downloaded_history = await snapshot.get_document(history_document.sha256)

        assert resolved.attachments
        assert resolved.attachments[0].content == b"attachment details"
        assert downloaded_history is not None

    @pytest.mark.asyncio
    async def test_download_exports_documents_from_earlier_deployment_chain(self, tmp_path: Path) -> None:
        source = MemoryDatabase()
        root_deployment_id = uuid4()
        current_deployment_id = uuid4()
        chain_document = _DownloadReplayDocument.create_root(
            name="chain-only.txt",
            content=b"needed for replay",
            reason="download chain export test",
        )
        await store_document_in_database(source, chain_document, deployment_id=root_deployment_id)

        root_deployment = _make_node(
            node_id=root_deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=root_deployment_id,
            _root_deployment_id=root_deployment_id,
            run_id="download-chain-root",
            name="download-chain-root",
            sequence_no=0,
        )
        current_deployment = _make_node(
            node_id=current_deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=current_deployment_id,
            _root_deployment_id=root_deployment_id,
            run_id="download-chain-current",
            name="download-chain-current",
            sequence_no=0,
        )

        await source.insert_node(root_deployment)
        await source.insert_node(current_deployment)

        download_dir = tmp_path / "downloaded"
        await download_deployment(source, current_deployment_id, download_dir)

        snapshot = FilesystemDatabase(download_dir)
        downloaded = await snapshot.get_document(chain_document.sha256)
        assert downloaded is not None

    @pytest.mark.asyncio
    async def test_download_writes_empty_logs_file_when_deployment_has_no_logs(self, tmp_path: Path) -> None:
        source = MemoryDatabase()
        deployment_id = uuid4()
        deployment = _make_node(
            node_id=deployment_id,
            node_kind=NodeKind.DEPLOYMENT,
            _deployment_id=deployment_id,
            _root_deployment_id=deployment_id,
            run_id="download-no-logs",
            name="download-no-logs",
            sequence_no=0,
        )
        await source.insert_node(deployment)

        download_dir = tmp_path / "downloaded"
        await download_deployment(source, deployment_id, download_dir)

        logs_path = download_dir / "logs.jsonl"
        assert logs_path.exists()
        assert logs_path.read_text(encoding="utf-8") == ""
