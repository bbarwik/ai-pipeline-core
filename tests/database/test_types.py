"""Tests for database type models: ExecutionNode, DocumentRecord, BlobRecord, ExecutionLog."""

import dataclasses
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ai_pipeline_core.database._ddl import DDL_DOCUMENTS, DDL_EXECUTION_LOGS, DDL_EXECUTION_NODES
from ai_pipeline_core.database import NULL_PARENT, BlobRecord, DocumentRecord, ExecutionLog, ExecutionNode, NodeKind, NodeStatus


class TestNodeKind:
    def test_values(self) -> None:
        assert NodeKind.DEPLOYMENT == "deployment"
        assert NodeKind.FLOW == "flow"
        assert NodeKind.TASK == "task"
        assert NodeKind.CONVERSATION == "conversation"
        assert NodeKind.CONVERSATION_TURN == "conversation_turn"

    def test_all_values(self) -> None:
        assert len(NodeKind) == 5


class TestNodeStatus:
    def test_values(self) -> None:
        assert NodeStatus.RUNNING == "running"
        assert NodeStatus.COMPLETED == "completed"
        assert NodeStatus.FAILED == "failed"
        assert NodeStatus.CACHED == "cached"
        assert NodeStatus.SKIPPED == "skipped"

    def test_all_values(self) -> None:
        assert len(NodeStatus) == 5


class TestExecutionNode:
    def _make_node(self, **kwargs: object) -> ExecutionNode:
        deploy_id = uuid4()
        defaults: dict[str, object] = {
            "node_id": uuid4(),
            "node_kind": NodeKind.TASK,
            "deployment_id": deploy_id,
            "root_deployment_id": deploy_id,
            "run_id": "test-run",
            "run_scope": "test-run/scope",
            "deployment_name": "test-pipeline",
            "name": "MyTask",
            "sequence_no": 0,
        }
        defaults.update(kwargs)
        return ExecutionNode(**defaults)  # type: ignore[arg-type]

    def test_creation_with_defaults(self) -> None:
        node = self._make_node()
        assert node.status == NodeStatus.RUNNING
        assert node.attempt == 0
        assert node.cost_usd == 0.0
        assert node.tokens_input == 0
        assert node.tokens_reasoning == 0
        assert node.error_type == ""
        assert node.input_document_shas == ()
        assert node.output_document_shas == ()
        assert node.context_document_shas == ()
        assert node.payload == {}
        assert node.flow_class == ""
        assert node.task_class == ""
        assert node.cache_key == ""
        assert node.version == 1

    def test_frozen_immutability(self) -> None:
        node = self._make_node()
        with pytest.raises(dataclasses.FrozenInstanceError):
            node.name = "changed"  # type: ignore[misc]

    def test_started_at_defaults_to_utcnow(self) -> None:
        before = datetime.now(UTC)
        node = self._make_node()
        after = datetime.now(UTC)
        assert before <= node.started_at <= after
        assert before <= node.updated_at <= after

    def test_all_node_kinds_as_node_kind(self) -> None:
        for kind in NodeKind:
            node = self._make_node(node_kind=kind)
            assert node.node_kind == kind

    def test_document_sha_tuples(self) -> None:
        node = self._make_node(
            input_document_shas=("sha1", "sha2"),
            output_document_shas=("sha3",),
            context_document_shas=("sha4", "sha5", "sha6"),
        )
        assert len(node.input_document_shas) == 2
        assert len(node.output_document_shas) == 1
        assert len(node.context_document_shas) == 3

    def test_payload_dict(self) -> None:
        node = self._make_node(payload={"flow_plan": [{"name": "step1"}]})
        assert node.payload["flow_plan"] == [{"name": "step1"}]

    def test_cross_deployment_fields(self) -> None:
        child_id = uuid4()
        parent_task_id = uuid4()
        node = self._make_node(
            remote_child_deployment_id=child_id,
            parent_deployment_task_id=parent_task_id,
        )
        assert node.remote_child_deployment_id == child_id
        assert node.parent_deployment_task_id == parent_task_id


class TestDocumentRecord:
    def _make_record(self, **kwargs: object) -> DocumentRecord:
        defaults: dict[str, object] = {
            "document_sha256": "abc123",
            "content_sha256": "def456",
            "deployment_id": uuid4(),
            "producing_node_id": uuid4(),
            "document_type": "ResearchReport",
            "name": "report.md",
        }
        defaults.update(kwargs)
        return DocumentRecord(**defaults)  # type: ignore[arg-type]

    def test_creation_with_defaults(self) -> None:
        record = self._make_record()
        assert record.run_scope == ""
        assert record.description == ""
        assert record.mime_type == ""
        assert record.size_bytes == 0
        assert not record.publicly_visible
        assert record.derived_from == ()
        assert record.triggered_by == ()
        assert record.attachment_names == ()
        assert record.summary == ""
        assert record.metadata_json == "{}"
        assert record.version == 1

    def test_frozen_immutability(self) -> None:
        record = self._make_record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.name = "changed"  # type: ignore[misc]

    def test_null_producing_node_id(self) -> None:
        record = self._make_record(producing_node_id=None)
        assert record.producing_node_id is None

    def test_attachment_parallel_arrays(self) -> None:
        record = self._make_record(
            attachment_names=("screenshot.png",),
            attachment_descriptions=("Page screenshot",),
            attachment_sha256s=("att_sha1",),
            attachment_mime_types=("image/png",),
            attachment_sizes=(1024,),
        )
        assert record.attachment_names == ("screenshot.png",)
        assert record.attachment_sizes == (1024,)


class TestBlobRecord:
    def test_creation(self) -> None:
        blob = BlobRecord(content_sha256="abc123", content=b"hello world", size_bytes=11)
        assert blob.content == b"hello world"
        assert blob.size_bytes == 11

    def test_frozen_immutability(self) -> None:
        blob = BlobRecord(content_sha256="abc123", content=b"data", size_bytes=4)
        with pytest.raises(dataclasses.FrozenInstanceError):
            blob.content = b"changed"  # type: ignore[misc]

    def test_created_at_default(self) -> None:
        before = datetime.now(UTC)
        blob = BlobRecord(content_sha256="abc", content=b"x", size_bytes=1)
        after = datetime.now(UTC)
        assert before <= blob.created_at <= after


class TestExecutionLog:
    def test_creation(self) -> None:
        deployment_id = uuid4()
        node_id = uuid4()
        flow_id = uuid4()
        task_id = uuid4()
        timestamp = datetime.now(UTC)
        log = ExecutionLog(
            node_id=node_id,
            deployment_id=deployment_id,
            root_deployment_id=deployment_id,
            flow_id=flow_id,
            task_id=task_id,
            timestamp=timestamp,
            sequence_no=3,
            level="INFO",
            category="lifecycle",
            logger_name="ai_pipeline_core.pipeline",
            message="Started task",
        )
        assert log.node_id == node_id
        assert log.flow_id == flow_id
        assert log.task_id == task_id
        assert log.fields == "{}"
        assert log.exception_text == ""

    def test_frozen_immutability(self) -> None:
        deployment_id = uuid4()
        log = ExecutionLog(
            node_id=uuid4(),
            deployment_id=deployment_id,
            root_deployment_id=deployment_id,
            flow_id=None,
            task_id=None,
            timestamp=datetime.now(UTC),
            sequence_no=0,
            level="INFO",
            category="framework",
            logger_name="ai_pipeline_core.tests",
            message="hello",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            log.message = "changed"  # type: ignore[misc]


class TestNullParent:
    def test_default_parent_is_null_parent(self) -> None:
        deploy_id = uuid4()
        node = ExecutionNode(
            node_id=uuid4(),
            node_kind=NodeKind.TASK,
            deployment_id=deploy_id,
            root_deployment_id=deploy_id,
            run_id="test",
            run_scope="test/scope",
            deployment_name="test",
            name="Task",
            sequence_no=0,
        )
        assert node.parent_node_id == NULL_PARENT

    def test_null_parent_is_zero_uuid(self) -> None:
        assert str(NULL_PARENT) == "00000000-0000-0000-0000-000000000000"


class TestAttachmentValidation:
    def _make_record(self, **kwargs: object) -> DocumentRecord:
        defaults: dict[str, object] = {
            "document_sha256": "abc123",
            "content_sha256": "def456",
            "deployment_id": uuid4(),
            "producing_node_id": uuid4(),
            "document_type": "TestDoc",
            "name": "test.md",
        }
        defaults.update(kwargs)
        return DocumentRecord(**defaults)  # type: ignore[arg-type]

    def test_equal_length_arrays_accepted(self) -> None:
        record = self._make_record(
            attachment_names=("a.png",),
            attachment_descriptions=("desc",),
            attachment_sha256s=("sha",),
            attachment_mime_types=("image/png",),
            attachment_sizes=(100,),
        )
        assert len(record.attachment_names) == 1

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="Attachment parallel arrays must have equal lengths"):
            self._make_record(
                attachment_names=("a.png", "b.png"),
                attachment_descriptions=("desc",),
                attachment_sha256s=("sha",),
                attachment_mime_types=("image/png",),
                attachment_sizes=(100,),
            )

    def test_empty_arrays_accepted(self) -> None:
        record = self._make_record()
        assert record.attachment_names == ()


class TestDatabaseDdl:
    def test_execution_nodes_uses_versioned_replacing_merge_tree(self) -> None:
        assert "ENGINE = ReplacingMergeTree(version)" in DDL_EXECUTION_NODES
        assert "version UInt64 DEFAULT 1" in DDL_EXECUTION_NODES
        assert "INDEX idx_node_id node_id TYPE bloom_filter GRANULARITY 1" in DDL_EXECUTION_NODES
        assert "INDEX idx_status status TYPE set(8) GRANULARITY 1" in DDL_EXECUTION_NODES

    def test_documents_ddl_includes_run_scope_indexes_and_codec(self) -> None:
        assert "run_scope String DEFAULT ''" in DDL_DOCUMENTS
        assert "metadata_json String DEFAULT '{}' CODEC(ZSTD(3))" in DDL_DOCUMENTS
        assert "document_type LowCardinality(String)" in DDL_DOCUMENTS
        assert "mime_type LowCardinality(String) DEFAULT ''" in DDL_DOCUMENTS
        assert "INDEX idx_name name TYPE bloom_filter GRANULARITY 1" in DDL_DOCUMENTS

    def test_execution_logs_ddl_exists(self) -> None:
        assert "CREATE TABLE IF NOT EXISTS execution_logs" in DDL_EXECUTION_LOGS
        assert "sequence_no UInt32" in DDL_EXECUTION_LOGS
        assert "category LowCardinality(String)" in DDL_EXECUTION_LOGS
        assert "TTL toDateTime(timestamp) + INTERVAL 90 DAY" in DDL_EXECUTION_LOGS
