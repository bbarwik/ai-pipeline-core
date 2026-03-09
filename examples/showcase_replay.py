#!/usr/bin/env python3
"""TaskReplay showcase using MemoryDatabase-backed document resolution."""

import asyncio
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from ai_pipeline_core import Document, RunScope
from ai_pipeline_core.database import (
    NULL_PARENT,
    BlobRecord,
    DatabaseReader,
    DatabaseWriter,
    DocumentRecord,
    ExecutionNode,
    MemoryDatabase,
    NodeKind,
    NodeStatus,
)
from ai_pipeline_core.pipeline import PipelineTask
from ai_pipeline_core.replay import TaskReplay


class ReplaySourceDocument(Document):
    """Input document referenced from the replay payload."""


class ReplayOutputDocument(Document):
    """Output document returned by the replayed task."""


class ReplayUppercaseTask(PipelineTask):
    """Minimal task used by the replay showcase."""

    name = "replay_uppercase"

    @classmethod
    async def run(cls, documents: tuple[ReplaySourceDocument, ...]) -> tuple[ReplayOutputDocument, ...]:
        source = documents[0]
        return (
            ReplayOutputDocument.derive(
                from_documents=(source,),
                name="replayed_notes.txt",
                content=source.text.upper(),
            ),
        )


def _completed_node(
    *,
    node_id: UUID,
    deployment_id: UUID,
    run_id: str,
    run_scope: RunScope,
    name: str,
    node_kind: NodeKind,
    sequence_no: int,
    parent_node_id: UUID,
    payload: dict[str, object] | None = None,
) -> ExecutionNode:
    timestamp = datetime.now(UTC)
    return ExecutionNode(
        node_id=node_id,
        node_kind=node_kind,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        run_id=run_id,
        run_scope=run_scope,
        deployment_name="replay-showcase",
        name=name,
        sequence_no=sequence_no,
        parent_node_id=parent_node_id,
        status=NodeStatus.COMPLETED,
        started_at=timestamp,
        ended_at=timestamp,
        updated_at=timestamp,
        payload=payload or {},
    )


async def _store_document(
    writer: DatabaseWriter,
    *,
    document: Document,
    deployment_id: UUID,
    run_scope: RunScope,
) -> DocumentRecord:
    record = DocumentRecord(
        document_sha256=document.sha256,
        content_sha256=document.content_sha256,
        deployment_id=deployment_id,
        producing_node_id=None,
        document_type=type(document).__name__,
        name=document.name,
        run_scope=run_scope,
        description=document.description or "",
        mime_type=document.mime_type,
        size_bytes=document.size,
        derived_from=document.derived_from,
        triggered_by=document.triggered_by,
    )
    await writer.save_blob(
        BlobRecord(
            content_sha256=document.content_sha256,
            content=document.content,
            size_bytes=len(document.content),
        )
    )
    await writer.save_document(record)
    return record


def _require_replay_outputs(value: tuple[object, ...] | object) -> tuple[ReplayOutputDocument, ...]:
    """Validate replay execution output for the example before printing it."""
    if not isinstance(value, tuple):
        raise TypeError(f"Expected TaskReplay.execute() to return a tuple, got {type(value).__name__}.")

    raw_outputs = cast(tuple[object, ...], value)
    outputs: list[ReplayOutputDocument] = []
    for item in raw_outputs:
        if not isinstance(item, ReplayOutputDocument):
            raise TypeError(f"Expected TaskReplay.execute() to return only ReplayOutputDocument instances. Got {type(item).__name__}.")
        outputs.append(item)
    return tuple(outputs)


async def main() -> None:
    database = MemoryDatabase()
    writer: DatabaseWriter = database
    reader: DatabaseReader = database

    deployment_id = uuid4()
    run_scope = RunScope("examples/replay/main")
    deployment_node = _completed_node(
        node_id=deployment_id,
        deployment_id=deployment_id,
        run_id="examples-replay",
        run_scope=run_scope,
        name="replay-showcase",
        node_kind=NodeKind.DEPLOYMENT,
        sequence_no=0,
        parent_node_id=NULL_PARENT,
    )
    await writer.insert_node(deployment_node)

    source = ReplaySourceDocument.create_root(
        name="notes.txt",
        content="Replay payloads resolve documents by SHA256 from the database.",
        reason="seed a document for TaskReplay resolution",
    )
    source_record = await _store_document(
        writer,
        document=source,
        deployment_id=deployment_id,
        run_scope=run_scope,
    )

    payload = TaskReplay(
        function_path=f"{ReplayUppercaseTask.__module__}:{ReplayUppercaseTask.__qualname__}",
        arguments={
            "documents": [
                {
                    "$doc_ref": source.sha256,
                    "class_name": type(source).__name__,
                    "name": source.name,
                }
            ]
        },
    )
    task_node_id = uuid4()
    await writer.insert_node(
        _completed_node(
            node_id=task_node_id,
            deployment_id=deployment_id,
            run_id="examples-replay",
            run_scope=run_scope,
            name="replay-upper-case",
            node_kind=NodeKind.TASK,
            sequence_no=1,
            parent_node_id=deployment_node.node_id,
            payload={"replay_payload": payload.model_dump(mode="json", by_alias=True)},
        )
    )

    stored_record = await reader.get_document(source_record.document_sha256)
    stored_node = await reader.get_node(task_node_id)
    if stored_record is None or stored_node is None:
        raise RuntimeError("Expected replay inputs to be stored in MemoryDatabase.")

    stored_payload = TaskReplay.model_validate(stored_node.payload["replay_payload"])
    outputs = _require_replay_outputs(await stored_payload.execute(reader))

    print("Stored document:")
    print(f"  - {stored_record.name} [{stored_record.document_type}]")

    print("\nReplay payload YAML:")
    print(payload.to_yaml().strip())

    print("\nReplay result:")
    for document in outputs:
        print(f"  - {document.name}: {document.text}")

    print("\nThe stored task node mirrors what ai-replay reads from payload['replay_payload'] when replaying from a database backend.")


if __name__ == "__main__":
    asyncio.run(main())
