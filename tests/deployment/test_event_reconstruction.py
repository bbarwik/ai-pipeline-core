"""Tests for _reconstruct_lifecycle_events."""

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from ai_pipeline_core.database import DocumentRecord, SpanKind, SpanRecord, SpanStatus
from ai_pipeline_core.database._memory import _MemoryDatabase
from ai_pipeline_core.deployment._event_reconstruction import _reconstruct_lifecycle_events
from ai_pipeline_core.deployment._types import EventType
from ai_pipeline_core.deployment._event_serialization import event_to_payload
from ai_pipeline_core.deployment._types import ErrorCode


def _make_span(**kwargs: object) -> SpanRecord:
    deployment_id = kwargs.pop("deployment_id", uuid4())
    root_deployment_id = kwargs.pop("root_deployment_id", deployment_id)
    started_at: datetime = kwargs.pop("started_at", datetime(2026, 3, 14, 12, 0, tzinfo=UTC))
    defaults: dict[str, object] = {
        "span_id": kwargs.pop("span_id", uuid4()),
        "parent_span_id": kwargs.pop("parent_span_id", None),
        "deployment_id": deployment_id,
        "root_deployment_id": root_deployment_id,
        "run_id": kwargs.pop("run_id", "test-run"),
        "deployment_name": "test-deploy",
        "kind": SpanKind.TASK,
        "name": "TestTask",
        "status": SpanStatus.COMPLETED,
        "sequence_no": 0,
        "started_at": started_at,
        "ended_at": started_at + timedelta(seconds=1),
        "version": 1,
        "meta_json": "",
        "metrics_json": "",
    }
    defaults.update(kwargs)
    return SpanRecord(**defaults)


def _make_document(**kwargs: object) -> DocumentRecord:
    defaults: dict[str, object] = {
        "document_sha256": f"doc-{uuid4().hex}",
        "content_sha256": f"blob-{uuid4().hex}",
        "document_type": "TestDocument",
        "name": "test.md",
        "mime_type": "text/markdown",
        "size_bytes": 10,
        "derived_from": (),
        "triggered_by": (),
    }
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


async def _seed_successful_run(db: _MemoryDatabase) -> tuple[UUID, UUID]:
    """Seed a complete deployment → flow → task span tree. Returns (root_deployment_id, deployment_span_id)."""
    root_id = uuid4()
    deploy_id = root_id
    deploy_span_id = uuid4()
    flow_span_id = uuid4()
    task_span_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    doc = _make_document(document_sha256="out-sha-1", publicly_visible=True)
    await db.save_document(doc)

    deploy = _make_span(
        span_id=deploy_span_id,
        deployment_id=deploy_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="TestDeploy",
        status=SpanStatus.COMPLETED,
        started_at=t0,
        ended_at=t0 + timedelta(seconds=10),
        meta_json=json.dumps({
            "input_fingerprint": "abc123",
            "flow_plan": [{"name": "Flow1", "flow_class": "TestFlow", "step": 1}],
            "deployment_class": "TestPipeline",
        }),
        output_json=json.dumps({"result": {"ok": True}}),
        output_document_shas=("out-sha-1",),
    )
    flow = _make_span(
        span_id=flow_span_id,
        parent_span_id=deploy_span_id,
        deployment_id=deploy_id,
        root_deployment_id=root_id,
        kind=SpanKind.FLOW,
        name="Flow1",
        status=SpanStatus.COMPLETED,
        started_at=t0 + timedelta(seconds=1),
        ended_at=t0 + timedelta(seconds=8),
        target="classmethod:my.mod:TestFlow.run",
        meta_json=json.dumps({
            "step": 1,
            "total_steps": 1,
            "expected_task_names": ["TestTask"],
        }),
        receiver_json=json.dumps({"mode": "constructor_args", "value": {"param": "val"}}),
        output_document_shas=("out-sha-1",),
    )
    task = _make_span(
        span_id=task_span_id,
        parent_span_id=flow_span_id,
        deployment_id=deploy_id,
        root_deployment_id=root_id,
        kind=SpanKind.TASK,
        name="TestTask",
        status=SpanStatus.COMPLETED,
        started_at=t0 + timedelta(seconds=2),
        ended_at=t0 + timedelta(seconds=5),
        target="classmethod:my.mod:TestTask.run",
        output_document_shas=("out-sha-1",),
    )
    for span in (deploy, flow, task):
        await db.insert_span(span)

    return root_id, deploy_span_id


@pytest.mark.asyncio
async def test_successful_run_reconstruction() -> None:
    db = _MemoryDatabase()
    root_id, _ = await _seed_successful_run(db)

    events = await _reconstruct_lifecycle_events(db, root_id)

    event_types = [e.event_type for e in events]
    assert EventType.RUN_STARTED in event_types
    assert EventType.FLOW_STARTED in event_types
    assert EventType.TASK_STARTED in event_types
    assert EventType.TASK_COMPLETED in event_types
    assert EventType.FLOW_COMPLETED in event_types
    assert EventType.RUN_COMPLETED in event_types

    run_started = next(e for e in events if e.event_type == EventType.RUN_STARTED)
    assert run_started.data["deployment_class"] == "TestPipeline"
    assert run_started.data["input_fingerprint"] == "abc123"
    assert len(run_started.data["flow_plan"]) == 1

    run_completed = next(e for e in events if e.event_type == EventType.RUN_COMPLETED)
    assert run_completed.data["result"] == {"ok": True}
    assert run_completed.data["duration_ms"] == 10000

    flow_started = next(e for e in events if e.event_type == EventType.FLOW_STARTED)
    assert flow_started.data["flow_name"] == "Flow1"
    assert flow_started.data["flow_params"] == {"param": "val"}
    assert flow_started.data["expected_tasks"] == ["TestTask"]

    flow_completed = next(e for e in events if e.event_type == EventType.FLOW_COMPLETED)
    assert len(flow_completed.data["output_documents"]) == 1
    assert flow_completed.data["output_documents"][0]["publicly_visible"] is True


@pytest.mark.asyncio
async def test_in_progress_run() -> None:
    db = _MemoryDatabase()
    root_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    deploy = _make_span(
        span_id=uuid4(),
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="RunningDeploy",
        status=SpanStatus.RUNNING,
        started_at=t0,
        ended_at=None,
        meta_json=json.dumps({"input_fingerprint": "fp", "deployment_class": "P", "flow_plan": []}),
    )
    await db.insert_span(deploy)

    events = await _reconstruct_lifecycle_events(db, root_id)
    event_types = [e.event_type for e in events]
    assert event_types == [EventType.RUN_STARTED]


@pytest.mark.asyncio
async def test_failed_run_with_error_code() -> None:
    db = _MemoryDatabase()
    root_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    deploy = _make_span(
        span_id=uuid4(),
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="FailedDeploy",
        status=SpanStatus.FAILED,
        started_at=t0,
        ended_at=t0 + timedelta(seconds=5),
        error_message="boom",
        meta_json=json.dumps({"input_fingerprint": "fp", "deployment_class": "P", "flow_plan": [], "error_code": "provider_error"}),
    )
    await db.insert_span(deploy)

    events = await _reconstruct_lifecycle_events(db, root_id)
    run_failed = next(e for e in events if e.event_type == EventType.RUN_FAILED)
    assert run_failed.data["error_code"] == "provider_error"
    assert run_failed.data["error_message"] == "boom"


@pytest.mark.asyncio
async def test_skipped_flow() -> None:
    db = _MemoryDatabase()
    root_id = uuid4()
    deploy_span_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    deploy = _make_span(
        span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="Deploy",
        status=SpanStatus.COMPLETED,
        started_at=t0,
        ended_at=t0 + timedelta(seconds=5),
        meta_json=json.dumps({"input_fingerprint": "fp", "deployment_class": "P", "flow_plan": []}),
        output_json=json.dumps({"result": {}}),
    )
    flow = _make_span(
        span_id=uuid4(),
        parent_span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.FLOW,
        name="SkippedFlow",
        status=SpanStatus.SKIPPED,
        started_at=t0 + timedelta(seconds=1),
        ended_at=t0 + timedelta(seconds=1),
        meta_json=json.dumps({"step": 1, "total_steps": 2, "skip_reason": "resumed_past_step"}),
    )
    for span in (deploy, flow):
        await db.insert_span(span)

    events = await _reconstruct_lifecycle_events(db, root_id)
    skipped = next(e for e in events if e.event_type == EventType.FLOW_SKIPPED)
    assert skipped.data["reason"] == "resumed_past_step"
    assert skipped.data["status"] == "skipped"

    flow_started_events = [e for e in events if e.event_type == EventType.FLOW_STARTED]
    assert len(flow_started_events) == 0


@pytest.mark.asyncio
async def test_cached_flow() -> None:
    db = _MemoryDatabase()
    root_id = uuid4()
    deploy_span_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    deploy = _make_span(
        span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="Deploy",
        status=SpanStatus.COMPLETED,
        started_at=t0,
        ended_at=t0 + timedelta(seconds=5),
        meta_json=json.dumps({"input_fingerprint": "fp", "deployment_class": "P", "flow_plan": []}),
        output_json=json.dumps({"result": {}}),
    )
    flow = _make_span(
        span_id=uuid4(),
        parent_span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.FLOW,
        name="CachedFlow",
        status=SpanStatus.CACHED,
        started_at=t0 + timedelta(seconds=1),
        ended_at=t0 + timedelta(seconds=1),
        meta_json=json.dumps({"step": 1, "total_steps": 1, "skip_reason": "cached_result_available"}),
    )
    for span in (deploy, flow):
        await db.insert_span(span)

    events = await _reconstruct_lifecycle_events(db, root_id)
    skipped = next(e for e in events if e.event_type == EventType.FLOW_SKIPPED)
    assert skipped.data["status"] == "cached"
    assert skipped.data["reason"] == "cached_result_available"


@pytest.mark.asyncio
async def test_cached_task_no_started_event() -> None:
    db = _MemoryDatabase()
    root_id = uuid4()
    deploy_span_id = uuid4()
    flow_span_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    deploy = _make_span(
        span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="Deploy",
        status=SpanStatus.COMPLETED,
        started_at=t0,
        ended_at=t0 + timedelta(seconds=5),
        meta_json=json.dumps({"input_fingerprint": "fp", "deployment_class": "P", "flow_plan": []}),
        output_json=json.dumps({"result": {}}),
    )
    flow = _make_span(
        span_id=flow_span_id,
        parent_span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.FLOW,
        name="Flow1",
        status=SpanStatus.COMPLETED,
        started_at=t0 + timedelta(seconds=1),
        ended_at=t0 + timedelta(seconds=4),
        meta_json=json.dumps({"step": 1, "total_steps": 1}),
    )
    task = _make_span(
        span_id=uuid4(),
        parent_span_id=flow_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.TASK,
        name="CachedTask",
        status=SpanStatus.CACHED,
        started_at=t0 + timedelta(seconds=2),
        ended_at=t0 + timedelta(seconds=3),
    )
    for span in (deploy, flow, task):
        await db.insert_span(span)

    events = await _reconstruct_lifecycle_events(db, root_id)

    task_started = [e for e in events if e.event_type == EventType.TASK_STARTED]
    assert len(task_started) == 0

    task_completed = [e for e in events if e.event_type == EventType.TASK_COMPLETED]
    assert len(task_completed) == 1
    assert task_completed[0].data["status"] == "cached"


@pytest.mark.asyncio
async def test_event_ordering_deterministic() -> None:
    db = _MemoryDatabase()
    root_id, _ = await _seed_successful_run(db)

    events = await _reconstruct_lifecycle_events(db, root_id)
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)

    types_in_order = [e.event_type for e in events]
    run_started_idx = types_in_order.index(EventType.RUN_STARTED)
    run_completed_idx = types_in_order.index(EventType.RUN_COMPLETED)
    assert run_started_idx < run_completed_idx


@pytest.mark.asyncio
async def test_empty_tree_returns_empty() -> None:
    db = _MemoryDatabase()
    events = await _reconstruct_lifecycle_events(db, uuid4())
    assert events == []


@pytest.mark.asyncio
async def test_operation_spans_are_ignored() -> None:
    db = _MemoryDatabase()
    root_id = uuid4()
    deploy_span_id = uuid4()
    t0 = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)

    deploy = _make_span(
        span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.DEPLOYMENT,
        name="Deploy",
        status=SpanStatus.RUNNING,
        started_at=t0,
        ended_at=None,
        meta_json=json.dumps({"input_fingerprint": "fp", "deployment_class": "P", "flow_plan": []}),
    )
    operation = _make_span(
        span_id=uuid4(),
        parent_span_id=deploy_span_id,
        deployment_id=root_id,
        root_deployment_id=root_id,
        kind=SpanKind.OPERATION,
        name="SomeOp",
        status=SpanStatus.COMPLETED,
        started_at=t0 + timedelta(seconds=1),
        ended_at=t0 + timedelta(seconds=2),
    )
    for span in (deploy, operation):
        await db.insert_span(span)

    events = await _reconstruct_lifecycle_events(db, root_id)
    for event in events:
        assert "operation" not in event.event_type


@pytest.mark.asyncio
async def test_event_to_payload_strenum_normalized() -> None:
    """Verify that StrEnum values are converted to plain strings."""
    from ai_pipeline_core.deployment._types import RunFailedEvent

    event = RunFailedEvent(
        run_id="r1",
        span_id="s1",
        root_deployment_id="rd1",
        parent_deployment_task_id=None,
        status="failed",
        error_code=ErrorCode.PROVIDER_ERROR,
        error_message="err",
    )
    payload = event_to_payload(event)
    assert isinstance(payload["error_code"], str)
    assert type(payload["error_code"]) is str
    assert payload["error_code"] == "provider_error"


@pytest.mark.asyncio
async def test_document_ref_from_record() -> None:
    from ai_pipeline_core._lifecycle_events import DocumentRef

    record = _make_document(
        document_sha256="sha-abc",
        document_type="MyDoc",
        name="doc.md",
        summary="A doc",
        publicly_visible=True,
        derived_from=("https://a.com",),
        triggered_by=("sha-xyz",),
    )
    ref = DocumentRef.from_record(record)
    assert ref.sha256 == "sha-abc"
    assert ref.class_name == "MyDoc"
    assert ref.name == "doc.md"
    assert ref.summary == "A doc"
    assert ref.publicly_visible is True
    assert ref.derived_from == ("https://a.com",)
    assert ref.triggered_by == ("sha-xyz",)
