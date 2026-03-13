"""Tests for span-era summary and cost report generation."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ai_pipeline_core.database import BlobRecord, DocumentRecord, SpanKind, SpanRecord, SpanStatus
from ai_pipeline_core.database._memory import MemoryDatabase
from ai_pipeline_core.database.snapshot._download import download_deployment
from ai_pipeline_core.database.snapshot._spans import build_span_tree_view, format_span_tree_lines
from ai_pipeline_core.database.snapshot._summary import generate_costs, generate_summary


def _make_span(**kwargs: object) -> SpanRecord:
    deployment_id = kwargs.pop("deployment_id", uuid4())
    root_deployment_id = kwargs.pop("root_deployment_id", deployment_id)
    started_at = kwargs.pop("started_at", datetime(2026, 3, 11, 12, 0, tzinfo=UTC))
    if not isinstance(started_at, datetime):
        msg = "started_at must be a datetime"
        raise TypeError(msg)
    defaults: dict[str, object] = {
        "span_id": kwargs.pop("span_id", uuid4()),
        "parent_span_id": None,
        "deployment_id": deployment_id,
        "root_deployment_id": root_deployment_id,
        "run_id": "span-run",
        "deployment_name": "span-pipeline",
        "kind": SpanKind.TASK,
        "name": "ExampleTask",
        "description": "",
        "status": SpanStatus.COMPLETED,
        "sequence_no": 0,
        "started_at": started_at,
        "ended_at": started_at + timedelta(seconds=1),
        "version": 1,
        "cache_key": "",
        "previous_conversation_id": None,
        "cost_usd": 0.0,
        "error_type": "",
        "error_message": "",
        "input_document_shas": (),
        "output_document_shas": (),
        "target": "",
        "receiver_json": "",
        "input_json": "",
        "output_json": "",
        "error_json": "",
        "meta_json": "",
        "metrics_json": "",
        "input_blob_shas": (),
        "output_blob_shas": (),
    }
    defaults.update(kwargs)
    return SpanRecord(**defaults)


async def _seed_summary_database() -> tuple[MemoryDatabase, UUID]:
    database = MemoryDatabase()
    deployment_id = uuid4()
    base = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)

    deployment = _make_span(
        span_id=uuid4(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.DEPLOYMENT,
        name="RootDeployment",
        deployment_name="span-pipeline",
        run_id="span-run",
        started_at=base,
        ended_at=base + timedelta(minutes=1),
        meta_json=json.dumps({"flow_plan": [{"name": "GatherFlow"}]}),
    )
    flow = _make_span(
        span_id=uuid4(),
        parent_span_id=deployment.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.FLOW,
        name="GatherFlow",
        sequence_no=1,
        started_at=base + timedelta(seconds=1),
        ended_at=base + timedelta(seconds=51),
        meta_json=json.dumps({"step": 1, "total_steps": 1, "cache_hit": True}),
    )
    task = _make_span(
        span_id=uuid4(),
        parent_span_id=flow.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.TASK,
        name="GatherTask",
        sequence_no=1,
        started_at=base + timedelta(seconds=2),
        ended_at=base + timedelta(seconds=42),
        meta_json=json.dumps({"attempt": 1}),
    )
    operation = _make_span(
        span_id=uuid4(),
        parent_span_id=task.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.OPERATION,
        name="collect_sources",
        sequence_no=1,
        started_at=base + timedelta(seconds=3),
        ended_at=base + timedelta(seconds=33),
    )
    conversation = _make_span(
        span_id=uuid4(),
        parent_span_id=operation.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.CONVERSATION,
        name="ConversationBoundary",
        description="analysis boundary",
        sequence_no=1,
        started_at=base + timedelta(seconds=4),
        ended_at=base + timedelta(seconds=8),
        cost_usd=999.0,
        meta_json=json.dumps({
            "model": "gpt-5.1",
            "purpose": "analyze_document",
            "cache_hit": True,
        }),
        metrics_json=json.dumps({
            "tokens_input": 5000,
            "tokens_output": 250,
            "tokens_cache_read": 2000,
            "tokens_reasoning": 50,
        }),
    )
    first_round = _make_span(
        span_id=uuid4(),
        parent_span_id=conversation.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        sequence_no=1,
        started_at=base + timedelta(seconds=4),
        ended_at=base + timedelta(seconds=5, milliseconds=500),
        cost_usd=0.2,
        meta_json=json.dumps({
            "model": "gpt-5.1",
            "round_index": 1,
        }),
        metrics_json=json.dumps({
            "tokens_input": 3000,
            "tokens_output": 150,
            "tokens_cache_read": 1000,
            "tokens_reasoning": 20,
        }),
    )
    tool_call = _make_span(
        span_id=uuid4(),
        parent_span_id=conversation.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.TOOL_CALL,
        name="tool-call",
        sequence_no=2,
        started_at=base + timedelta(seconds=5, milliseconds=500),
        ended_at=base + timedelta(seconds=6, milliseconds=100),
        meta_json=json.dumps({"round_index": 1, "tool_name": "web_search"}),
    )
    second_round = _make_span(
        span_id=uuid4(),
        parent_span_id=conversation.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.LLM_ROUND,
        name="round-2",
        sequence_no=3,
        started_at=base + timedelta(seconds=6, milliseconds=100),
        ended_at=base + timedelta(seconds=7, milliseconds=300),
        cost_usd=0.3,
        meta_json=json.dumps({
            "model": "gemini-3-flash",
            "round_index": 2,
        }),
        metrics_json=json.dumps({
            "tokens_input": 4000,
            "tokens_output": 200,
            "tokens_cache_read": 2500,
            "tokens_reasoning": 40,
        }),
    )
    failed_task = _make_span(
        span_id=uuid4(),
        parent_span_id=flow.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.TASK,
        name="FailedTask",
        sequence_no=2,
        status=SpanStatus.FAILED,
        started_at=base + timedelta(seconds=9),
        ended_at=base + timedelta(seconds=10),
        error_type="RuntimeError",
        error_message="boom",
        output_document_shas=("summary-doc",),
    )

    for span in (deployment, flow, task, operation, conversation, first_round, tool_call, second_round, failed_task):
        await database.insert_span(span)

    await database.save_document(
        DocumentRecord(
            document_sha256="summary-doc",
            content_sha256="summary-blob",
            document_type="SummaryDocument",
            name="summary.md",
            description="Summary document",
            mime_type="text/markdown",
            size_bytes=12,
            summary="Short summary",
        )
    )
    await database.save_blob(BlobRecord(content_sha256="summary-blob", content=b"# Summary\nHi"))

    return database, deployment_id


@pytest.mark.asyncio
async def test_generate_summary_renders_all_span_kinds_and_cache_hits() -> None:
    database, deployment_id = await _seed_summary_database()

    summary = await generate_summary(database, deployment_id)

    assert "# span-pipeline / span-run" in summary
    assert "**Total Cost**: $0.5000" in summary
    assert "999.0000" not in summary
    assert "flow[1/1]: GatherFlow completed 50s cache-hit" in summary
    assert "task: GatherTask completed 40s" in summary
    assert "operation: collect_sources completed 30s" in summary
    assert "conversation: analyze_document 4.0s gpt-5.1 5K in / 2K cache / 250 out / 50 reasoning $0.5000 cache-hit" in summary
    assert "llm_round[1]: gpt-5.1 1.5s 3K in / 1K cache / 150 out / 20 reasoning $0.2000" in summary
    assert "tool_call[1]: web_search completed 600ms" in summary
    assert "llm_round[2]: gemini-3-flash 1.2s 4K in / 2K cache / 200 out / 40 reasoning $0.3000" in summary


@pytest.mark.asyncio
async def test_generate_costs_groups_llm_round_costs_by_model() -> None:
    database, deployment_id = await _seed_summary_database()

    costs = await generate_costs(database, deployment_id)

    assert "# Cost by Model" in costs
    assert "| gemini-3-flash | 1 | 4,000 | 2,500 | 200 | 40 | $0.3000 |" in costs
    assert "| gpt-5.1 | 1 | 3,000 | 1,000 | 150 | 20 | $0.2000 |" in costs
    assert "**Total**: $0.5000 across 2 llm_round spans" in costs


@pytest.mark.asyncio
async def test_download_deployment_writes_summary_and_cost_artifacts(tmp_path: Path) -> None:
    database, deployment_id = await _seed_summary_database()
    output_dir = tmp_path / "download"

    await download_deployment(database, deployment_id, output_dir)

    assert (output_dir / "summary.md").exists()
    assert (output_dir / "costs.md").exists()
    assert "span-pipeline / span-run" in (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "# Cost by Model" in (output_dir / "costs.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_download_deployment_writes_llm_calls_errors_and_documents_artifacts(tmp_path: Path) -> None:
    database, deployment_id = await _seed_summary_database()
    output_dir = tmp_path / "download"

    await download_deployment(database, deployment_id, output_dir)

    assert (output_dir / "llm_calls.jsonl").exists()
    assert (output_dir / "documents.md").exists()
    assert (output_dir / "errors.md").exists()
    assert "gpt-5.1" in (output_dir / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "summary.md" in (output_dir / "documents.md").read_text(encoding="utf-8")
    assert "FailedTask" in (output_dir / "errors.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_generate_summary_returns_no_data_for_empty_tree() -> None:
    summary = await generate_summary(MemoryDatabase(), uuid4())

    assert summary == "# No execution data found\n"


@pytest.mark.asyncio
async def test_generate_summary_raises_actionable_error_for_invalid_meta_json() -> None:
    database = MemoryDatabase()
    deployment_id = uuid4()
    base = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)
    await database.insert_span(
        _make_span(
            deployment_id=deployment_id,
            root_deployment_id=deployment_id,
            kind=SpanKind.DEPLOYMENT,
            name="BrokenDeployment",
            deployment_name="broken-deployment",
            started_at=base,
            ended_at=base + timedelta(seconds=5),
            meta_json="{not-json}",
        )
    )

    with pytest.raises(ValueError, match="invalid meta_json"):
        await generate_summary(database, deployment_id)


@pytest.mark.asyncio
async def test_generate_summary_flow_plan_includes_unplanned_executed_flows() -> None:
    database = MemoryDatabase()
    deployment_id = uuid4()
    base = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)

    deployment = _make_span(
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.DEPLOYMENT,
        name="RootDeployment",
        deployment_name="span-pipeline",
        started_at=base,
        ended_at=base + timedelta(minutes=1),
        meta_json=json.dumps({"flow_plan": [{"name": "PlannedFlow"}]}),
    )
    planned_flow = _make_span(
        parent_span_id=deployment.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.FLOW,
        name="PlannedFlow",
        sequence_no=1,
        started_at=base + timedelta(seconds=1),
        ended_at=base + timedelta(seconds=11),
    )
    surprise_flow = _make_span(
        parent_span_id=deployment.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.FLOW,
        name="SurpriseFlow",
        sequence_no=2,
        started_at=base + timedelta(seconds=12),
        ended_at=base + timedelta(seconds=20),
    )
    for span in (deployment, planned_flow, surprise_flow):
        await database.insert_span(span)

    summary = await generate_summary(database, deployment_id)

    assert "| 1 | PlannedFlow | completed | 10s | $0.0000 |" in summary
    assert "| 2 | SurpriseFlow | completed | 8.0s | $0.0000 |" in summary


def test_format_span_tree_lines_reports_cycle_once() -> None:
    deployment_id = uuid4()
    base = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)
    deployment = _make_span(
        span_id=uuid4(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.DEPLOYMENT,
        name="RootDeployment",
        deployment_name="span-pipeline",
        started_at=base,
        ended_at=base + timedelta(seconds=10),
    )
    first = _make_span(
        span_id=uuid4(),
        parent_span_id=deployment.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.TASK,
        name="FirstTask",
        sequence_no=1,
        started_at=base + timedelta(seconds=1),
        ended_at=base + timedelta(seconds=4),
    )
    second = _make_span(
        span_id=uuid4(),
        parent_span_id=first.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.TASK,
        name="SecondTask",
        sequence_no=2,
        started_at=base + timedelta(seconds=4),
        ended_at=base + timedelta(seconds=7),
    )

    view = build_span_tree_view([deployment, first, second], deployment_id)
    assert view is not None
    view.children_map[second.span_id] = [first.span_id]

    lines = format_span_tree_lines(view)

    assert any("cycle detected" in line for line in lines)


def test_format_span_tree_lines_include_snapshot_filenames() -> None:
    deployment_id = uuid4()
    base = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)
    deployment = _make_span(
        span_id=uuid4(),
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.DEPLOYMENT,
        name="RootDeployment",
        deployment_name="span-pipeline",
        started_at=base,
        ended_at=base + timedelta(seconds=10),
    )
    flow = _make_span(
        span_id=uuid4(),
        parent_span_id=deployment.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.FLOW,
        name="GatherFlow",
        sequence_no=1,
        started_at=base + timedelta(seconds=1),
        ended_at=base + timedelta(seconds=4),
    )
    task = _make_span(
        span_id=uuid4(),
        parent_span_id=flow.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.TASK,
        name="GatherTask",
        sequence_no=1,
        started_at=base + timedelta(seconds=2),
        ended_at=base + timedelta(seconds=3),
    )
    conversation = _make_span(
        span_id=uuid4(),
        parent_span_id=task.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.CONVERSATION,
        name="ConversationBoundary",
        sequence_no=1,
        started_at=base + timedelta(seconds=3),
        ended_at=base + timedelta(seconds=4),
        meta_json=json.dumps({"model": "gpt-5.1", "purpose": "analyze_document"}),
    )
    llm_round = _make_span(
        span_id=uuid4(),
        parent_span_id=conversation.span_id,
        deployment_id=deployment_id,
        root_deployment_id=deployment_id,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        sequence_no=1,
        started_at=base + timedelta(seconds=4),
        ended_at=base + timedelta(seconds=5),
        meta_json=json.dumps({"model": "gpt-5.1", "round_index": 1}),
    )

    view = build_span_tree_view([deployment, flow, task, conversation, llm_round], deployment_id)
    assert view is not None

    lines = format_span_tree_lines(view, include_filenames=True)

    assert any("flow-" in line and ".json" in line for line in lines)
    assert any("task-" in line and ".json" in line for line in lines)
    assert any("conv-" in line and ".json" in line for line in lines)
    assert not any("llm_round" in line and ".json" in line for line in lines)
