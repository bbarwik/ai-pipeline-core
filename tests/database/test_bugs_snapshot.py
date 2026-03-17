"""Regression tests for snapshot export correctness."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ai_pipeline_core.database._types import DocumentRecord, SpanKind, SpanRecord, SpanStatus
from ai_pipeline_core.database.snapshot._download import _build_document_lines, _build_llm_call_lines


def _make_span(**kwargs: object) -> SpanRecord:
    deployment_id = kwargs.pop("deployment_id", uuid4())
    root_deployment_id = kwargs.pop("root_deployment_id", deployment_id)
    started_at: datetime = kwargs.pop("started_at", datetime(2026, 3, 11, 12, 0, tzinfo=UTC))  # type: ignore[assignment]
    defaults: dict[str, object] = {
        "span_id": kwargs.pop("span_id", uuid4()),
        "parent_span_id": kwargs.pop("parent_span_id", None),
        "deployment_id": deployment_id,
        "root_deployment_id": root_deployment_id,
        "run_id": "bug-test-run",
        "deployment_name": "bug-test",
        "kind": SpanKind.TASK,
        "name": "TestSpan",
        "status": SpanStatus.COMPLETED,
        "sequence_no": 0,
        "started_at": started_at,
        "ended_at": started_at + timedelta(seconds=2),
        "version": 1,
        "meta_json": "{}",
        "metrics_json": "{}",
    }
    defaults.update(kwargs)
    return SpanRecord(**defaults)


def _make_document(**kwargs: object) -> DocumentRecord:
    defaults: dict[str, object] = {
        "document_sha256": f"doc-{uuid4().hex[:40]}",
        "content_sha256": f"blob-{uuid4().hex[:40]}",
        "document_type": "TestDocument",
        "name": "test.md",
        "mime_type": "text/markdown",
        "size_bytes": 100,
    }
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


def test_input_documents_not_in_producer_map() -> None:
    """Documents that are inputs to a conversation must NOT appear as produced by that conversation.

    Currently: _build_document_lines iterates output_document_shas without subtracting
    input_document_shas, so input context documents appear as "produced by" the conversation.
    """
    input_doc = _make_document(document_sha256="INPUT_DOC_SHA_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", name="input_context.md")
    output_doc = _make_document(document_sha256="OUTPUT_DOC_SHA_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", name="generated_report.md")

    conversation_span = _make_span(
        kind=SpanKind.CONVERSATION,
        name="ResearchConversation",
        # The conversation receives input_doc as context and produces output_doc
        input_document_shas=(input_doc.document_sha256,),
        output_document_shas=(input_doc.document_sha256, output_doc.document_sha256),
    )

    documents = {
        input_doc.document_sha256: input_doc,
        output_doc.document_sha256: output_doc,
    }

    lines = _build_document_lines([conversation_span], documents)
    text = "\n".join(lines)

    # The output doc SHOULD be attributed to the conversation
    assert "generated_report.md" in text
    # Document name and producer are on adjacent lines — find the producer line for input_context.md
    assert "input_context.md" in text, "Input document should appear in the document listing"
    for i, line in enumerate(lines):
        if "input_context.md" in line:
            assert i + 1 < len(lines), "Expected a detail line after the document name line"
            detail_line = lines[i + 1]
            assert "producer=" in detail_line
            assert "ResearchConversation" not in detail_line, "Input document must not be attributed as produced by the conversation"
            break


def test_input_doc_shows_as_referenced_input() -> None:
    """A document that only appears in input_document_shas (not genuinely produced)
    should have producer='referenced input' in the document listing."""
    shared_doc = _make_document(document_sha256="SHARED_DOC_SHA_CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", name="shared_context.md")

    span = _make_span(
        kind=SpanKind.CONVERSATION,
        name="AnalysisConversation",
        input_document_shas=(shared_doc.document_sha256,),
        output_document_shas=(shared_doc.document_sha256,),  # appears in both — but was only an input
    )

    lines = _build_document_lines([span], {shared_doc.document_sha256: shared_doc})
    text = "\n".join(lines)

    assert "referenced input" in text, "Document that is only an input should show producer='referenced input', not be attributed to the conversation"


def test_llm_call_lines_include_purpose() -> None:
    """LLM call export must include purpose from the parent conversation span's metadata."""
    parent_id = uuid4()
    deployment_id = uuid4()

    conversation_span = _make_span(
        span_id=parent_id,
        deployment_id=deployment_id,
        kind=SpanKind.CONVERSATION,
        name="ResearchConversation",
        meta_json=json.dumps({"purpose": "research_analysis"}),
    )

    llm_round_span = _make_span(
        deployment_id=deployment_id,
        parent_span_id=parent_id,
        kind=SpanKind.LLM_ROUND,
        name="llm_round",
        meta_json=json.dumps({
            "model": "gpt-5",
            "round_index": 0,
            "response_content": "analysis result",
            "finish_reason": "stop",
        }),
        metrics_json=json.dumps({
            "tokens_input": 1000,
            "tokens_output": 200,
        }),
    )

    lines = _build_llm_call_lines([conversation_span, llm_round_span])
    assert lines, "Should produce at least one JSONL line for the LLM round"
    record = json.loads(lines[0])
    assert "purpose" in record, "LLM call record must include 'purpose' field"
    assert record["purpose"] == "research_analysis"


def test_llm_call_lines_include_latency() -> None:
    """LLM call export must include time_taken_ms computed from span timestamps."""
    start = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)
    end = start + timedelta(milliseconds=1500)

    llm_round = _make_span(
        kind=SpanKind.LLM_ROUND,
        name="llm_round",
        started_at=start,
        ended_at=end,
        meta_json=json.dumps({"model": "gpt-5", "round_index": 0}),
        metrics_json=json.dumps({
            "tokens_input": 500,
            "tokens_output": 100,
            "first_token_ms": 350,
        }),
    )

    lines = _build_llm_call_lines([llm_round])
    assert lines
    record = json.loads(lines[0])
    assert "time_taken_ms" in record, "LLM call record must include 'time_taken_ms'"
    assert record["time_taken_ms"] == pytest.approx(1500, abs=10)
    assert "first_token_ms" in record, "LLM call record must include 'first_token_ms'"
    assert record["first_token_ms"] == 350


def test_validate_bundle_requires_schema_meta(tmp_path: Path) -> None:
    """A snapshot bundle must contain schema_meta.json for provenance tracking."""
    from ai_pipeline_core.database.filesystem._validation import validate_bundle

    # Create a minimal valid bundle structure without schema_meta
    (tmp_path / "runs").mkdir()
    (tmp_path / "documents").mkdir()
    (tmp_path / "blobs").mkdir()
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    # validation should fail because schema_meta.json is missing
    with pytest.raises((FileNotFoundError, ValueError), match="schema_meta"):
        validate_bundle(tmp_path)


def test_conversation_chain_section_in_summary() -> None:
    """Summary should contain a dedicated 'Conversation Chains' section that shows
    the warmup → fork topology, not just inline tree line suffixes."""
    from ai_pipeline_core.database.snapshot._spans import generate_summary_from_tree

    deployment_id = uuid4()
    warmup_id = uuid4()
    base = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)

    deployment_span = _make_span(
        span_id=uuid4(),
        deployment_id=deployment_id,
        kind=SpanKind.DEPLOYMENT,
        name="ChainTest",
        started_at=base,
    )

    warmup = _make_span(
        span_id=warmup_id,
        deployment_id=deployment_id,
        parent_span_id=deployment_span.span_id,
        kind=SpanKind.CONVERSATION,
        name="warmup",
        started_at=base + timedelta(seconds=1),
        previous_conversation_id=None,
        meta_json=json.dumps({"model": "gpt-5"}),
    )

    fork1 = _make_span(
        deployment_id=deployment_id,
        parent_span_id=deployment_span.span_id,
        kind=SpanKind.CONVERSATION,
        name="fork_analysis",
        started_at=base + timedelta(seconds=2),
        previous_conversation_id=warmup_id,
        meta_json=json.dumps({"model": "gpt-5"}),
    )

    fork2 = _make_span(
        deployment_id=deployment_id,
        parent_span_id=deployment_span.span_id,
        kind=SpanKind.CONVERSATION,
        name="fork_synthesis",
        started_at=base + timedelta(seconds=2),
        previous_conversation_id=warmup_id,
        meta_json=json.dumps({"model": "gpt-5"}),
    )

    tree = [deployment_span, warmup, fork1, fork2]
    summary = generate_summary_from_tree(tree, deployment_id)

    # The summary already shows inline "(continues ...)" suffixes in the tree.
    # What's missing is a DEDICATED section showing the chain topology.
    assert "## Conversation Chains" in summary or "## Chains" in summary, (
        "Summary must contain a dedicated conversation chain section showing warmup → [fork_analysis, fork_synthesis] topology, not just inline suffixes"
    )
