"""Unit tests for SpanData — OTel extraction, ClickHouse roundtrip, classification."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from opentelemetry.trace import StatusCode

from ai_pipeline_core.observability._span_data import (
    ATTR_INPUT_DOC_SHA256S,
    ATTR_OUTPUT_DOC_SHA256S,
    RES_EXECUTION_ID,
    RES_FLOW_NAME,
    RES_RUN_ID,
    RES_RUN_SCOPE,
    SpanData,
    _CONTENT_ATTRS,
    _classify_span_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_UUID = UUID("12345678-1234-5678-1234-567812345678")


def _make_otel_span(
    *,
    span_id: int = 1,
    trace_id: int = 2,
    name: str = "test_span",
    attrs: dict[str, object] | None = None,
    parent_span_id: int | None = None,
    start_time: int = 1_000_000_000,
    end_time: int = 2_000_000_000,
    status_code: StatusCode = StatusCode.OK,
    status_description: str | None = None,
    events: list[object] | None = None,
    resource_attrs: dict[str, object] | None = None,
) -> MagicMock:
    ctx = SimpleNamespace(span_id=span_id, trace_id=trace_id)
    parent = SimpleNamespace(span_id=parent_span_id) if parent_span_id else None

    span = MagicMock()
    span.get_span_context.return_value = ctx
    span.name = name
    span.attributes = attrs or {}
    span.parent = parent
    span.start_time = start_time
    span.end_time = end_time
    span.status = SimpleNamespace(status_code=status_code, description=status_description)
    span.events = events or []

    if resource_attrs is not None:
        span.resource = SimpleNamespace(attributes=resource_attrs)
    else:
        span.resource = None
    return span


# ---------------------------------------------------------------------------
# from_otel_span
# ---------------------------------------------------------------------------


class TestFromOtelSpanExtractsAllFields:
    def test_identity_and_timing(self):
        span = _make_otel_span(
            span_id=0xABCD,
            trace_id=0xFF,
            name="my_span",
            parent_span_id=0x42,
            start_time=1_000_000_000,
            end_time=2_000_000_000,
            attrs={
                RES_EXECUTION_ID: str(_TEST_UUID),
                RES_RUN_ID: "run-1",
                RES_FLOW_NAME: "flow-1",
                RES_RUN_SCOPE: "scope/run-1",
                "gen_ai.usage.cost": 0.05,
                "gen_ai.usage.input_tokens": 1000,
                "gen_ai.usage.output_tokens": 200,
                "gen_ai.usage.cache_read_input_tokens": 500,
                "gen_ai.request.model": "gpt-5",
            },
        )
        data = SpanData.from_otel_span(span, span_order=7)

        assert data.execution_id == _TEST_UUID
        assert data.span_id == format(0xABCD, "016x")
        assert data.trace_id == format(0xFF, "032x")
        assert data.parent_span_id == format(0x42, "016x")
        assert data.name == "my_span"
        assert data.run_id == "run-1"
        assert data.flow_name == "flow-1"
        assert data.run_scope == "scope/run-1"
        assert data.span_order == 7
        assert data.duration_ms == 1000
        assert data.cost == 0.05
        assert data.tokens_input == 1000
        assert data.tokens_output == 200
        assert data.tokens_cached == 500
        assert data.llm_model == "gpt-5"
        assert data.status == "completed"


class TestFromOtelSpanNoExecutionId:
    def test_missing_execution_id_returns_none(self):
        span = _make_otel_span(attrs={})
        data = SpanData.from_otel_span(span, span_order=1)
        assert data.execution_id is None


class TestFromOtelSpanExecutionIdFromAttrs:
    def test_execution_id_from_span_attrs(self):
        uid = uuid4()
        span = _make_otel_span(attrs={RES_EXECUTION_ID: str(uid)})
        data = SpanData.from_otel_span(span, span_order=1)
        assert data.execution_id == uid

    def test_execution_id_from_resource_attrs(self):
        uid = uuid4()
        span = _make_otel_span(resource_attrs={RES_EXECUTION_ID: str(uid)})
        data = SpanData.from_otel_span(span, span_order=1)
        assert data.execution_id == uid


class TestFromOtelSpanContentAttrsExcluded:
    def test_pipeline_keys_not_in_attributes(self):
        span = _make_otel_span(
            attrs={
                RES_EXECUTION_ID: str(_TEST_UUID),
                RES_RUN_ID: "r",
                RES_FLOW_NAME: "f",
                RES_RUN_SCOPE: "s",
                "lmnr.span.input": '{"x": 1}',
                "lmnr.span.output": '{"y": 2}',
                "replay.payload": '{"z": 3}',
                ATTR_INPUT_DOC_SHA256S: ["sha1"],
                ATTR_OUTPUT_DOC_SHA256S: ["sha2"],
                "custom.attr": "keep_me",
            },
        )
        data = SpanData.from_otel_span(span, span_order=1)
        for key in _CONTENT_ATTRS:
            assert key not in data.attributes
        assert data.attributes["custom.attr"] == "keep_me"


class TestFromOtelSpanErrorStatus:
    def test_error_maps_to_failed(self):
        span = _make_otel_span(
            status_code=StatusCode.ERROR,
            status_description="something broke",
        )
        data = SpanData.from_otel_span(span, span_order=1)
        assert data.status == "failed"
        assert data.error_message == "something broke"
        assert data.attributes.get("status_description") == "something broke"


class TestFromOtelSpanEventsExtracted:
    def test_events_tuple_of_dicts(self):
        event = MagicMock()
        event.name = "test_event"
        event.timestamp = 1_500_000_000
        event.attributes = {"key": "val"}
        span = _make_otel_span(events=[event])
        data = SpanData.from_otel_span(span, span_order=1)
        assert len(data.events) == 1
        assert data.events[0]["name"] == "test_event"
        assert data.events[0]["timestamp"] == 1_500_000_000
        assert data.events[0]["attributes"]["key"] == "val"


class TestFromOtelSpanDocLineage:
    def test_input_output_sha256_tuples(self):
        span = _make_otel_span(
            attrs={
                ATTR_INPUT_DOC_SHA256S: ["sha_in_1", "sha_in_2"],
                ATTR_OUTPUT_DOC_SHA256S: ["sha_out_1"],
            },
        )
        data = SpanData.from_otel_span(span, span_order=1)
        assert data.input_doc_sha256s == ("sha_in_1", "sha_in_2")
        assert data.output_doc_sha256s == ("sha_out_1",)


# ---------------------------------------------------------------------------
# from_clickhouse_row
# ---------------------------------------------------------------------------


class TestFromClickhouseRowRoundtrip:
    def test_all_fields(self):
        uid = uuid4()
        now = datetime.now(UTC)
        row = {
            "execution_id": uid,
            "span_id": "abc123",
            "trace_id": "def456",
            "parent_span_id": "parent1",
            "name": "test",
            "run_id": "run-1",
            "flow_name": "flow-1",
            "run_scope": "scope/run-1",
            "span_type": "llm",
            "status": "completed",
            "start_time": now,
            "end_time": now,
            "duration_ms": 500,
            "span_order": 3,
            "cost": 0.01,
            "tokens_input": 100,
            "tokens_output": 50,
            "tokens_cached": 80,
            "llm_model": "gpt-5",
            "error_message": "",
            "input_json": '{"x": 1}',
            "output_json": '{"y": 2}',
            "replay_payload": '{"z": 3}',
            "attributes_json": '{"custom": "val"}',
            "events_json": '[{"name": "evt"}]',
            "input_doc_sha256s": ("sha1",),
            "output_doc_sha256s": ("sha2",),
        }
        data = SpanData.from_clickhouse_row(row)
        assert data.execution_id == uid
        assert data.span_id == "abc123"
        assert data.span_type == "llm"
        assert data.cost == 0.01
        assert data.attributes == {"custom": "val"}
        assert data.events == ({"name": "evt"},)
        assert data.input_doc_sha256s == ("sha1",)


class TestFromClickhouseRowTimezoneAwareness:
    """ClickHouse returns naive datetimes but SpanData must always be timezone-aware.

    clickhouse_connect strips tzinfo from DateTime64('UTC') columns, returning
    naive Python datetimes. The live OTel path creates aware datetimes via
    datetime.fromtimestamp(ns, tz=UTC). Mixing aware and naive datetimes in
    summary generation causes: TypeError: can't subtract offset-naive and offset-aware datetimes
    """

    def test_naive_datetimes_become_utc_aware(self):
        """Naive datetimes from ClickHouse must be converted to UTC-aware."""
        naive_start = datetime(2025, 1, 15, 10, 30, 0)
        naive_end = datetime(2025, 1, 15, 10, 30, 5)
        assert naive_start.tzinfo is None, "Test precondition: datetime must be naive"

        row = {
            "span_id": "s1",
            "trace_id": "t1",
            "name": "span",
            "start_time": naive_start,
            "end_time": naive_end,
        }
        data = SpanData.from_clickhouse_row(row)

        assert data.start_time.tzinfo is not None, "start_time must be timezone-aware"
        assert data.end_time.tzinfo is not None, "end_time must be timezone-aware"
        assert data.start_time.tzinfo == UTC
        assert data.end_time.tzinfo == UTC


class TestFromClickhouseRowDefaults:
    def test_minimal_row(self):
        now = datetime.now(UTC)
        row = {
            "span_id": "s1",
            "trace_id": "t1",
            "name": "span",
            "start_time": now,
            "end_time": now,
        }
        data = SpanData.from_clickhouse_row(row)
        assert data.execution_id is None
        assert data.parent_span_id is None
        assert data.run_id == ""
        assert data.flow_name == ""
        assert data.span_type == "trace"
        assert data.status == "completed"
        assert data.cost == 0.0
        assert data.tokens_input == 0
        assert data.llm_model is None
        assert data.attributes == {}
        assert data.events == ()
        assert data.input_doc_sha256s == ()


# ---------------------------------------------------------------------------
# _classify_span_type
# ---------------------------------------------------------------------------


class TestClassifySpanType:
    def test_llm_type(self):
        assert _classify_span_type({"lmnr.span.type": "LLM"}) == "llm"

    def test_flow_type(self):
        assert _classify_span_type({"prefect.flow.name": "my_flow"}) == "flow"

    def test_task_type(self):
        assert _classify_span_type({"prefect.task.name": "my_task"}) == "task"

    def test_default_trace(self):
        assert _classify_span_type({}) == "trace"

    def test_llm_takes_precedence(self):
        assert _classify_span_type({"lmnr.span.type": "LLM", "prefect.flow.name": "f"}) == "llm"


# ---------------------------------------------------------------------------
# _CONTENT_ATTRS completeness
# ---------------------------------------------------------------------------


class TestContentAttrsCompleteness:
    def test_contains_all_expected_keys(self):
        expected = {
            "lmnr.span.input",
            "lmnr.span.output",
            "replay.payload",
            RES_EXECUTION_ID,
            RES_RUN_ID,
            RES_FLOW_NAME,
            RES_RUN_SCOPE,
            ATTR_INPUT_DOC_SHA256S,
            ATTR_OUTPUT_DOC_SHA256S,
        }
        assert _CONTENT_ATTRS == expected
        assert len(_CONTENT_ATTRS) == 9
