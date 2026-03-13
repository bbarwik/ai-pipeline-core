"""Tests for LaminarSpanSink."""

from pathlib import Path
from types import MappingProxyType
from uuid import uuid7

import pytest

from ai_pipeline_core.database import SpanKind
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.observability._laminar_sink import LaminarSpanSink, _reset_for_testing
from ai_pipeline_core.pipeline._execution_context import ExecutionContext
from ai_pipeline_core.pipeline._runtime_sinks import build_runtime_sinks
from ai_pipeline_core.pipeline._span_sink import SpanMetrics
from ai_pipeline_core.pipeline.limits import _SharedStatus
from ai_pipeline_core.settings import Settings


class _FakeSpan:
    def __init__(self, record: dict[str, object]) -> None:
        self._record = record

    def get_laminar_span_context(self) -> str:
        if _FakeLaminar.fail_get_context:
            raise RuntimeError("get context failed")
        return self._record["context"]  # type: ignore[return-value]

    def set_attributes(self, attributes: dict[str, object]) -> None:
        if _FakeLaminar.fail_set_attributes:
            raise RuntimeError("set_attributes failed")
        self._record["attribute_updates"] = attributes

    def set_output(self, output: object) -> None:
        if _FakeLaminar.fail_set_output:
            raise RuntimeError("set_output failed")
        self._record["output"] = output

    def end(self) -> None:
        if _FakeLaminar.fail_end:
            raise RuntimeError("end failed")
        self._record["ended"] = True


class _FakeLaminar:
    initialize_calls: list[dict[str, object]] = []
    start_calls: list[dict[str, object]] = []
    fail_initialize = False
    fail_start = False
    fail_get_context = False
    fail_set_attributes = False
    fail_set_output = False
    fail_end = False

    @classmethod
    def initialize(cls, *, project_api_key: str, disabled_instruments: set[object]) -> None:
        if cls.fail_initialize:
            raise RuntimeError("initialize failed")
        cls.initialize_calls.append({
            "project_api_key": project_api_key,
            "disabled_instruments": disabled_instruments,
        })

    @classmethod
    def start_span(
        cls,
        *,
        name: str,
        span_type: str,
        parent_span_context: object,
        input: object,
        attributes: dict[str, object],
    ) -> _FakeSpan:
        if cls.fail_start:
            raise RuntimeError("start failed")
        record: dict[str, object] = {
            "name": name,
            "span_type": span_type,
            "parent_span_context": parent_span_context,
            "input": input,
            "attributes": attributes,
            "context": f"context-{len(cls.start_calls)}",
            "ended": False,
        }
        cls.start_calls.append(record)
        return _FakeSpan(record)


class _FakeInstruments:
    OPENAI = "openai"
    LITELLM = "litellm"


def _install_fake_lmnr(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_pipeline_core.observability import _laminar_sink

    monkeypatch.setattr(_laminar_sink, "Laminar", _FakeLaminar)
    monkeypatch.setattr(_laminar_sink, "Instruments", _FakeInstruments)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    _reset_for_testing()
    _FakeLaminar.initialize_calls.clear()
    _FakeLaminar.start_calls.clear()
    _FakeLaminar.fail_initialize = False
    _FakeLaminar.fail_start = False
    _FakeLaminar.fail_get_context = False
    _FakeLaminar.fail_set_attributes = False
    _FakeLaminar.fail_set_output = False
    _FakeLaminar.fail_end = False


def _build_context() -> ExecutionContext:
    return ExecutionContext(
        run_id="test-run",
        execution_id=None,
        publisher=_NoopPublisher(),
        limits=MappingProxyType({}),
        limits_status=_SharedStatus(),
    )


def test_build_runtime_sinks_includes_laminar_sink_when_key_is_set() -> None:
    sinks = build_runtime_sinks(database=None, settings_obj=Settings(lmnr_project_api_key="secret"))

    assert len(sinks) == 1
    assert isinstance(sinks[0], LaminarSpanSink)


@pytest.mark.asyncio
async def test_laminar_sink_maps_llm_round_to_gen_ai_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lmnr(monkeypatch)
    sink = LaminarSpanSink(Settings(lmnr_project_api_key="secret"))
    parent_span_id = uuid7()
    child_span_id = uuid7()

    await sink.on_span_started(
        span_id=parent_span_id,
        parent_span_id=None,
        kind=SpanKind.CONVERSATION,
        name="conversation",
        target="decoded_method:ai_pipeline_core.llm.conversation:Conversation.send",
        started_at=None,
        input_preview={"conversation": True},
    )
    await sink.on_span_started(
        span_id=child_span_id,
        parent_span_id=parent_span_id,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        target="",
        started_at=None,
        input_preview=[{"role": "user", "content": "hello"}],
    )
    await sink.on_span_finished(
        span_id=child_span_id,
        ended_at=None,
        output_preview="world",
        error=None,
        metrics=SpanMetrics(
            time_taken_ms=25,
            log_summary={"total": 0, "warnings": 0, "errors": 0, "last_error": ""},
            tokens_input=11,
            tokens_output=7,
            tokens_cache_read=3,
            tokens_reasoning=2,
            cost_usd=0.5,
            first_token_ms=4,
        ),
        meta={"model": "gpt-5.1", "response_id": "resp-123"},
    )

    assert _FakeLaminar.initialize_calls == [
        {
            "project_api_key": "secret",
            "disabled_instruments": {"openai", "litellm"},
        }
    ]
    assert len(_FakeLaminar.start_calls) == 2
    assert _FakeLaminar.start_calls[0]["span_type"] == "DEFAULT"
    assert _FakeLaminar.start_calls[1]["span_type"] == "LLM"
    assert _FakeLaminar.start_calls[1]["parent_span_context"] == "context-0"
    assert _FakeLaminar.start_calls[1]["attributes"] == {
        "ai_pipeline.span_kind": SpanKind.LLM_ROUND.value,
        "ai_pipeline.target": "",
    }
    assert _FakeLaminar.start_calls[1]["attribute_updates"] == {
        "gen_ai.system": "litellm",
        "gen_ai.usage.input_tokens": 11,
        "gen_ai.usage.output_tokens": 7,
        "gen_ai.usage.cache_read_input_tokens": 3,
        "gen_ai.usage.reasoning_tokens": 2,
        "gen_ai.usage.cost": 0.5,
        "gen_ai.input.messages": '[{"role": "user", "content": "hello"}]',
        "gen_ai.output.messages": '"world"',
        "gen_ai.request.model": "gpt-5.1",
        "gen_ai.response.model": "gpt-5.1",
        "gen_ai.response.id": "resp-123",
    }
    assert _FakeLaminar.start_calls[1]["output"] == "world"
    assert _FakeLaminar.start_calls[1]["ended"] is True
    assert child_span_id not in sink._open_spans


@pytest.mark.asyncio
async def test_laminar_sink_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lmnr(monkeypatch)
    sink = LaminarSpanSink(Settings(lmnr_project_api_key="secret"))
    span_id = uuid7()

    _FakeLaminar.fail_start = True
    await sink.on_span_started(
        span_id=span_id,
        parent_span_id=None,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        target="",
        started_at=None,
        input_preview="input",
    )
    assert sink._open_spans == {}

    _FakeLaminar.fail_start = False
    await sink.on_span_started(
        span_id=span_id,
        parent_span_id=None,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        target="",
        started_at=None,
        input_preview="input",
    )
    _FakeLaminar.fail_set_attributes = True
    _FakeLaminar.fail_set_output = True
    _FakeLaminar.fail_end = True
    await sink.on_span_finished(
        span_id=span_id,
        ended_at=None,
        output_preview="output",
        error=RuntimeError("boom"),
        metrics=SpanMetrics(
            time_taken_ms=10,
            log_summary={"total": 0, "warnings": 0, "errors": 0, "last_error": ""},
        ),
        meta={"model": "gpt-5.1", "response_id": "resp-123"},
    )


@pytest.mark.asyncio
async def test_laminar_sink_ends_span_when_context_capture_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_lmnr(monkeypatch)
    sink = LaminarSpanSink(Settings(lmnr_project_api_key="secret"))
    span_id = uuid7()

    _FakeLaminar.fail_get_context = True
    await sink.on_span_started(
        span_id=span_id,
        parent_span_id=None,
        kind=SpanKind.LLM_ROUND,
        name="round-1",
        target="",
        started_at=None,
        input_preview="input",
    )

    assert sink._open_spans == {}
    assert _FakeLaminar.start_calls[0]["ended"] is True


@pytest.mark.asyncio
async def test_laminar_sink_skips_export_when_process_was_initialized_with_different_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lmnr(monkeypatch)
    first_sink = LaminarSpanSink(Settings(lmnr_project_api_key="project-a"))
    second_sink = LaminarSpanSink(Settings(lmnr_project_api_key="project-b"))

    await first_sink.on_span_started(
        span_id=uuid7(),
        parent_span_id=None,
        kind=SpanKind.CONVERSATION,
        name="conversation",
        target="decoded_method:ai_pipeline_core.llm.conversation:Conversation.send",
        started_at=None,
        input_preview={"conversation": True},
    )
    await second_sink.on_span_started(
        span_id=uuid7(),
        parent_span_id=None,
        kind=SpanKind.CONVERSATION,
        name="conversation",
        target="decoded_method:ai_pipeline_core.llm.conversation:Conversation.send",
        started_at=None,
        input_preview={"conversation": True},
    )

    assert _FakeLaminar.initialize_calls == [
        {
            "project_api_key": "project-a",
            "disabled_instruments": {"openai", "litellm"},
        }
    ]
    assert len(_FakeLaminar.start_calls) == 1


def test_no_laminar_span_calls_remain() -> None:
    needle = "laminar" + "_span("
    hits: list[str] = []
    for root in (Path("ai_pipeline_core"), Path("tests")):
        for path in root.rglob("*.py"):
            if path.name == "test_laminar_sink.py":
                continue
            if needle in path.read_text(encoding="utf-8"):
                hits.append(str(path))

    assert hits == []
