"""Generic replay execution for recorded spans."""

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from ai_pipeline_core._codec import UniversalCodec, import_by_path
from ai_pipeline_core._llm_core import ModelOptions
from ai_pipeline_core.database._protocol import DatabaseReader, DatabaseWriter
from ai_pipeline_core.deployment._types import _NoopPublisher
from ai_pipeline_core.llm.conversation import _LLM_ROUND_REPLAY_TARGET, Conversation
from ai_pipeline_core.llm.tools import Tool
from ai_pipeline_core.pipeline._execution_context import ReplayExecutionContext, set_execution_context
from ai_pipeline_core.pipeline._runtime_sinks import build_runtime_sinks
from ai_pipeline_core.settings import settings

from ._adapters import _invoke_callable, resolve_callable

__all__ = ["execute_span"]

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _ExecutionOutcome:
    result: Any
    context: ReplayExecutionContext


def _resolve_replay_target(kind: str, target: str, *, span_id: UUID) -> str:
    if target:
        return target
    if kind == "llm_round":
        return _LLM_ROUND_REPLAY_TARGET
    raise ValueError(f"Span {span_id} has an empty target and is not replayable.")


def _parse_json_field(payload_json: str, *, field_name: str, span_id: UUID) -> Any:
    try:
        return json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Span {span_id} has invalid {field_name}. Store valid JSON in spans.{field_name} before replaying.") from exc


def _merge_model_options(base: ModelOptions | None, overrides: dict[str, Any] | None) -> ModelOptions | None:
    if not overrides:
        return base
    base_payload = base.model_dump(mode="python", exclude_none=False) if base is not None else {}
    base_payload.update(overrides)
    return ModelOptions.model_validate(base_payload)


def _override_tools_in_recorded_order(value: Any, override_tools: dict[str, Tool]) -> list[Tool]:
    if not isinstance(value, (list, tuple)):
        return list(override_tools.values())
    ordered: list[Tool] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name in override_tools:
            ordered.append(override_tools[name])
    return ordered or list(override_tools.values())


def _materialize_recorded_tool(item: Any) -> Tool:
    if not isinstance(item, dict):
        raise TypeError(f"Recorded tool config must decode to a JSON object, got {type(item).__name__}.")
    class_path = item.get("class_path")
    constructor_args = item.get("constructor_args", {})
    if not isinstance(class_path, str) or not class_path:
        raise TypeError("Recorded tool config is missing class_path. Replay requires class_path and constructor_args for every stored tool.")
    tool_cls = import_by_path(class_path)
    if not isinstance(tool_cls, type) or not issubclass(tool_cls, Tool):
        raise TypeError(f"Recorded tool path {class_path!r} resolved to {type(tool_cls).__name__}, not a Tool subclass.")
    if not isinstance(constructor_args, dict):
        raise TypeError(f"Recorded constructor_args for tool {class_path!r} must decode to a JSON object.")
    return tool_cls(**constructor_args)


def _materialize_tools(value: Any, override_tools: dict[str, Tool] | None) -> list[Tool] | None:
    if override_tools is not None:
        return _override_tools_in_recorded_order(value, override_tools)
    if not value:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Recorded tools must decode to a list or tuple of tool configs. Got {type(value).__name__}.")
    return [_materialize_recorded_tool(item) for item in value]


def _apply_overrides(
    *,
    receiver: Any,
    arguments: Any,
    overrides: Any | None,
) -> tuple[Any, Any]:
    if overrides is None:
        normalized_arguments = arguments
    else:
        normalized_arguments = arguments
        if isinstance(receiver, dict) and receiver.get("mode") == "decoded_state" and isinstance(receiver.get("value"), Conversation):
            conversation = receiver["value"]
            updated_receiver = conversation
            if getattr(overrides, "model", None):
                updated_receiver = updated_receiver.model_copy(update={"model": overrides.model})
            updated_model_options = _merge_model_options(updated_receiver.model_options, getattr(overrides, "model_options", None))
            if updated_model_options != updated_receiver.model_options:
                updated_receiver = updated_receiver.model_copy(update={"model_options": updated_model_options})
            receiver = {"mode": "decoded_state", "value": updated_receiver}
        if isinstance(normalized_arguments, dict):
            normalized_arguments = dict(normalized_arguments)
            if "tools" in normalized_arguments:
                override_tools = getattr(overrides, "tools", None)
                normalized_arguments["tools"] = _materialize_tools(
                    normalized_arguments["tools"],
                    dict(override_tools) if override_tools is not None else None,
                )
            if getattr(overrides, "response_format", None) is not None and "response_format" in normalized_arguments:
                normalized_arguments["response_format"] = overrides.response_format
            if getattr(overrides, "model", None) is not None and "model" in normalized_arguments:
                normalized_arguments["model"] = overrides.model
            if "model_options" in normalized_arguments:
                normalized_arguments["model_options"] = _merge_model_options(
                    normalized_arguments.get("model_options"),
                    getattr(overrides, "model_options", None),
                )
    if overrides is None and isinstance(arguments, dict) and "tools" in arguments:
        normalized_arguments = dict(arguments)
        normalized_arguments["tools"] = _materialize_tools(arguments["tools"], None)
    return receiver, normalized_arguments


async def _copy_blob(blob_sha: str, *, source_db: DatabaseReader, sink_db: DatabaseWriter) -> None:
    blob = await source_db.get_blob(blob_sha)
    if blob is None:
        raise FileNotFoundError(f"Replay could not copy blob {blob_sha[:12]}... into the sink database because it is missing from the source database.")
    await sink_db.save_blob(blob)


async def _copy_document(document_sha: str, *, source_db: DatabaseReader, sink_db: DatabaseWriter) -> None:
    document = await source_db.get_document(document_sha)
    if document is None:
        raise FileNotFoundError(f"Replay could not copy document {document_sha[:12]}... into the sink database because it is missing from the source database.")
    hydrated = await source_db.get_document_with_content(document_sha)
    if hydrated is None:
        raise FileNotFoundError(
            f"Replay could not hydrate document {document_sha[:12]}... from the source database. "
            "Persist the document record and all referenced blobs before replaying."
        )
    await sink_db.save_document(document)
    from ai_pipeline_core.database import BlobRecord

    blobs = [BlobRecord(content_sha256=hydrated.record.content_sha256, content=hydrated.content)]
    for att_sha, att_content in hydrated.attachment_contents.items():
        blobs.append(BlobRecord(content_sha256=att_sha, content=att_content))
    await sink_db.save_blob_batch(blobs)


async def _copy_input_artifacts(
    *,
    input_document_shas: tuple[str, ...],
    input_blob_shas: tuple[str, ...],
    source_db: DatabaseReader,
    sink_db: DatabaseWriter | None,
) -> None:
    if sink_db is None or sink_db is source_db:
        return
    for blob_sha in input_blob_shas:
        await _copy_blob(blob_sha, source_db=source_db, sink_db=sink_db)
    for document_sha in input_document_shas:
        await _copy_document(document_sha, source_db=source_db, sink_db=sink_db)


async def _execute_span_internal(
    span_id: UUID,
    *,
    source_db: DatabaseReader,
    sink_db: DatabaseWriter | None,
    overrides: Any | None = None,
) -> _ExecutionOutcome:
    span = await source_db.get_span(span_id)
    if span is None:
        raise FileNotFoundError(f"Span {span_id} was not found in the source database.")
    replay_target = _resolve_replay_target(span.kind, span.target, span_id=span.span_id)

    codec = UniversalCodec()
    raw_receiver = _parse_json_field(span.receiver_json, field_name="receiver_json", span_id=span.span_id) if span.receiver_json else None
    raw_input = _parse_json_field(span.input_json, field_name="input_json", span_id=span.span_id) if span.input_json else _MISSING

    await _copy_input_artifacts(
        input_document_shas=span.input_document_shas,
        input_blob_shas=span.input_blob_shas,
        source_db=source_db,
        sink_db=sink_db,
    )

    decoded_receiver = None
    if raw_receiver is not None:
        if not isinstance(raw_receiver, dict):
            raise TypeError(f"Span {span.span_id} receiver_json must decode to a JSON object with mode/value fields.")
        decoded_receiver = {
            "mode": raw_receiver.get("mode"),
            "value": await codec.decode_async(raw_receiver.get("value"), db=source_db),
        }
    decoded_input = _MISSING
    if raw_input is not _MISSING:
        decoded_input = await codec.decode_async(raw_input, db=source_db)

    decoded_receiver, decoded_input = _apply_overrides(
        receiver=decoded_receiver,
        arguments=decoded_input,
        overrides=overrides,
    )

    replay_context = ReplayExecutionContext.create(
        source_span_id=span.span_id,
        database=sink_db,
        publisher=_NoopPublisher(),
        sinks=build_runtime_sinks(database=sink_db, settings_obj=settings),
    )

    callable_obj = resolve_callable(replay_target, decoded_receiver)
    with set_execution_context(replay_context):
        result = await _invoke_callable(callable_obj, decoded_input)
    return _ExecutionOutcome(result=result, context=replay_context)


async def execute_span(
    span_id: UUID,
    *,
    source_db: DatabaseReader,
    sink_db: DatabaseWriter | None = None,
) -> Any:
    """Replay one recorded span against live code."""
    outcome = await _execute_span_internal(
        span_id,
        source_db=source_db,
        sink_db=sink_db,
    )
    return outcome.result
