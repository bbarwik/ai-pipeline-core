"""CLI tool for replaying captured pipeline conversations, tasks, and flows.

Usage:
    python -m ai_pipeline_core.replay run conversation.yaml --import examples.showcase
    python -m ai_pipeline_core.replay run task.yaml --set model=grok-4.1-fast --import my_app.tasks
    python -m ai_pipeline_core.replay show flow.yaml
"""

import argparse
import asyncio
import importlib
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

from ai_pipeline_core.deployment._helpers import init_observability_best_effort
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.observability._debug import FilesystemBackend, TraceDebugConfig
from ai_pipeline_core.observability._initialization import register_pipeline_processor
from ai_pipeline_core.observability._tracking._processor import PipelineSpanProcessor

from .types import ConversationReplay, FlowReplay, TaskReplay, _infer_store_base

logger = get_pipeline_logger(__name__)

_PAYLOAD_CLASSES: dict[str, type[ConversationReplay | TaskReplay | FlowReplay]] = {
    "conversation": ConversationReplay,
    "pipeline_task": TaskReplay,
    "pipeline_flow": FlowReplay,
}

_CONTENT_PREVIEW_LENGTH = 200

_MAIN_MODULE_PATTERN = re.compile(r"__main__:")


def _init_replay_tracing(output_dir: Path) -> tuple[PipelineSpanProcessor, FilesystemBackend] | None:
    """Initialize debug tracing for replay execution, writing to output_dir/.trace/.

    Sets up the OTel TracerProvider (via Laminar), then registers a
    PipelineSpanProcessor with FilesystemBackend so all spans produced
    during replay are captured.
    Returns the processor and backend for shutdown, or None if setup fails.
    """
    # Ensure OTel TracerProvider is initialized
    try:
        init_observability_best_effort()
    except (OSError, RuntimeError, ImportError) as e:
        logger.debug("Observability init skipped: %s", e)

    trace_path = output_dir / ".trace"
    if trace_path.exists():
        shutil.rmtree(trace_path)
    trace_path.mkdir(parents=True, exist_ok=True)

    config = TraceDebugConfig(path=trace_path)
    fs_backend = FilesystemBackend(config)
    processor = PipelineSpanProcessor(backends=(fs_backend,), verbose=config.verbose)

    provider: Any = otel_trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)
        register_pipeline_processor(processor)
        return processor, fs_backend

    logger.debug("TracerProvider has no add_span_processor — tracing disabled")
    fs_backend.shutdown()
    return None


def _import_modules(modules: list[str]) -> None:
    """Import user-specified modules so their Document subclasses and functions are registered."""
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import '{module_name}': {e}", file=sys.stderr)


def _remap_main_references(payload: ConversationReplay | TaskReplay | FlowReplay, imported_modules: list[str]) -> ConversationReplay | TaskReplay | FlowReplay:
    """Remap __main__:X references to the actual module path.

    When a script is run as __main__, all function/class paths use __main__ as the module.
    During replay, __main__ is the replay CLI, so we search imported modules for the symbol.
    """
    if not imported_modules:
        return payload

    updates: dict[str, Any] = {}

    # Remap function_path for task/flow payloads
    if isinstance(payload, (TaskReplay, FlowReplay)) and _MAIN_MODULE_PATTERN.match(payload.function_path):
        qualname = payload.function_path.split(":", 1)[1]
        resolved = _find_symbol_in_modules(qualname, imported_modules)
        if resolved:
            updates["function_path"] = resolved
            logger.debug("Remapped function_path: __main__:%s -> %s", qualname, resolved)

    # Remap response_format for conversation payloads
    if isinstance(payload, ConversationReplay) and payload.response_format and _MAIN_MODULE_PATTERN.match(payload.response_format):
        qualname = payload.response_format.split(":", 1)[1]
        resolved = _find_symbol_in_modules(qualname, imported_modules)
        if resolved:
            updates["response_format"] = resolved
            logger.debug("Remapped response_format: __main__:%s -> %s", qualname, resolved)

    return payload.model_copy(update=updates) if updates else payload


def _find_symbol_in_modules(qualname: str, modules: list[str]) -> str | None:
    """Search imported modules for a symbol by qualname, return 'module:qualname' if found."""
    top_name = qualname.split(".", maxsplit=1)[0]
    for module_name in modules:
        module = sys.modules.get(module_name)
        if module and hasattr(module, top_name):
            return f"{module_name}:{qualname}"
    return None


def _load_payload(path: Path) -> ConversationReplay | TaskReplay | FlowReplay:
    """Load a replay YAML file and return the typed payload."""
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}")

    data = cast(dict[str, Any], raw)
    payload_type = data.get("payload_type")
    if payload_type not in _PAYLOAD_CLASSES:
        valid = ", ".join(sorted(_PAYLOAD_CLASSES))
        raise ValueError(
            f"Unknown payload_type '{payload_type}' in {path}. Valid types: {valid}. Check that the file is a replay YAML produced by the trace writer."
        )

    cls = _PAYLOAD_CLASSES[payload_type]
    return cls.from_yaml(text)


def _apply_overrides(
    payload: ConversationReplay | TaskReplay | FlowReplay,
    overrides: list[str],
) -> ConversationReplay | TaskReplay | FlowReplay:
    """Apply --set KEY=VALUE overrides to a payload via model_copy."""
    if not overrides:
        return payload

    updates: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set format: '{item}'. Expected KEY=VALUE (e.g., --set model=grok-4.1-fast).")
        key, value = item.split("=", 1)
        field_names = set(type(payload).model_fields)
        if key not in field_names:
            raise ValueError(f"Unknown field '{key}' for {type(payload).__name__}. Valid fields: {', '.join(sorted(field_names))}.")
        updates[key] = value

    return payload.model_copy(update=updates)


def _format_result(result: Any) -> str:
    """Format an execution result for terminal output."""
    if hasattr(result, "content") and hasattr(result, "usage"):
        # Conversation result
        content = result.content or ""
        preview = content[:_CONTENT_PREVIEW_LENGTH] + ("..." if len(content) > _CONTENT_PREVIEW_LENGTH else "")
        lines = [preview]
        if result.usage:
            lines.append(f"\n[tokens: {result.usage.total_tokens}, cost: ${result.cost:.4f}]")
        if hasattr(result, "parsed") and result.parsed is not None:
            lines.append(f"[parsed: {type(result.parsed).__name__}]")
        return "\n".join(lines)

    if isinstance(result, Document):
        doc = cast("Document[Any]", result)
        return f"[Document: {type(doc).__name__}] {doc.name} ({len(doc.content)} bytes)"

    if isinstance(result, list):
        results = cast(list[Any], result)
        items: list[str] = []
        for d in results:
            if isinstance(d, Document):
                typed_d = cast("Document[Any]", d)
                items.append(f"  {type(typed_d).__name__}: {typed_d.name}")
            else:
                items.append(f"  {d!r}")
        return f"[{len(results)} result(s)]\n" + "\n".join(items)

    return repr(result)


def _serialize_result(result: Any) -> dict[str, Any]:
    """Serialize an execution result to a dict for YAML output."""
    output: dict[str, Any] = {"timestamp": datetime.now(UTC).isoformat()}

    if hasattr(result, "content") and hasattr(result, "usage"):
        output["type"] = "conversation"
        output["content"] = result.content or ""
        if result.usage:
            usage = result.usage
            output["usage"] = usage.model_dump() if isinstance(usage, BaseModel) else {"total_tokens": usage.total_tokens}
        output["cost"] = result.cost
        if hasattr(result, "parsed") and result.parsed is not None:
            output["parsed_type"] = type(result.parsed).__qualname__
            output["parsed"] = result.parsed.model_dump() if isinstance(result.parsed, BaseModel) else str(result.parsed)
        return output

    if isinstance(result, Document):
        doc = cast("Document[Any]", result)
        output["type"] = "document"
        output["class_name"] = type(doc).__name__
        output["name"] = doc.name
        output["content_bytes"] = len(doc.content)
        output["sha256"] = doc.sha256
        return output

    if isinstance(result, list):
        results = cast(list[Any], result)
        output["type"] = "document_list"
        output["count"] = len(results)
        docs_list: list[dict[str, Any]] = []
        for d in results:
            if isinstance(d, Document):
                typed_d = cast("Document[Any]", d)
                docs_list.append({"class_name": type(typed_d).__name__, "name": typed_d.name, "content_bytes": len(typed_d.content), "sha256": typed_d.sha256})
            else:
                docs_list.append({"value": repr(d)})
        output["documents"] = docs_list
        return output

    output["type"] = "unknown"
    output["value"] = repr(result)
    return output


def _write_output(output_dir: Path, result: Any) -> Path:
    """Write replay execution output to output_dir/output.yaml."""
    output_data = _serialize_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.yaml"
    output_path.write_text(
        yaml.dump(output_data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def _print_run_header(
    payload: ConversationReplay | TaskReplay | FlowReplay,
    replay_file: Path,
    store_base: Path,
    output_dir: Path | None,
) -> None:
    """Print replay execution header with payload details."""
    print(f"Replaying {type(payload).__name__} from {replay_file.name}")
    if isinstance(payload, ConversationReplay):
        print(f"  model: {payload.model}")
    else:
        print(f"  function: {payload.function_path}")
    print(f"  store: {store_base}")
    if output_dir is not None:
        print(f"  output: {output_dir}")
    print()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a replay YAML file."""
    replay_file = Path(args.replay_file).resolve()
    if not replay_file.exists():
        print(f"Error: File not found: {replay_file}", file=sys.stderr)
        return 1

    imported_modules: list[str] = args.modules or []
    _import_modules(imported_modules)

    try:
        payload = _load_payload(replay_file)
        payload = _remap_main_references(payload, imported_modules)
        payload = _apply_overrides(payload, args.set or [])
    except (ValueError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    store_base: Path
    if args.store:
        store_base = Path(args.store).resolve()
    else:
        try:
            store_base = _infer_store_base(replay_file)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = replay_file.parent / f"{replay_file.stem}_replay"

    # Initialize tracing
    tracing_pair: tuple[PipelineSpanProcessor, FilesystemBackend] | None = None
    if not args.no_trace:
        tracing_pair = _init_replay_tracing(output_dir)

    _print_run_header(payload, replay_file, store_base, output_dir if not args.no_trace else None)

    try:
        result = asyncio.run(payload.execute(store_base))
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        logger.debug("Replay execution failed", exc_info=True)
        return 1
    finally:
        if tracing_pair is not None:
            _processor, fs_backend = tracing_pair
            fs_backend.shutdown()

    print(_format_result(result))

    try:
        _write_output(output_dir, result)
        print(f"\n[output: {output_dir.relative_to(replay_file.parent)}]")
    except OSError as e:
        print(f"Warning: Could not save output: {e}", file=sys.stderr)

    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Pretty-print a replay YAML file."""
    replay_file = Path(args.replay_file).resolve()
    if not replay_file.exists():
        print(f"Error: File not found: {replay_file}", file=sys.stderr)
        return 1

    try:
        payload = _load_payload(replay_file)
    except (ValueError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Type: {type(payload).__name__}")

    if isinstance(payload, ConversationReplay):
        print(f"Model: {payload.model}")
        if payload.model_options:
            print(f"Options: {payload.model_options}")
        if payload.response_format:
            print(f"Response format: {payload.response_format}")
        print(f"Context docs: {len(payload.context)}")
        print(f"History entries: {len(payload.history)}")
        print(f"\nPrompt:\n  {payload.prompt}")

    elif isinstance(payload, TaskReplay):
        print(f"Function: {payload.function_path}")
        task_args: dict[str, Any] = payload.arguments
        print(f"Arguments: {len(task_args)}")
        for key, value in task_args.items():
            preview = str(value)[:80]
            if isinstance(value, dict) and "$doc_ref" in value:
                print(f"  {key}: [doc_ref {value['$doc_ref'][:12]}...]")
            else:
                print(f"  {key}: {preview}")

    else:
        print(f"Function: {payload.function_path}")
        print(f"Run ID: {payload.run_id}")
        print(f"Documents: {len(payload.documents)}")
        if payload.flow_options:
            print(f"Flow options: {payload.flow_options}")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for replay operations.

    ``--import MODULE`` also remaps ``__main__:X`` references to ``MODULE:X`` in replay
    payloads — required when replaying scripts originally run as ``python script.py``.

    Usage:
        ai-replay show conversation.yaml
        ai-replay run conversation.yaml --store ./output
        ai-replay run task.yaml --set model=grok-4.1-fast --import my_app
        ai-replay run flow.yaml --output-dir ./replay_out --import my_app
    """
    parser = argparse.ArgumentParser(prog="ai-replay", description="Execute or inspect replay YAML files")
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Execute a replay YAML file")
    run_parser.add_argument("replay_file", help="Path to replay YAML file (conversation.yaml, task.yaml, flow.yaml)")
    run_parser.add_argument("--store", type=str, help="Override store base path (default: inferred from .trace/ ancestor)")
    run_parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override a field before execution (repeatable)")
    run_parser.add_argument(
        "--import",
        dest="modules",
        action="append",
        metavar="MODULE",
        help="Import a module before replay (registers Document subclasses and functions)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results and traces (default: {replay_file_stem}_replay/ next to replay file)",
    )
    run_parser.add_argument(
        "--no-trace",
        action="store_true",
        default=False,
        help="Skip tracing setup — only print result without producing .trace/ output",
    )

    # show
    show_parser = subparsers.add_parser("show", help="Pretty-print a replay YAML file")
    show_parser.add_argument("replay_file", help="Path to replay YAML file")

    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {"run": _cmd_run, "show": _cmd_show}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


__all__ = ["main"]
