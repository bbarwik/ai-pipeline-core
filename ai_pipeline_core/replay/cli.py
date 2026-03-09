"""CLI tool for replaying captured pipeline conversations, tasks, and flows."""

import argparse
import asyncio
import importlib
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import yaml
from pydantic import BaseModel

from ai_pipeline_core.database import create_database_from_settings
from ai_pipeline_core.database._factory import Database
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import NodeKind
from ai_pipeline_core.documents.document import Document
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import Settings

from .types import ConversationReplay, FlowReplay, TaskReplay, _infer_db_path

logger = get_pipeline_logger(__name__)

_PAYLOAD_CLASSES: dict[str, type[ConversationReplay | TaskReplay | FlowReplay]] = {
    "conversation": ConversationReplay,
    "pipeline_task": TaskReplay,
    "pipeline_flow": FlowReplay,
}

_NODE_KIND_TO_PAYLOAD_TYPE: dict[NodeKind, str] = {
    NodeKind.CONVERSATION_TURN: "conversation",
    NodeKind.TASK: "pipeline_task",
    NodeKind.FLOW: "pipeline_flow",
}

_CONTENT_PREVIEW_LENGTH = 200
_MAIN_MODULE_PATTERN = re.compile(r"__main__:")


def _string_key_dict(data: Any) -> dict[str, Any]:
    """Normalize arbitrary YAML/object mappings to ``dict[str, Any]``."""
    if not isinstance(data, dict):
        return {}
    mapping = cast(dict[Any, Any], data)
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        normalized[str(key)] = value
    return normalized


def _import_modules(modules: list[str]) -> None:
    """Import user-specified modules so their Document subclasses and functions are registered."""
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            print(f"Warning: Could not import '{module_name}': {exc}", file=sys.stderr)


def _remap_main_references(payload: ConversationReplay | TaskReplay | FlowReplay, imported_modules: list[str]) -> ConversationReplay | TaskReplay | FlowReplay:
    """Remap __main__:X references to the actual module path."""
    if not imported_modules:
        return payload

    updates: dict[str, Any] = {}

    if isinstance(payload, (TaskReplay, FlowReplay)) and _MAIN_MODULE_PATTERN.match(payload.function_path):
        qualname = payload.function_path.split(":", 1)[1]
        resolved = _find_symbol_in_modules(qualname, imported_modules)
        if resolved:
            updates["function_path"] = resolved
            logger.debug("Remapped function_path: __main__:%s -> %s", qualname, resolved)

    if isinstance(payload, ConversationReplay) and payload.response_format and _MAIN_MODULE_PATTERN.match(payload.response_format):
        qualname = payload.response_format.split(":", 1)[1]
        resolved = _find_symbol_in_modules(qualname, imported_modules)
        if resolved:
            updates["response_format"] = resolved
            logger.debug("Remapped response_format: __main__:%s -> %s", qualname, resolved)

    return payload.model_copy(update=updates) if updates else payload


def _find_symbol_in_modules(qualname: str, modules: list[str]) -> str | None:
    """Search imported modules for a symbol by qualname."""
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

    data = _string_key_dict(raw)
    payload_type_raw = data.get("payload_type")
    payload_type = payload_type_raw if isinstance(payload_type_raw, str) else None
    if payload_type not in _PAYLOAD_CLASSES:
        valid = ", ".join(sorted(_PAYLOAD_CLASSES))
        raise ValueError(f"Unknown payload_type '{payload_type}' in {path}. Valid types: {valid}. Check that the file is a supported replay YAML payload.")

    cls = _PAYLOAD_CLASSES[payload_type]
    return cls.from_yaml(text)


async def _load_payload_from_database(database: DatabaseReader, node_id: UUID) -> ConversationReplay | TaskReplay | FlowReplay:
    """Load a replay payload from an execution node."""
    node = await database.get_node(node_id)
    if node is None:
        raise FileNotFoundError(
            f"Execution node {node_id} was not found in the database. Use ai-trace list/show to inspect available deployment and node identifiers."
        )

    payload_data = node.payload.get("replay_payload")
    if not isinstance(payload_data, dict):
        fallback_type = _NODE_KIND_TO_PAYLOAD_TYPE.get(node.node_kind)
        if fallback_type is None:
            raise ValueError(
                f"Execution node {node_id} ({node.node_kind.value}) does not carry a replay payload. "
                f"Replay is supported for conversation_turn, task, and flow nodes."
            )
        raise TypeError(
            f"Execution node {node_id} ({node.node_kind.value}, status={node.status.value}) has no replay payload and cannot be replayed. "
            "Replay requires payload['replay_payload'] to be persisted on the conversation_turn/task/flow node, "
            "including failed nodes that recorded replay data."
        )
    payload_data = _string_key_dict(payload_data)

    payload_type_raw = payload_data.get("payload_type")
    payload_type = payload_type_raw if isinstance(payload_type_raw, str) else None
    if payload_type not in _PAYLOAD_CLASSES:
        fallback_type = _NODE_KIND_TO_PAYLOAD_TYPE.get(node.node_kind)
        if fallback_type is None:
            raise ValueError(
                f"Execution node {node_id} has an unsupported replay payload_type={payload_type!r}. "
                f"Replay is supported for conversation, pipeline_task, and pipeline_flow payloads."
            )
        payload_type = fallback_type
        payload_data = dict(payload_data)
        payload_data["payload_type"] = payload_type

    payload_cls = _PAYLOAD_CLASSES[payload_type]
    return payload_cls.model_validate(payload_data)


def _apply_overrides(
    payload: ConversationReplay | TaskReplay | FlowReplay,
    overrides: list[str],
) -> ConversationReplay | TaskReplay | FlowReplay:
    """Apply --set KEY=VALUE overrides with YAML parsing and Pydantic validation."""
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
        updates[key] = yaml.safe_load(value)

    updated_data = payload.model_dump(mode="python")
    updated_data.update(updates)
    return type(payload).model_validate(updated_data)


def _format_result(result: Any) -> str:
    """Format an execution result for terminal output."""
    if hasattr(result, "content") and hasattr(result, "usage"):
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

    if isinstance(result, (list, tuple)):
        results = cast("list[Any] | tuple[Any, ...]", result)
        items: list[str] = []
        for item in results:
            if isinstance(item, Document):
                typed_doc = cast("Document[Any]", item)
                items.append(f"  {type(typed_doc).__name__}: {typed_doc.name}")
                continue
            items.append(f"  {item!r}")
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

    if isinstance(result, (list, tuple)):
        results = cast("list[Any] | tuple[Any, ...]", result)
        output["type"] = "document_list"
        output["count"] = len(results)
        documents: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, Document):
                typed_doc = cast("Document[Any]", item)
                documents.append({
                    "class_name": type(typed_doc).__name__,
                    "name": typed_doc.name,
                    "content_bytes": len(typed_doc.content),
                    "sha256": typed_doc.sha256,
                })
                continue
            documents.append({"value": repr(item)})
        output["documents"] = documents
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
    source_label: str,
    database_label: str,
    output_dir: Path | None,
) -> None:
    """Print replay execution header with payload details."""
    print(f"Replaying {type(payload).__name__} from {source_label}")
    if isinstance(payload, ConversationReplay):
        print(f"  model: {payload.model}")
    else:
        print(f"  function: {payload.function_path}")
    print(f"  database: {database_label}")
    if output_dir is not None:
        print(f"  output: {output_dir}")
    print()


def _resolve_database_for_file(replay_file: Path, db_path: str | None) -> tuple[Database, str]:
    """Open the database used to resolve document refs for a replay YAML file."""
    resolved_path = Path(db_path).resolve() if db_path else _infer_db_path(replay_file)
    return FilesystemDatabase(resolved_path), str(resolved_path)


def _resolve_database_for_node(db_path: str | None) -> tuple[Database, str]:
    """Open the database used to load a replay payload by node id."""
    if db_path is not None:
        resolved_path = Path(db_path).resolve()
        return FilesystemDatabase(resolved_path), str(resolved_path)

    settings = Settings()
    if not settings.clickhouse_host:
        raise ValueError(
            "--from-db without --db-path requires ClickHouse settings. Set CLICKHOUSE_HOST or provide --db-path pointing at a FilesystemDatabase snapshot."
        )
    return create_database_from_settings(settings), "clickhouse"


async def _execute_with_database(
    payload: ConversationReplay | TaskReplay | FlowReplay,
    database: Database,
) -> Any:
    """Execute a replay payload and always close the database."""
    try:
        return await payload.execute(database)
    finally:
        await database.shutdown()


def _default_output_dir(replay_file: Path | None, from_db: str | None) -> Path:
    """Compute the default output directory for a replay run."""
    if replay_file is not None:
        return replay_file.parent / f"{replay_file.stem}_replay"
    if from_db is not None:
        return Path.cwd() / f"node_{from_db[:8]}_replay"
    return Path.cwd() / "replay_output"


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a replay YAML file or a replay payload loaded from the database."""
    imported_modules: list[str] = args.modules or []
    _import_modules(imported_modules)

    replay_file = Path(args.replay_file).resolve() if args.replay_file else None
    payload: ConversationReplay | TaskReplay | FlowReplay
    database: Database | None = None
    source_label: str
    database_label: str

    try:
        if args.from_db:
            node_id = UUID(args.from_db)
            database, database_label = _resolve_database_for_node(args.db_path)
            try:
                payload = asyncio.run(_load_payload_from_database(database, node_id))
            except Exception:
                asyncio.run(database.shutdown())
                raise
            source_label = f"database node {node_id}"
        else:
            if replay_file is None:
                raise ValueError("Replay file is required unless --from-db is used.")
            if not replay_file.exists():
                print(f"Error: File not found: {replay_file}", file=sys.stderr)
                return 1
            payload = _load_payload(replay_file)
            database, database_label = _resolve_database_for_file(replay_file, args.db_path)
            source_label = replay_file.name

        payload = _remap_main_references(payload, imported_modules)
        payload = _apply_overrides(payload, args.set or [])
    except (FileNotFoundError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(replay_file, args.from_db)
    _print_run_header(payload, source_label, database_label, output_dir)
    assert database is not None

    try:
        result = asyncio.run(_execute_with_database(payload, database))
    except Exception as exc:
        print(f"Error during execution: {exc}", file=sys.stderr)
        logger.debug("Replay execution failed", exc_info=True)
        return 1

    print(_format_result(result))

    try:
        _write_output(output_dir, result)
        print(f"\n[output: {output_dir}]")
    except OSError as exc:
        print(f"Warning: Could not save output: {exc}", file=sys.stderr)

    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Pretty-print a replay YAML file."""
    replay_file = Path(args.replay_file).resolve()
    if not replay_file.exists():
        print(f"Error: File not found: {replay_file}", file=sys.stderr)
        return 1

    try:
        payload = _load_payload(replay_file)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
        print(f"Arguments: {len(payload.arguments)}")
        for key, value in payload.arguments.items():
            preview = str(value)[:80]
            if isinstance(value, dict) and "$doc_ref" in value:
                print(f"  {key}: [doc_ref {value['$doc_ref'][:12]}...]")
                continue
            print(f"  {key}: {preview}")
    else:
        print(f"Function: {payload.function_path}")
        print(f"Run ID: {payload.run_id}")
        print(f"Documents: {len(payload.documents)}")
        if payload.flow_options:
            print(f"Flow options: {payload.flow_options}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for replay operations."""
    parser = argparse.ArgumentParser(prog="ai-replay", description="Execute or inspect replay payloads")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute a replay payload")
    run_parser.add_argument("replay_file", nargs="?", help="Path to replay YAML file (conversation.yaml, task.yaml, flow.yaml)")
    run_parser.add_argument("--from-db", type=str, help="Load replay payload from an execution node ID in the database")
    run_parser.add_argument("--db-path", type=str, help="Use a FilesystemDatabase at this path instead of ClickHouse or auto-discovery")
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
        help="Output directory for replay results",
    )

    show_parser = subparsers.add_parser("show", help="Pretty-print a replay YAML file")
    show_parser.add_argument("replay_file", help="Path to replay YAML file")

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.replay_file is None and args.from_db is None:
            parser.error("run requires either a replay_file or --from-db <node_id>")
        if args.replay_file is not None and args.from_db is not None:
            parser.error("run accepts a replay_file or --from-db <node_id>, not both")

    handlers: dict[str, Any] = {"run": _cmd_run, "show": _cmd_show}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


__all__ = ["main"]
