"""CLI tool for span-tree inspection."""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

from ai_pipeline_core.database._factory import Database
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import SpanRecord
from ai_pipeline_core.database.clickhouse._backend import ClickHouseDatabase
from ai_pipeline_core.database.filesystem._backend import FilesystemDatabase
from ai_pipeline_core.database.snapshot._download import download_deployment
from ai_pipeline_core.database.snapshot._spans import build_span_tree_view, format_span_overview_lines, format_span_tree_lines
from ai_pipeline_core.database.snapshot._summary import generate_summary
from ai_pipeline_core.logger._types import LogRecord
from ai_pipeline_core.settings import settings

__all__ = [
    "_parse_execution_id",
    "_resolve_connection",
    "_resolve_identifier",
    "main",
    "show_deployment",
]

TraceDatabase = Database | FilesystemDatabase | ClickHouseDatabase


def _parse_execution_id(value: str) -> UUID:
    """Parse a CLI execution identifier."""
    try:
        return UUID(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid execution id {value!r}. Expected a UUID.") from exc


def _resolve_connection(args: argparse.Namespace) -> TraceDatabase:
    """Resolve CLI connection parameters."""
    if getattr(args, "db_path", None):
        base_path = Path(args.db_path).resolve()
        return FilesystemDatabase(base_path, read_only=True)

    if not settings.clickhouse_host:
        raise SystemExit("ClickHouse is not configured. Set CLICKHOUSE_HOST or use --db-path with a FilesystemDatabase snapshot.")
    return ClickHouseDatabase(settings=settings)


async def _resolve_identifier_async(identifier: str, client: DatabaseReader) -> tuple[UUID, str]:
    """Resolve a deployment or span identifier to a concrete deployment."""
    try:
        execution_id = _parse_execution_id(identifier)
    except SystemExit:
        execution_id = None

    if execution_id is not None:
        span = await client.get_span(execution_id)
        if span is not None:
            return span.root_deployment_id, span.run_id

    deployment = await client.get_deployment_by_run_id(identifier)
    if deployment is not None:
        return deployment.root_deployment_id, deployment.run_id

    raise SystemExit(f"Could not resolve {identifier!r} to a deployment. Pass a deployment/span UUID or a known run_id from ai-trace list.")


def _resolve_identifier(identifier: str, client: DatabaseReader) -> tuple[UUID, str]:
    """Resolve a CLI identifier to a concrete deployment."""
    return asyncio.run(_resolve_identifier_async(identifier, client))


def _format_duration(node: SpanRecord) -> str:
    """Format a deployment duration for list output."""
    if node.ended_at is None:
        return "running"
    return str(node.ended_at - node.started_at)


def _log_sort_key(log: LogRecord) -> tuple[object, int, str]:
    return log.timestamp, log.sequence_no, str(log.span_id)


def _print_logs(logs: list[LogRecord]) -> None:
    """Render execution logs in chronological order."""
    if not logs:
        print("\nLogs: none")
        return

    print("\nLogs")
    for log in sorted(logs, key=_log_sort_key):
        timestamp = log.timestamp.isoformat()
        print(f"{timestamp} {log.level} {log.category} {log.logger_name}: {log.message}")
        fields = log.fields_json
        if fields and fields != "{}":
            print(f"  fields: {fields}")
        if log.exception_text:
            for line in log.exception_text.splitlines():
                print(f"  {line}")


async def _get_tree_logs(database: DatabaseReader, deployment_ids: list[UUID]) -> list[LogRecord]:
    return await database.get_deployment_logs_batch(deployment_ids)


async def show_deployment(reader: DatabaseReader, deployment_id: UUID) -> str:
    """Render a deployment summary for ai-trace show."""
    return await generate_summary(reader, deployment_id)


def _render_deployment_v2(tree: list[SpanRecord], root_deployment_id: UUID) -> str:
    view = build_span_tree_view(tree, root_deployment_id)
    if view is None:
        return "No execution data found."

    lines = format_span_overview_lines(view)
    lines.extend(["", "Tree", ""])
    lines.extend(format_span_tree_lines(view))
    return "\n".join(lines)


async def _list_deployments_async(database: TraceDatabase, limit: int, status: str | None) -> int:
    try:
        deployments = await database.list_deployments(limit=limit, status=status)
    finally:
        await database.shutdown()

    if not deployments:
        print("No deployments found.")
        return 0

    for node in deployments:
        status_text = getattr(node.status, "value", str(node.status))
        print(
            f"{node.deployment_id}  {status_text:9}  {node.started_at.isoformat()}  "
            f"{node.deployment_name}  run_id={node.run_id}  duration={_format_duration(node)}"
        )
    return 0


async def _show_deployment_async(database: TraceDatabase, identifier: str) -> int:
    try:
        deployment_id, _run_id = await _resolve_identifier_async(identifier, database)
        tree = await database.get_deployment_tree(deployment_id)
        summary = _render_deployment_v2(tree, deployment_id)
        logs = await _get_tree_logs(database, sorted({span.deployment_id for span in tree}, key=str))
    finally:
        await database.shutdown()

    print(summary)
    _print_logs(logs)
    return 0


async def _download_deployment_async(
    database: TraceDatabase,
    identifier: str,
    output_dir: Path,
) -> int:
    try:
        deployment_id, _run_id = await _resolve_identifier_async(identifier, database)
        await download_deployment(database, deployment_id, output_dir)
    finally:
        await database.shutdown()
    print(f"Downloaded deployment to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ai-trace CLI."""
    parser = argparse.ArgumentParser(prog="ai-trace", description="Inspect deployment execution trees")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List recent deployments")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of deployments to show")
    list_parser.add_argument("--status", type=str, default=None, help="Filter deployments by status")
    list_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    show_parser = subparsers.add_parser("show", help="Show deployment summary and logs")
    show_parser.add_argument("identifier", help="Deployment/span UUID or deployment run_id")
    show_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    download_parser = subparsers.add_parser("download", help="Download a deployment as a FilesystemDatabase snapshot")
    download_parser.add_argument("identifier", help="Deployment/span UUID or deployment run_id")
    download_parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory for the snapshot")
    download_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    try:
        database = _resolve_connection(args)
        if args.command == "list":
            return asyncio.run(_list_deployments_async(database, args.limit, args.status))
        if args.command == "show":
            return asyncio.run(_show_deployment_async(database, args.identifier))
        if args.command == "download":
            return asyncio.run(_download_deployment_async(database, args.identifier, Path(args.output_dir).resolve()))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 1
