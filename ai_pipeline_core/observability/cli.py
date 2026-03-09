"""CLI tool for execution-tree inspection."""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

from ai_pipeline_core.database import create_database_from_settings, download_deployment
from ai_pipeline_core.database._factory import Database
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._summary import generate_summary
from ai_pipeline_core.database._types import ExecutionLog, ExecutionNode
from ai_pipeline_core.settings import Settings

__all__ = ["_parse_execution_id", "_resolve_connection", "_resolve_identifier", "main"]


def _parse_execution_id(value: str) -> UUID:
    """Parse a CLI execution identifier."""
    try:
        return UUID(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid execution id {value!r}. Expected a UUID.") from exc


def _resolve_connection(args: argparse.Namespace) -> Database:
    """Resolve CLI connection parameters."""
    if getattr(args, "db_path", None):
        return FilesystemDatabase(Path(args.db_path).resolve())

    settings = Settings()
    if not settings.clickhouse_host:
        raise SystemExit("ClickHouse is not configured. Set CLICKHOUSE_HOST or use --db-path with a FilesystemDatabase snapshot.")
    return create_database_from_settings(settings)


async def _resolve_identifier_async(identifier: str, client: DatabaseReader) -> tuple[UUID, str]:
    """Resolve a deployment or node identifier to a concrete deployment."""
    try:
        node = await client.get_node(_parse_execution_id(identifier))
        if node is not None:
            return node.deployment_id, node.run_id
    except SystemExit:
        pass

    deployment = await client.get_deployment_by_run_id(identifier)
    if deployment is None:
        raise SystemExit(f"Could not resolve {identifier!r} to a deployment. Pass a deployment/node UUID or a known run_id from ai-trace list.")
    return deployment.deployment_id, deployment.run_id


def _resolve_identifier(identifier: str, client: DatabaseReader) -> tuple[UUID, str]:
    """Resolve a CLI identifier to a concrete deployment."""
    return asyncio.run(_resolve_identifier_async(identifier, client))


def _format_duration(node: ExecutionNode) -> str:
    """Format a deployment duration for list output."""
    if node.ended_at is None:
        return "running"
    return str(node.ended_at - node.started_at)


def _print_logs(logs: list[ExecutionLog]) -> None:
    """Render execution logs in chronological order."""
    if not logs:
        print("\nLogs: none")
        return

    print("\nLogs")
    for log in logs:
        timestamp = log.timestamp.isoformat()
        print(f"{timestamp} {log.level} {log.category} {log.logger_name}: {log.message}")
        if log.fields and log.fields != "{}":
            print(f"  fields: {log.fields}")
        if log.exception_text:
            for line in log.exception_text.splitlines():
                print(f"  {line}")


async def _list_deployments_async(database: Database, limit: int, status: str | None) -> int:
    try:
        deployments = await database.list_deployments(limit=limit, status=status)
    finally:
        await database.shutdown()

    if not deployments:
        print("No deployments found.")
        return 0

    for node in deployments:
        print(
            f"{node.deployment_id}  {node.status.value:9}  {node.started_at.isoformat()}  "
            f"{node.deployment_name}  run_id={node.run_id}  duration={_format_duration(node)}"
        )
    return 0


async def _show_deployment_async(database: Database, identifier: str) -> int:
    try:
        deployment_id, _run_id = await _resolve_identifier_async(identifier, database)
        summary = await generate_summary(database, deployment_id)
        logs = await database.get_deployment_logs(deployment_id)
    finally:
        await database.shutdown()

    print(summary)
    _print_logs(logs)
    return 0


async def _download_deployment_async(
    database: Database,
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
    show_parser.add_argument("identifier", help="Deployment/node UUID or deployment run_id")
    show_parser.add_argument("--db-path", type=str, default=None, help="Use a FilesystemDatabase snapshot instead of ClickHouse")

    download_parser = subparsers.add_parser("download", help="Download a deployment as a FilesystemDatabase snapshot")
    download_parser.add_argument("identifier", help="Deployment/node UUID or deployment run_id")
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
