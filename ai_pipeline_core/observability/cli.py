"""CLI tool for downloading and inspecting pipeline traces from ClickHouse.

Usage:
    ai-trace download 550e8400-e29b-41d4-a716-446655440000 -o ./debug/
    ai-trace list --limit 10 --status completed
    ai-trace show 550e8400-e29b-41d4-a716-446655440000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

__all__ = ["main"]

# ClickHouse queries for list/show subcommands
_LIST_RUNS_QUERY = """
SELECT execution_id, run_id, flow_name, status, start_time, end_time,
       total_cost, total_tokens
FROM pipeline_runs FINAL
WHERE 1=1
{where_clauses}
ORDER BY start_time DESC
LIMIT {{limit:UInt32}}
"""

_SPAN_SUMMARY_QUERY = """
SELECT span_type, count() AS cnt,
       sum(duration_ms) AS total_duration_ms,
       sum(cost) AS total_cost,
       sum(tokens_input) AS total_input,
       sum(tokens_output) AS total_output
FROM tracked_spans FINAL
WHERE execution_id = {execution_id:UUID}
GROUP BY span_type
ORDER BY total_duration_ms DESC
"""

_RUN_METADATA_QUERY = """
SELECT run_id, flow_name, run_scope, status, start_time, end_time,
       total_cost, total_tokens, metadata
FROM pipeline_runs FINAL
WHERE execution_id = {execution_id:UUID}
"""

DEFAULT_LIST_LIMIT = 20
DEFAULT_PORT = 8443
EXECUTION_ID_SHORT_LENGTH = 8

# Table column widths for list output
_COL_ID = 36
_COL_FLOW = 24
_COL_STATUS = 10
_COL_STARTED = 19
_COL_DURATION = 12
_COL_COST = 10
_COL_TOKENS = 12

# ---------------------------------------------------------------------------
# Shared connection parser (parent parser pattern)
# ---------------------------------------------------------------------------

_connection_parser = argparse.ArgumentParser(add_help=False)
_connection_parser.add_argument("--host", help="ClickHouse host (env: CLICKHOUSE_HOST)")
_connection_parser.add_argument("--port", type=int, help="ClickHouse port (env: CLICKHOUSE_PORT, default: 8443)")
_connection_parser.add_argument("--database", help="Database (env: CLICKHOUSE_DATABASE, default: 'default')")
_connection_parser.add_argument("--user", help="Username (env: CLICKHOUSE_USER, default: 'default')")
_connection_parser.add_argument("--password", help="Password (env: CLICKHOUSE_PASSWORD)")
_connection_parser.add_argument("--no-secure", action="store_true", help="Disable TLS (default: secure)")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _resolve_connection(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve ClickHouse connection params: CLI flags > env vars > defaults."""
    from ai_pipeline_core.settings import Settings

    settings = Settings()

    host = args.host or settings.clickhouse_host
    if not host:
        print(
            "Error: ClickHouse host not configured. Set CLICKHOUSE_HOST environment variable or use --host flag.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return {
        "host": host,
        "port": args.port or settings.clickhouse_port,
        "database": args.database or settings.clickhouse_database,
        "username": args.user or settings.clickhouse_user,
        "password": args.password or settings.clickhouse_password,
        "secure": not args.no_secure and settings.clickhouse_secure,
    }


def _create_client(args: argparse.Namespace) -> Any:
    """Create clickhouse_connect client from resolved connection params."""
    try:
        import clickhouse_connect
    except ImportError:
        print(
            "Error: clickhouse-connect package is required. Install with: pip install clickhouse-connect",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    params = _resolve_connection(args)
    try:
        return clickhouse_connect.get_client(**params)
    except Exception as e:
        print(
            f"Error: Could not connect to ClickHouse at {params['host']}:{params['port']}: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _parse_execution_id(value: str) -> UUID:
    """Parse a string as UUID, printing an actionable error on failure."""
    try:
        return UUID(value)
    except ValueError:
        print(
            f"Error: Invalid execution_id '{value}'. Expected a UUID (e.g., 550e8400-e29b-41d4-a716-446655440000).",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_duration(start: datetime | None, end: datetime | None) -> str:
    """Format duration between two timestamps, or 'running...' if end is missing."""
    if start is None:
        return "—"
    if end is None:
        return "running..."
    delta = end - start
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _format_cost(cost: float | None) -> str:
    """Format cost as dollar amount."""
    if cost is None or cost == 0:
        return "—"
    return f"${cost:.4f}"


def _format_tokens(tokens: int | None) -> str:
    """Format token count with comma separators."""
    if tokens is None or tokens == 0:
        return "—"
    return f"{tokens:,}"


def _format_timestamp(ts: datetime | None) -> str:
    """Format datetime for display."""
    if ts is None:
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_download(args: argparse.Namespace) -> int:
    """Download trace and rebuild .trace/ directory from ClickHouse."""
    from ai_pipeline_core.observability._debug._reconstruction import TraceDownloader

    execution_id = _parse_execution_id(args.execution_id)
    client = _create_client(args)
    downloader = TraceDownloader(client=client)

    short_id = str(execution_id)[:EXECUTION_ID_SHORT_LENGTH]
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path(f"./{short_id}_trace").resolve()

    print(f"Downloading trace {execution_id}")
    print(f"  output: {output_path}")

    try:
        result_path = downloader.download_trace(
            execution_id,
            output_path,
            include_documents=args.documents,
            follow_children=args.children,
        )
    except Exception as e:
        print(f"Error: Download failed: {e}", file=sys.stderr)
        logger.debug("Trace download failed", exc_info=True)
        return 1

    # Count span directories (directories with numeric prefix)
    span_dirs = [d for d in result_path.iterdir() if d.is_dir() and d.name[:1].isdigit()]
    span_count = len(span_dirs)

    # Try to extract flow name from summary
    summary_path = result_path / "summary.md"
    flow_name = "unknown"
    if summary_path.exists():
        first_line = summary_path.read_text(encoding="utf-8").split("\n", maxsplit=1)[0]
        if first_line.startswith("# "):
            flow_name = first_line[2:].strip()

    print(f"\nDownloaded: {flow_name}")
    print(f"  spans: {span_count}")
    print(f"  path: {result_path}")

    if args.children:
        child_dirs = [d for d in result_path.iterdir() if d.is_dir() and d.name.startswith("child_")]
        if child_dirs:
            print(f"  children: {len(child_dirs)}")

    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """List recent pipeline runs from ClickHouse."""
    client = _create_client(args)

    where_clauses = ""
    params: dict[str, Any] = {"limit": args.limit}

    if args.status:
        where_clauses += "\nAND status = {status:String}"
        params["status"] = args.status
    if args.flow:
        where_clauses += "\nAND flow_name = {flow_name:String}"
        params["flow_name"] = args.flow

    query = _LIST_RUNS_QUERY.format(where_clauses=where_clauses)

    try:
        result = client.query(query, parameters=params)
    except Exception as e:
        print(f"Error: Query failed: {e}", file=sys.stderr)
        return 1

    rows = result.result_rows
    if not rows:
        print("No runs found.")
        return 0

    # Print header
    header = (
        f"{'EXECUTION_ID':<{_COL_ID}}  "
        f"{'FLOW':<{_COL_FLOW}}  "
        f"{'STATUS':<{_COL_STATUS}}  "
        f"{'STARTED':<{_COL_STARTED}}  "
        f"{'DURATION':<{_COL_DURATION}}  "
        f"{'COST':<{_COL_COST}}  "
        f"{'TOKENS':<{_COL_TOKENS}}"
    )
    print(header)
    print("—" * len(header))

    for row in rows:
        execution_id, _run_id, flow_name, status, start_time, end_time, total_cost, total_tokens = row
        print(
            f"{execution_id!s:<{_COL_ID}}  "
            f"{(flow_name or '—'):<{_COL_FLOW}}  "
            f"{(status or '—'):<{_COL_STATUS}}  "
            f"{_format_timestamp(start_time):<{_COL_STARTED}}  "
            f"{_format_duration(start_time, end_time):<{_COL_DURATION}}  "
            f"{_format_cost(total_cost):<{_COL_COST}}  "
            f"{_format_tokens(total_tokens):<{_COL_TOKENS}}"
        )

    print(f"\n{len(rows)} run(s) found.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Show trace summary for a pipeline execution."""
    execution_id = _parse_execution_id(args.execution_id)
    client = _create_client(args)

    # Query run metadata
    try:
        run_result = client.query(_RUN_METADATA_QUERY, parameters={"execution_id": str(execution_id)})
    except Exception as e:
        print(f"Error: Query failed: {e}", file=sys.stderr)
        return 1

    if not run_result.result_rows:
        print(f"No run found for execution_id {execution_id}.", file=sys.stderr)
        return 1

    row = run_result.result_rows[0]
    run_id, flow_name, run_scope, status, start_time, end_time, total_cost, total_tokens, _metadata = row

    print(f"Execution: {execution_id}")
    print(f"  run_id: {run_id}")
    print(f"  flow: {flow_name or '—'}")
    print(f"  scope: {run_scope or '—'}")
    print(f"  status: {status or '—'}")
    print(f"  started: {_format_timestamp(start_time)}")
    print(f"  ended: {_format_timestamp(end_time)}")
    print(f"  duration: {_format_duration(start_time, end_time)}")
    print(f"  cost: {_format_cost(total_cost)}")
    print(f"  tokens: {_format_tokens(total_tokens)}")

    # Query span summary
    try:
        span_result = client.query(_SPAN_SUMMARY_QUERY, parameters={"execution_id": str(execution_id)})
    except Exception as e:
        print(f"\nWarning: Could not query span summary: {e}", file=sys.stderr)
        span_result = None

    if span_result and span_result.result_rows:
        print("\nSpan breakdown:")
        for span_row in span_result.result_rows:
            span_type, cnt, total_dur_ms, span_cost, span_input, span_output = span_row
            dur_str = f"{total_dur_ms:,}ms" if total_dur_ms else "—"
            print(
                f"  {span_type or 'default':<16}  "
                f"count={cnt:<4}  "
                f"duration={dur_str:<12}  "
                f"cost={_format_cost(span_cost):<10}  "
                f"tokens={_format_tokens((span_input or 0) + (span_output or 0))}"
            )

    print(f"\nUse 'ai-trace download {execution_id}' to download full trace.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for trace download and inspection.

    Usage:
        ai-trace list --limit 10 --status completed
        ai-trace show 550e8400-e29b-41d4-a716-446655440000
        ai-trace download 550e8400-e29b-41d4-a716-446655440000 -o ./debug/
        ai-trace download 550e8400-... --documents --children
    """
    parser = argparse.ArgumentParser(
        prog="ai-trace",
        description="Download and inspect pipeline traces from ClickHouse",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download
    dl = subparsers.add_parser(
        "download",
        parents=[_connection_parser],
        help="Download trace and rebuild .trace/ directory",
    )
    dl.add_argument("execution_id", help="Execution UUID")
    dl.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: ./{id[:8]}_trace/)")
    dl.add_argument("--documents", action="store_true", default=False, help="Include document content for replay")
    dl.add_argument("--children", action="store_true", default=False, help="Follow child pipelines")

    # list
    ls = subparsers.add_parser(
        "list",
        parents=[_connection_parser],
        help="List recent pipeline runs",
    )
    ls.add_argument("--limit", type=int, default=DEFAULT_LIST_LIMIT, help="Number of runs (default: 20)")
    ls.add_argument("--status", choices=["running", "completed", "failed"], help="Filter by status")
    ls.add_argument("--flow", type=str, help="Filter by flow name")

    # show
    sh = subparsers.add_parser(
        "show",
        parents=[_connection_parser],
        help="Show trace summary",
    )
    sh.add_argument("execution_id", help="Execution UUID")

    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {"download": _cmd_download, "list": _cmd_list, "show": _cmd_show}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)
