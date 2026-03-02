"""CLI tool for listing, inspecting, and downloading pipeline traces from ClickHouse.

Usage:
    ai-trace list --limit 10 --status completed
    ai-trace show 550e8400-e29b-41d4-a716-446655440000
    ai-trace download 550e8400-e29b-41d4-a716-446655440000 -o ./debug/
    ai-trace download my-run-id --children -o ./debug/
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
FROM pipeline_spans FINAL
WHERE execution_id = {execution_id:UUID}
GROUP BY span_type
ORDER BY total_duration_ms DESC
"""

_RUN_METADATA_QUERY = """
SELECT run_id, flow_name, run_scope, status, start_time, end_time,
       total_cost, total_tokens, metadata_json
FROM pipeline_runs FINAL
WHERE execution_id = {execution_id:UUID}
"""

_DOWNLOAD_SPANS_QUERY = """
SELECT span_id, trace_id, parent_span_id, name, span_type, status,
       start_time, end_time, duration_ms, span_order,
       cost, tokens_input, tokens_output, tokens_cached, llm_model,
       error_message, input_json, output_json, replay_payload,
       attributes_json, events_json, execution_id,
       run_id, flow_name, run_scope,
       input_doc_sha256s, output_doc_sha256s
FROM pipeline_spans FINAL
WHERE execution_id = {execution_id:UUID}
ORDER BY span_order
"""

_DOWNLOAD_CHILDREN_SPANS_QUERY = """
SELECT s.span_id, s.trace_id, s.parent_span_id, s.name, s.span_type, s.status,
       s.start_time, s.end_time, s.duration_ms, s.span_order,
       s.cost, s.tokens_input, s.tokens_output, s.tokens_cached, s.llm_model,
       s.error_message, s.input_json, s.output_json, s.replay_payload,
       s.attributes_json, s.events_json, s.execution_id,
       s.run_id, s.flow_name, s.run_scope,
       s.input_doc_sha256s, s.output_doc_sha256s
FROM pipeline_spans FINAL AS s
WHERE s.execution_id = {execution_id:UUID}
   OR s.execution_id IN (
       SELECT execution_id FROM pipeline_runs FINAL
       WHERE parent_execution_id = {execution_id:UUID}
   )
ORDER BY s.run_id, s.span_order
"""

_RESOLVE_RUN_ID_QUERY = """
SELECT execution_id, run_id
FROM pipeline_runs FINAL
WHERE run_id = {run_id:String}
"""

_DEFAULT_LIST_LIMIT = 20
_EXECUTION_ID_SHORT_LENGTH = 8

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
        "connect_timeout": settings.clickhouse_connect_timeout,
        "send_receive_timeout": settings.clickhouse_send_receive_timeout,
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
        return clickhouse_connect.get_client(**params)  # pyright: ignore[reportUnknownMemberType]
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


def _resolve_identifier(identifier: str, client: Any) -> tuple[UUID, str]:
    """Resolve a CLI identifier to (execution_id, run_id).

    Accepts either a UUID string (used as execution_id directly) or a
    run_id string (looked up via pipeline_runs). Always returns the run_id
    so --children can use it for prefix matching.
    """
    try:
        execution_id = UUID(identifier)
    except ValueError:
        # Not a UUID — treat as run_id, look up in pipeline_runs
        result = client.query(_RESOLVE_RUN_ID_QUERY, parameters={"run_id": identifier})
        if not result.result_rows:
            print(
                f"Error: No run found for run_id '{identifier}'. Use 'ai-trace list' to find available runs.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        execution_id, run_id = result.result_rows[0]
        return execution_id, run_id

    # Got UUID — resolve run_id from pipeline_runs (needed for --children)
    result = client.query(_RUN_METADATA_QUERY, parameters={"execution_id": str(execution_id)})
    if result.result_rows:
        run_id = result.result_rows[0][0]  # first column is run_id
        return execution_id, run_id
    return execution_id, ""


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


def _fetch_and_materialize(client: Any, *, execution_id: UUID, run_id: str, use_children: bool, output_path: Path) -> list[str]:
    """Fetch spans from ClickHouse and materialize into .trace/ directories.

    Returns list of child run_ids (empty if no children found).
    """
    from collections import defaultdict

    from ai_pipeline_core.observability._debug import TraceDebugConfig, TraceMaterializer
    from ai_pipeline_core.observability._span_data import SpanData

    if use_children:
        result = client.query(_DOWNLOAD_CHILDREN_SPANS_QUERY, parameters={"execution_id": str(execution_id)})
    else:
        result = client.query(_DOWNLOAD_SPANS_QUERY, parameters={"execution_id": str(execution_id)})

    rows = result.result_rows
    if not rows:
        identifier_desc = f"execution_id {execution_id} (with children)" if use_children else f"execution_id {execution_id}"
        print(f"No spans found for {identifier_desc}.", file=sys.stderr)
        raise SystemExit(1)

    column_names = result.column_names
    all_spans = [SpanData.from_clickhouse_row(dict(zip(column_names, row, strict=True))) for row in rows]

    # Group spans by run_id for multi-directory materialization
    spans_by_run: dict[str, list[SpanData]] = defaultdict(list)
    for span_data in all_spans:
        spans_by_run[span_data.run_id or ""].append(span_data)

    child_run_ids: list[str] = []
    for span_run_id, spans in spans_by_run.items():
        if span_run_id == run_id or not use_children:
            run_path = output_path
        else:
            child_run_ids.append(span_run_id)
            run_path = output_path / f"child_{span_run_id}"

        materializer = TraceMaterializer(TraceDebugConfig(path=run_path / ".trace"), batch_mode=True)
        real_spans = sorted((s for s in spans if s.span_order > 0), key=lambda s: s.span_order)
        filtered_spans = [s for s in spans if s.span_order == 0]
        # Pass 1: register all spans (status="running") to prevent premature finalization
        for span in real_spans:
            materializer.on_span_start(span)
        # Pass 2: write span data (status→"completed")
        for span in real_spans:
            materializer.add_span(span)
        # Route filtered LLM spans to metrics only (matching live FilesystemBackend behavior)
        for span in filtered_spans:
            materializer.record_filtered_llm_metrics(span)
        materializer.finalize_all()

    return child_run_ids


def _cmd_download(args: argparse.Namespace) -> int:
    """Fetch trace data and reconstruct .trace/ directory from ClickHouse."""
    client = _create_client(args)
    execution_id, run_id = _resolve_identifier(args.identifier, client)

    short_id = str(execution_id)[:_EXECUTION_ID_SHORT_LENGTH]
    output_path = Path(args.output).resolve() if args.output else Path(f"./{short_id}_trace").resolve()

    use_children = args.children
    print(f"Downloading trace {execution_id}" + (f" (run_id: {run_id})" if run_id else ""))
    if use_children:
        print(f"  including children of: {execution_id}")
    print(f"  output: {output_path}")

    try:
        child_run_ids = _fetch_and_materialize(
            client,
            execution_id=execution_id,
            run_id=run_id,
            use_children=use_children,
            output_path=output_path,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: Download failed: {e}", file=sys.stderr)
        logger.debug("Trace download failed", exc_info=True)
        return 1

    # Download documents referenced by replay files (parent + each child)
    found, total = 0, 0
    if not args.no_docs:
        from ai_pipeline_core.observability._download_docs import fetch_trace_documents

        doc_paths = [output_path] + [output_path / f"child_{rid}" for rid in child_run_ids]
        for doc_path in doc_paths:
            try:
                f, t = fetch_trace_documents(client, doc_path)
                found += f
                total += t
            except Exception as e:
                logger.warning("Document download failed for %s: %s", doc_path, e)

    # Print summary
    trace_dir = output_path / ".trace"
    span_dirs = [d for d in trace_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()] if trace_dir.is_dir() else []
    flow_name = _extract_flow_name(output_path)
    print(f"\nDownloaded: {flow_name}")
    print(f"  spans: {len(span_dirs)}")
    if child_run_ids:
        print(f"  children: {len(child_run_ids)}")
    if total > 0:
        doc_msg = f"  documents: {found}/{total}"
        if found < total:
            doc_msg += f" ({total - found} not found in ClickHouse)"
        print(doc_msg)
    print(f"  path: {output_path}")

    return 0


def _extract_flow_name(output_path: Path) -> str:
    """Extract flow name from summary.md in the output directory."""
    summary_path = output_path / ".trace" / "summary.md"
    if summary_path.exists():
        first_line = summary_path.read_text(encoding="utf-8").split("\n", maxsplit=1)[0]
        if first_line.startswith("# "):
            return first_line[2:].strip()
    return "unknown"


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
        ai-trace download my-run-id --children -o ./debug/
    """
    parser = argparse.ArgumentParser(
        prog="ai-trace",
        description="List, inspect, and download pipeline traces from ClickHouse",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download
    dl = subparsers.add_parser(
        "download",
        parents=[_connection_parser],
        help="Fetch trace data and reconstruct .trace/ directory",
    )
    dl.add_argument("identifier", help="Execution UUID or run_id")
    dl.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: ./{id[:8]}_trace/)")
    dl.add_argument("--children", action="store_true", help="Include child pipeline runs (matched by run_id prefix)")
    dl.add_argument("--no-docs", action="store_true", help="Skip downloading documents referenced by replay files")

    # list
    ls = subparsers.add_parser(
        "list",
        parents=[_connection_parser],
        help="List recent pipeline runs",
    )
    ls.add_argument("--limit", type=int, default=_DEFAULT_LIST_LIMIT, help="Number of runs (default: 20)")
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
