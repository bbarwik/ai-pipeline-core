"""Static summary generation for trace debugging.

Generates summary.md and costs.md files with compact execution tree,
cost breakdown, and navigation guide. No LLM dependencies — pure text formatting.
"""

from typing import Any

import yaml

from ai_pipeline_core.logging import get_pipeline_logger

from ._config import SpanInfo, TraceState

logger = get_pipeline_logger(__name__)

_FOUR_SPACES = "    "

_REPLAY_FILENAMES: dict[str, str] = {
    "conversation.yaml": "conversation",
    "task.yaml": "task",
    "flow.yaml": "flow",
}


def generate_summary(trace: TraceState) -> str:
    """Generate summary.md with overview, navigation, root span, and execution tree."""
    lines: list[str] = [f"# Trace Summary: {trace.name}", ""]

    # Status line
    failed_spans = [s for s in trace.spans.values() if s.status == "failed"]
    status_text = f"Failed ({len(failed_spans)} errors)" if failed_spans else "Completed"
    duration_str = _format_duration(trace)

    cost_str = f"**Total Cost**: ${trace.total_cost:.4f}"
    if trace.total_expected_cost > 0:
        cost_str += f" (expected: ${trace.total_expected_cost:.4f})"

    lines.extend([
        f"**Status**: {status_text} | **Duration**: {duration_str} | "
        f"**Spans**: {len(trace.spans)} | **LLM Calls**: {trace.llm_call_count} | "
        f"**Total Tokens**: {trace.total_tokens:,} | {cost_str}",
        "",
    ])

    # Navigation (at top for quick reference)
    lines.extend([
        "## Navigation",
        "",
        "- `llm_calls.yaml` — all LLM calls with model, tokens, cost",
        "- `errors.yaml` — failed spans with parent chain (only present when errors exist)",
        "- `costs.md` — cost aggregation by task",
        "- Each span directory: `span.yaml` (metadata), `input.yaml`, `output.yaml`, `events.yaml` (log records)",
        "",
    ])

    # Replay section (only if replay files exist)
    replay_lines = _build_replay_section(trace)
    if replay_lines:
        lines.extend(replay_lines)

    # Root span
    if trace.root_span_id and trace.root_span_id in trace.spans:
        root = trace.spans[trace.root_span_id]
        root_path = root.path.relative_to(trace.path).as_posix()
        lines.extend([
            "## Root Span",
            "",
            f"- **Name**: {root.name}",
            f"- **Type**: {root.span_type}",
            f"- **Duration**: {root.duration_ms}ms",
            f"- **Input**: `{root_path}/input.yaml`",
            f"- **Output**: `{root_path}/output.yaml`",
            "",
        ])

    # Execution tree (compact format)
    lines.extend(["## Execution Tree", ""])
    if trace.root_span_id and trace.root_span_id in trace.spans:
        _build_compact_tree(trace, trace.root_span_id, 0, lines)
    else:
        lines.extend(_format_compact_line(span, 0) for span in sorted(trace.spans.values(), key=lambda s: s.start_time))
    lines.append("")

    return "\n".join(lines)


def generate_costs(trace: TraceState) -> str | None:
    """Generate costs.md content. Returns None if no LLM costs."""
    cost_by_parent = _aggregate_costs_by_parent(trace)
    if not cost_by_parent:
        return None

    headers = ["Span", "Type", "LLM Calls", "Cost", "Status"]
    rows = [[entry["name"], entry["type"], str(entry["llm_calls"]), f"${entry['actual_cost']:.4f}", entry["status"]] for entry in cost_by_parent]

    lines = ["# Cost by Task", "", _format_markdown_table(headers, rows), ""]
    return "\n".join(lines)


def _build_replay_section(trace: TraceState) -> list[str]:
    """Scan span directories for replay YAML files and build a ## Replay section.

    Groups files by payload type with relative paths and example CLI commands.
    Returns empty list if no replay files found.
    """
    found: dict[str, list[tuple[str, dict[str, Any]]]] = {}  # label -> [(relative_path, yaml_data)]

    for span in sorted(trace.spans.values(), key=lambda s: s.order):
        for filename, label in _REPLAY_FILENAMES.items():
            replay_path = span.path / filename
            if not replay_path.exists():
                continue
            relative = replay_path.relative_to(trace.path).as_posix()
            data: dict[str, Any] = {}
            try:
                data = yaml.safe_load(replay_path.read_text(encoding="utf-8")) or {}
            except (yaml.YAMLError, OSError) as e:
                logger.debug("Failed to parse replay file %s: %s", replay_path, e)
            found.setdefault(label, []).append((relative, data))

    if not found:
        return []

    lines = ["## Replay", "", "Re-execute any captured boundary with `ai-replay run <file> --import <your_app>`.", ""]

    for label in ("conversation", "task", "flow"):
        entries = found.get(label)
        if not entries:
            continue
        lines.append(f"**{label.title()}s** ({len(entries)}):")
        for rel_path, data in entries:
            detail = _replay_entry_detail(label, data)
            lines.append(f"- `{rel_path}`{detail}")
        lines.append("")

    # Example commands
    first_path = next(iter(next(iter(found.values()))))[0]
    lines.extend([
        "```bash",
        f"ai-replay show {first_path}",
        f"ai-replay run {first_path} --import my_app.tasks",
        f"ai-replay run {first_path} --set model=grok-4.1-fast --import my_app.tasks",
        "```",
        "",
    ])

    return lines


def _replay_entry_detail(label: str, data: dict[str, Any]) -> str:
    """Format a short detail suffix for a replay entry line."""
    if label == "conversation":
        model = data.get("model", "")
        original = data.get("original", {})
        cost = original.get("cost")
        suffix = f" — {model}" if model else ""
        if cost is not None:
            suffix += f" (${cost:.4f})"
        return suffix
    if label == "task":
        fn = data.get("function_path", "")
        name = fn.rsplit(":", 1)[-1] if ":" in fn else fn
        return f" — {name}" if name else ""
    if label == "flow":
        fn = data.get("function_path", "")
        name = fn.rsplit(":", 1)[-1] if ":" in fn else fn
        doc_count = len(data.get("documents", []))
        suffix = f" — {name}" if name else ""
        if doc_count:
            suffix += f" ({doc_count} docs)"
        return suffix
    return ""


def _format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a markdown table with fixed-width columns."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        padded = [cells[i].ljust(col_widths[i]) for i in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(r) for r in rows)])


def _build_compact_tree(trace: TraceState, span_id: str, depth: int, lines: list[str]) -> None:
    """Build compact execution tree recursively."""
    span = trace.spans.get(span_id)
    if not span or span.span_id in trace.merged_wrapper_ids:
        return
    lines.append(_format_compact_line(span, depth))
    for child_id in span.children:
        _build_compact_tree(trace, child_id, depth + 1, lines)


def _format_compact_line(span: SpanInfo, depth: int) -> str:
    """Format one span line.

    Format: {indent}{id}_{name}    {duration}[    ERROR|WAIT][    {in} IN[ {cache} CACHE] {out} OUT ${cost}]
    """
    indent = _FOUR_SPACES * depth
    width = 3 if span.order < 1000 else len(str(span.order))
    span_id_str = f"{span.order:0{width}d}"

    # Duration: integer for >=10s, one decimal for <10s, 0s for sub-second
    secs = span.duration_ms / 1000
    if secs < 0.5:
        dur = "0s"
    elif secs < 10:
        dur = f"{secs:.1f}s"
    else:
        dur = f"{round(secs)}s"

    parts = [f"{indent}{span_id_str}_{span.name}    {dur}"]

    # Status (only for non-success — implicit OK)
    if span.status == "failed":
        parts.append("    ERROR")
    elif span.status == "running":
        parts.append("    WAIT")

    # LLM token/cost info (only for spans with llm_info, skip on ERROR)
    if span.llm_info and span.status != "failed":
        in_tok = _format_token_count(span.llm_info.get("input_tokens", 0))
        out_tok = _format_token_count(span.llm_info.get("output_tokens", 0))
        cost = span.llm_info.get("cost", 0.0)
        cached = span.llm_info.get("cached_tokens", 0)
        cache_part = f" {_format_token_count(cached)} CACHE" if cached else ""
        parts.append(f"    {in_tok} IN{cache_part} {out_tok} OUT ${cost:.3f}")

    return "".join(parts)


def _format_token_count(n: int) -> str:
    """Format token count: raw if <1000, K-suffix if >=1000."""
    if n >= 1000:
        return f"{round(n / 1000)}K"
    return str(n)


def _format_duration(trace: TraceState) -> str:
    """Format trace duration as human-readable string."""
    if not trace.spans:
        return "unknown"

    spans = trace.spans.values()
    end_times = [s.end_time for s in spans if s.end_time]
    if not end_times:
        return "running..."

    secs = (max(end_times) - min(s.start_time for s in spans)).total_seconds()
    if secs < 1:
        return f"{int(secs * 1000)}ms"
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m {int(secs % 60)}s"
    return f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"


def _aggregate_costs_by_parent(trace: TraceState) -> list[dict[str, Any]]:
    """Aggregate LLM costs by parent task/flow span."""
    parent_costs: dict[str, dict[str, Any]] = {}

    for span in trace.spans.values():
        if not span.llm_info:
            continue
        cost = span.llm_info.get("cost", 0.0)
        if not cost:
            continue

        parent_id = span.parent_id
        if not parent_id or parent_id not in trace.spans:
            continue
        parent = trace.spans[parent_id]

        if parent_id not in parent_costs:
            width = 3 if parent.order < 1000 else len(str(parent.order))
            parent_costs[parent_id] = {
                "name": f"{parent.order:0{width}d}_{parent.name}",
                "type": parent.span_type,
                "actual_cost": 0.0,
                "status": parent.status,
                "llm_calls": 0,
            }
        parent_costs[parent_id]["actual_cost"] += cost
        parent_costs[parent_id]["llm_calls"] += 1

    return sorted(parent_costs.values(), key=lambda x: x["actual_cost"], reverse=True)
