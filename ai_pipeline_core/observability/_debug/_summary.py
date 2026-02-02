"""Static summary generation for trace debugging.

Generates _summary.md files with execution tree, LLM calls, cost breakdown,
and navigation guide. No LLM dependencies — pure text formatting.

For LLM-powered auto-summary, see _auto_summary.py.
"""

from typing import Any

from ._types import SpanInfo, TraceState


def generate_summary(trace: TraceState) -> str:  # noqa: PLR0912, PLR0914, PLR0915
    """Generate unified _summary.md file.

    Single file optimized for both human inspection and LLM debugger context.
    Structure: Overview -> Tree -> Root Span -> LLM Calls -> Cost by Task -> Errors -> Navigation.
    Cost by Task table includes expected cost comparison with OVER/OK status indicators.
    """
    lines = [
        f"# Trace Summary: {trace.name}",
        "",
    ]

    # Status and stats
    failed_spans = [s for s in trace.spans.values() if s.status == "failed"]
    status_emoji = "\u274c" if failed_spans else "\u2705"
    status_text = f"Failed ({len(failed_spans)} errors)" if failed_spans else "Completed"
    duration_str = _format_duration(trace)

    cost_str = f"**Total Cost**: ${trace.total_cost:.4f}"
    if trace.total_expected_cost > 0:
        cost_str += f" (expected: ${trace.total_expected_cost:.4f})"

    lines.extend([
        f"**Status**: {status_emoji} {status_text} | "
        f"**Duration**: {duration_str} | "
        f"**Spans**: {len(trace.spans)} | "
        f"**LLM Calls**: {trace.llm_call_count} | "
        f"**Total Tokens**: {trace.total_tokens:,} | "
        f"{cost_str}",
        "",
    ])

    # Execution tree
    lines.extend([
        "## Execution Tree",
        "",
        "```",
    ])

    if trace.root_span_id and trace.root_span_id in trace.spans:
        tree_lines = _build_tree(trace, trace.root_span_id, "")
        lines.extend(tree_lines)
    else:
        # Fallback: list all spans
        lines.extend(_format_span_line(span) for span in sorted(trace.spans.values(), key=lambda s: s.start_time))

    lines.extend([
        "```",
        "",
    ])

    # Root span details
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

    # LLM calls table with path column
    llm_spans = [s for s in trace.spans.values() if s.llm_info]
    if llm_spans:
        llm_spans.sort(key=lambda s: s.llm_info.get("cost", 0) if s.llm_info else 0, reverse=True)

        lines.extend([
            "## LLM Calls (by cost)",
            "",
            "| # | Span | Purpose | Model | Input\u2192Output | Total | Cost | Expected | Path |",
            "|---|------|---------|-------|--------------|-------|------|----------|------|",
        ])

        for i, span in enumerate(llm_spans, 1):
            info = span.llm_info
            if info:
                model = info.get("model", "unknown")
                purpose = info.get("purpose", "")
                in_tokens = info.get("input_tokens", 0)
                out_tokens = info.get("output_tokens", 0)
                total_tokens = info.get("total_tokens", 0)
                cost = info.get("cost", 0)
                expected = info.get("expected_cost")
                expected_str = f"${expected:.4f}" if expected else ""
                span_path = span.path.relative_to(trace.path).as_posix()
                lines.append(
                    f"| {i} | {span.name} | {purpose} | {model} | "
                    f"{in_tokens:,}\u2192{out_tokens:,} | {total_tokens:,} | ${cost:.4f} | "
                    f"{expected_str} | `{span_path}/` |"
                )

        lines.append("")

    # Cost aggregation by parent task/flow
    cost_by_parent = _aggregate_costs_by_parent(trace)
    if cost_by_parent:
        lines.extend([
            "## Cost by Task",
            "",
            "| Name | Type | LLM Calls | Cost | Expected | Status |",
            "|------|------|-----------|------|----------|--------|",
        ])
        for entry in cost_by_parent:
            expected_str = f"${entry['expected_cost']:.4f}" if entry["expected_cost"] else ""
            status = ""
            if entry["expected_cost"] and entry["actual_cost"] > 0:
                ratio = entry["actual_cost"] / entry["expected_cost"]
                status = "OVER" if ratio > 1.1 else "OK"
            lines.append(f"| {entry['name']} | {entry['type']} | {entry['llm_calls']} | ${entry['actual_cost']:.4f} | {expected_str} | {status} |")
        lines.append("")

    # Errors
    if failed_spans:
        lines.extend([
            "## Errors",
            "",
        ])
        for span in failed_spans:
            span_path = span.path.relative_to(trace.path).as_posix()
            lines.append(f"- **{span.name}**: `{span_path}/_span.yaml`")
        lines.append("")
    else:
        lines.extend([
            "## Errors",
            "",
            "None - all spans completed successfully.",
            "",
        ])

    # Navigation guide
    lines.extend([
        "## Navigation",
        "",
        "- Each span directory contains `_span.yaml` (metadata), `input.yaml`, `output.yaml`",
        "- LLM span inputs contain the full message list",
        "- `_tree.yaml` has span_id \u2192 path mapping and full hierarchy",
        "",
    ])

    return "\n".join(lines)


def _aggregate_costs_by_parent(trace: TraceState) -> list[dict[str, Any]]:
    """Aggregate LLM costs by parent task/flow span."""
    parent_costs: dict[str, dict[str, Any]] = {}

    for span in trace.spans.values():
        if not span.llm_info:
            continue
        cost = span.llm_info.get("cost", 0.0)
        if not cost:
            continue

        # Find parent (task or flow span)
        parent_id = span.parent_id
        if not parent_id or parent_id not in trace.spans:
            continue
        parent = trace.spans[parent_id]

        if parent_id not in parent_costs:
            run_type = "unknown"
            if parent.prefect_info:
                run_type = parent.prefect_info.get("run_type", "unknown")
            parent_costs[parent_id] = {
                "name": parent.name,
                "type": run_type,
                "actual_cost": 0.0,
                "expected_cost": parent.expected_cost,
                "llm_calls": 0,
            }
        parent_costs[parent_id]["actual_cost"] += cost
        parent_costs[parent_id]["llm_calls"] += 1

    # Sort by cost descending
    return sorted(parent_costs.values(), key=lambda x: x["actual_cost"], reverse=True)


def _format_duration(trace: TraceState) -> str:
    """Format trace duration as human-readable string."""
    # Calculate from spans if we have them
    if not trace.spans:
        return "unknown"

    spans_list = list(trace.spans.values())
    start = min(s.start_time for s in spans_list)
    end_times = [s.end_time for s in spans_list if s.end_time]

    if not end_times:
        return "running..."

    end = max(end_times)
    duration = (end - start).total_seconds()

    if duration < 1:
        return f"{int(duration * 1000)}ms"
    if duration < 60:
        return f"{duration:.1f}s"
    if duration < 3600:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s"
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    return f"{hours}h {minutes}m"


def _format_span_line(span: SpanInfo) -> str:
    """Format a single span as a tree line (without prefix)."""
    if span.status == "completed":
        status_icon = "\u2705"
    elif span.status == "failed":
        status_icon = "\u274c"
    else:
        status_icon = "\u23f3"
    duration = f"{span.duration_ms}ms" if span.duration_ms < 1000 else f"{span.duration_ms / 1000:.1f}s"

    # Description suffix for task/flow spans
    desc_suffix = ""
    if span.description and span.span_type != "llm":
        desc_suffix = f" -- {span.description}"

    # LLM suffix: show purpose (if available) alongside model, plus cost
    llm_suffix = ""
    if span.llm_info:
        model = span.llm_info.get("model", "?")
        tokens = span.llm_info.get("total_tokens", 0)
        cost = span.llm_info.get("cost", 0)
        purpose = span.llm_info.get("purpose")

        purpose_part = f"{purpose} | " if purpose else ""
        cost_part = f", ${cost:.4f}" if cost else ""
        llm_suffix = f" [LLM: {purpose_part}{model}, {tokens:,} tokens{cost_part}]"

    return f"{span.name} ({duration}) {status_icon}{desc_suffix}{llm_suffix}"


def _build_tree(trace: TraceState, span_id: str, prefix: str = "") -> list[str]:
    """Build tree representation of span hierarchy (fully recursive)."""
    lines: list[str] = []
    span = trace.spans.get(span_id)
    if not span:
        return lines

    # Add this span's line
    lines.append(f"{prefix}{_format_span_line(span)}")

    # Process children recursively
    children = span.children
    for i, child_id in enumerate(children):
        is_last = i == len(children) - 1
        child_prefix = prefix + ("\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 ")
        continuation_prefix = prefix + ("    " if is_last else "\u2502   ")

        child_span = trace.spans.get(child_id)
        if child_span:
            # Add child line
            lines.append(f"{child_prefix}{_format_span_line(child_span)}")

            # Recursively add all descendants
            for j, grandchild_id in enumerate(child_span.children):
                gc_is_last = j == len(child_span.children) - 1
                gc_connector = "\u2514\u2500\u2500 " if gc_is_last else "\u251c\u2500\u2500 "
                gc_prefix = continuation_prefix + gc_connector
                gc_continuation = continuation_prefix + ("    " if gc_is_last else "\u2502   ")

                # Recursively build subtree for grandchild and all its descendants
                subtree = _build_tree_recursive(trace, grandchild_id, gc_prefix, gc_continuation)
                lines.extend(subtree)

    return lines


def _build_tree_recursive(trace: TraceState, span_id: str, prefix: str, continuation: str) -> list[str]:
    """Recursively build tree for a span and all descendants."""
    lines: list[str] = []
    span = trace.spans.get(span_id)
    if not span:
        return lines

    # Add this span's line with the given prefix
    lines.append(f"{prefix}{_format_span_line(span)}")

    # Process children
    children = span.children
    for i, child_id in enumerate(children):
        is_last = i == len(children) - 1
        child_prefix = continuation + ("\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 ")
        child_continuation = continuation + ("    " if is_last else "\u2502   ")

        # Recurse for all children
        subtree = _build_tree_recursive(trace, child_id, child_prefix, child_continuation)
        lines.extend(subtree)

    return lines
