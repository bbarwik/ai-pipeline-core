"""Summary generation for trace debugging.

Generates a single _summary.md file that serves both human inspection and LLM debugging.
Combines high-level overview with detailed navigation for comprehensive trace analysis.
"""

from .writer import SpanInfo, TraceState


def generate_summary(trace: TraceState) -> str:
    """Generate unified _summary.md file.

    Single file optimized for both human inspection and LLM debugger context.
    Structure: Overview → Tree → Root Span → LLM Calls → Errors → Navigation.
    """
    lines = [
        f"# Trace Summary: {trace.name}",
        "",
    ]

    # Status and stats
    failed_spans = [s for s in trace.spans.values() if s.status == "failed"]
    status_emoji = "❌" if failed_spans else "✅"
    status_text = f"Failed ({len(failed_spans)} errors)" if failed_spans else "Completed"
    duration_str = _format_duration(trace)

    lines.extend([
        f"**Status**: {status_emoji} {status_text} | "
        f"**Duration**: {duration_str} | "
        f"**Spans**: {len(trace.spans)} | "
        f"**LLM Calls**: {trace.llm_call_count} | "
        f"**Total Tokens**: {trace.total_tokens:,} | "
        f"**Total Cost**: ${trace.total_cost:.4f}",
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
        for span in sorted(trace.spans.values(), key=lambda s: s.start_time):
            lines.append(_format_span_line(span))

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
            "| # | Span | Model | Input→Output | Total | Cost | Path |",
            "|---|------|-------|--------------|-------|------|------|",
        ])

        for i, span in enumerate(llm_spans, 1):
            info = span.llm_info
            if info:
                model = info.get("model", "unknown")
                in_tokens = info.get("input_tokens", 0)
                out_tokens = info.get("output_tokens", 0)
                total_tokens = info.get("total_tokens", 0)
                cost = info.get("cost", 0)
                span_path = span.path.relative_to(trace.path).as_posix()
                lines.append(
                    f"| {i} | {span.name} | {model} | "
                    f"{in_tokens:,}→{out_tokens:,} | {total_tokens:,} | ${cost:.4f} | "
                    f"`{span_path}/` |"
                )

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
        "- `_tree.yaml` has span_id → path mapping and full hierarchy",
        "",
    ])

    return "\n".join(lines)


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
    elif duration < 60:
        return f"{duration:.1f}s"
    elif duration < 3600:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        return f"{hours}h {minutes}m"


def _format_span_line(span: SpanInfo) -> str:
    """Format a single span as a tree line (without prefix)."""
    status_icon = "✅" if span.status == "completed" else "❌" if span.status == "failed" else "⏳"
    duration = (
        f"{span.duration_ms}ms" if span.duration_ms < 1000 else f"{span.duration_ms / 1000:.1f}s"
    )

    llm_suffix = ""
    if span.llm_info:
        model = span.llm_info.get("model", "?")
        tokens = span.llm_info.get("total_tokens", 0)
        llm_suffix = f" [LLM: {model}, {tokens:,} tokens]"

    return f"{span.name} ({duration}) {status_icon}{llm_suffix}"


def _build_tree(trace: TraceState, span_id: str, prefix: str = "") -> list[str]:
    """Build tree representation of span hierarchy (fully recursive)."""
    lines = []
    span = trace.spans.get(span_id)
    if not span:
        return lines

    # Add this span's line
    lines.append(f"{prefix}{_format_span_line(span)}")

    # Process children recursively
    children = span.children
    for i, child_id in enumerate(children):
        is_last = i == len(children) - 1
        child_prefix = prefix + ("└── " if is_last else "├── ")
        continuation_prefix = prefix + ("    " if is_last else "│   ")

        child_span = trace.spans.get(child_id)
        if child_span:
            # Add child line
            lines.append(f"{child_prefix}{_format_span_line(child_span)}")

            # Recursively add all descendants
            for j, grandchild_id in enumerate(child_span.children):
                gc_is_last = j == len(child_span.children) - 1
                gc_prefix = continuation_prefix + ("└── " if gc_is_last else "├── ")
                gc_continuation = continuation_prefix + ("    " if gc_is_last else "│   ")

                # Recursively build subtree for grandchild and all its descendants
                subtree = _build_tree_recursive(trace, grandchild_id, gc_prefix, gc_continuation)
                lines.extend(subtree)

    return lines


def _build_tree_recursive(
    trace: TraceState, span_id: str, prefix: str, continuation: str
) -> list[str]:
    """Recursively build tree for a span and all descendants."""
    lines = []
    span = trace.spans.get(span_id)
    if not span:
        return lines

    # Add this span's line with the given prefix
    lines.append(f"{prefix}{_format_span_line(span)}")

    # Process children
    children = span.children
    for i, child_id in enumerate(children):
        is_last = i == len(children) - 1
        child_prefix = continuation + ("└── " if is_last else "├── ")
        child_continuation = continuation + ("    " if is_last else "│   ")

        # Recurse for all children
        subtree = _build_tree_recursive(trace, child_id, child_prefix, child_continuation)
        lines.extend(subtree)

    return lines
