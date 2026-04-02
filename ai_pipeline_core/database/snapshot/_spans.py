"""Helpers for rendering span-era deployment summaries and trees."""

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

from ai_pipeline_core.database._json_helpers import parse_json_object
from ai_pipeline_core.database._types import (
    TOKENS_CACHE_READ_KEY,
    TOKENS_INPUT_KEY,
    TOKENS_OUTPUT_KEY,
    TOKENS_REASONING_KEY,
    CostTotals,
    SpanKind,
    SpanRecord,
    SpanStatus,
    aggregate_cost_totals,
)
from ai_pipeline_core.database.filesystem._paths import run_directory_name, span_filename

__all__ = [
    "SpanTreeView",
    "build_span_tree_view",
    "format_span_overview_lines",
    "format_span_tree_lines",
    "generate_costs_from_tree",
    "generate_summary_from_tree",
]

_FOUR_SPACES = "    "
MAX_TREE_DEPTH = 100

MODEL_KEY = "model"
PURPOSE_KEY = "purpose"
CACHE_HIT_KEY = "cache_hit"
STEP_KEY = "step"
TOTAL_STEPS_KEY = "total_steps"
ROUND_INDEX_KEY = "round_index"
TOOL_NAME_KEY = "tool_name"
FLOW_PLAN_KEY = "flow_plan"

UNKNOWN_MODEL_LABEL = "(unknown)"

_SPAN_KIND_PRIORITY: dict[str, int] = {
    SpanKind.DEPLOYMENT: 0,
    SpanKind.FLOW: 1,
    SpanKind.TASK: 2,
    SpanKind.ATTEMPT: 3,
    SpanKind.OPERATION: 4,
    SpanKind.CONVERSATION: 5,
    SpanKind.LLM_ROUND: 6,
    SpanKind.TOOL_CALL: 7,
}


@dataclass(frozen=True, slots=True)
class _TokenTotals:
    """Aggregated token counts for descendant LLM rounds."""

    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tokens_reasoning: int = 0


@dataclass(frozen=True, slots=True)
class SpanTreeView:
    """Precomputed indexes and totals for one span deployment tree."""

    root_span: SpanRecord
    spans_by_id: dict[UUID, SpanRecord]
    children_map: dict[UUID | None, list[UUID]]
    meta_by_id: dict[UUID, dict[str, Any]]
    metrics_by_id: dict[UUID, dict[str, Any]]
    counts_by_kind: Counter[str]
    document_shas: set[str]
    descendant_costs: dict[UUID, float]
    descendant_tokens: dict[UUID, _TokenTotals]
    totals: CostTotals


def _detail_int(payload: dict[str, Any], key: str, *, context: str, field_name: str) -> int:
    value = payload.get(key, 0)
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        msg = f"{context} {field_name}[{key!r}] must be an integer value. Persist an integer before rendering summaries."
        raise ValueError(msg) from exc


def _detail_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if isinstance(value, str):
        return value
    return ""


def _detail_bool(payload: dict[str, Any], key: str) -> bool:
    return payload.get(key) is True


def _child_sort_key(span_id: UUID, spans_by_id: dict[UUID, SpanRecord]) -> tuple[Any, int, int, str]:
    span = spans_by_id[span_id]
    return (
        span.started_at,
        span.sequence_no,
        _SPAN_KIND_PRIORITY.get(span.kind, len(_SPAN_KIND_PRIORITY)),
        str(span.span_id),
    )


def _build_children_map(tree: list[SpanRecord]) -> dict[UUID | None, list[UUID]]:
    children_map: dict[UUID | None, list[UUID]] = {}
    spans_by_id = {span.span_id: span for span in tree}
    for span in tree:
        children_map.setdefault(span.parent_span_id, []).append(span.span_id)
    for parent_id, child_ids in children_map.items():
        children_map[parent_id] = sorted(child_ids, key=lambda child_id: _child_sort_key(child_id, spans_by_id))
    return children_map


def _select_root_span(tree: list[SpanRecord], root_deployment_id: UUID) -> SpanRecord | None:
    deployment_matches = [span for span in tree if span.kind == SpanKind.DEPLOYMENT and span.deployment_id == root_deployment_id]
    if deployment_matches:
        return min(deployment_matches, key=lambda span: (span.started_at, span.sequence_no, str(span.span_id)))

    top_level_matches = [span for span in tree if span.kind == SpanKind.DEPLOYMENT and span.parent_span_id is None]
    if top_level_matches:
        return min(top_level_matches, key=lambda span: (span.started_at, span.sequence_no, str(span.span_id)))

    deployment_spans = [span for span in tree if span.kind == SpanKind.DEPLOYMENT]
    if deployment_spans:
        return min(deployment_spans, key=lambda span: (span.started_at, span.sequence_no, str(span.span_id)))
    return None


def _compute_descendant_costs(
    *,
    spans_by_id: dict[UUID, SpanRecord],
    children_map: dict[UUID | None, list[UUID]],
    metrics_by_id: dict[UUID, dict[str, Any]],
) -> tuple[dict[UUID, float], dict[UUID, _TokenTotals]]:
    descendant_costs: dict[UUID, float] = {}
    descendant_tokens: dict[UUID, _TokenTotals] = {}
    visiting: set[UUID] = set()

    def _visit(span_id: UUID) -> tuple[float, _TokenTotals]:
        if span_id in descendant_costs:
            return descendant_costs[span_id], descendant_tokens[span_id]
        if span_id in visiting:
            return 0.0, _TokenTotals()

        visiting.add(span_id)
        total_cost = 0.0
        ti, to, tc, tr = 0, 0, 0, 0
        for child_id in children_map.get(span_id, []):
            child = spans_by_id[child_id]
            total_cost += child.cost_usd
            if child.kind == SpanKind.LLM_ROUND:
                metrics = metrics_by_id[child_id]
                context = f"Span {child_id}"
                ti += _detail_int(metrics, TOKENS_INPUT_KEY, context=context, field_name="metrics_json")
                to += _detail_int(metrics, TOKENS_OUTPUT_KEY, context=context, field_name="metrics_json")
                tc += _detail_int(metrics, TOKENS_CACHE_READ_KEY, context=context, field_name="metrics_json")
                tr += _detail_int(metrics, TOKENS_REASONING_KEY, context=context, field_name="metrics_json")
            child_cost, child_tokens = _visit(child_id)
            total_cost += child_cost
            ti += child_tokens.tokens_input
            to += child_tokens.tokens_output
            tc += child_tokens.tokens_cache_read
            tr += child_tokens.tokens_reasoning
        descendant_costs[span_id] = total_cost
        descendant_tokens[span_id] = _TokenTotals(tokens_input=ti, tokens_output=to, tokens_cache_read=tc, tokens_reasoning=tr)
        visiting.remove(span_id)
        return total_cost, descendant_tokens[span_id]

    for span_id in spans_by_id:
        _visit(span_id)
    return descendant_costs, descendant_tokens


def _sum_span_totals(tree: list[SpanRecord]) -> CostTotals:
    return aggregate_cost_totals((span.kind, span.cost_usd, span.metrics_json, f"Span {span.span_id}") for span in tree)


def _collect_document_shas(tree: list[SpanRecord]) -> set[str]:
    shas: set[str] = set()
    for span in tree:
        shas.update(span.input_document_shas)
        shas.update(span.output_document_shas)
    return shas


def build_span_tree_view(tree: list[SpanRecord], root_deployment_id: UUID) -> SpanTreeView | None:
    """Build reusable indexes and totals for a span deployment tree."""
    if not tree:
        return None

    root_span = _select_root_span(tree, root_deployment_id)
    if root_span is None:
        return None

    spans_by_id = {span.span_id: span for span in tree}
    meta_by_id = {span.span_id: parse_json_object(span.meta_json, context=f"Span {span.span_id}", field_name="meta_json") for span in tree}
    metrics_by_id = {span.span_id: parse_json_object(span.metrics_json, context=f"Span {span.span_id}", field_name="metrics_json") for span in tree}
    children_map = _build_children_map(tree)
    descendant_costs, descendant_tokens = _compute_descendant_costs(
        spans_by_id=spans_by_id,
        children_map=children_map,
        metrics_by_id=metrics_by_id,
    )
    return SpanTreeView(
        root_span=root_span,
        spans_by_id=spans_by_id,
        children_map=children_map,
        meta_by_id=meta_by_id,
        metrics_by_id=metrics_by_id,
        counts_by_kind=Counter(span.kind for span in tree),
        document_shas=_collect_document_shas(tree),
        descendant_costs=descendant_costs,
        descendant_tokens=descendant_tokens,
        totals=_sum_span_totals(tree),
    )


def _format_timedelta(delta: timedelta) -> str:
    secs = delta.total_seconds()
    if secs < 1:
        return f"{int(secs * 1000)}ms"
    if secs < 60:
        return f"{secs:.1f}s" if secs < 10 else f"{secs:.0f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m {int(secs % 60)}s"
    return f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"


def _format_duration(span: SpanRecord) -> str:
    if span.ended_at is None:
        return "running..." if span.status == SpanStatus.RUNNING else "-"
    return _format_timedelta(span.ended_at - span.started_at)


def _format_token_count(count: int) -> str:
    if count >= 1000:
        return f"{round(count / 1000)}K"
    return str(count)


def _format_token_parts(span: SpanRecord, metrics: dict[str, Any], view: SpanTreeView) -> str:
    if span.kind not in {SpanKind.CONVERSATION, SpanKind.LLM_ROUND}:
        return ""
    if span.kind == SpanKind.CONVERSATION:
        dt = view.descendant_tokens.get(span.span_id, _TokenTotals())
        tokens_input, tokens_output = dt.tokens_input, dt.tokens_output
        tokens_cache_read, tokens_reasoning = dt.tokens_cache_read, dt.tokens_reasoning
    else:
        context = f"Span {span.span_id}"
        tokens_input = _detail_int(metrics, TOKENS_INPUT_KEY, context=context, field_name="metrics_json")
        tokens_output = _detail_int(metrics, TOKENS_OUTPUT_KEY, context=context, field_name="metrics_json")
        tokens_cache_read = _detail_int(metrics, TOKENS_CACHE_READ_KEY, context=context, field_name="metrics_json")
        tokens_reasoning = _detail_int(metrics, TOKENS_REASONING_KEY, context=context, field_name="metrics_json")
    parts = [f"{_format_token_count(tokens_input)} in"]
    if tokens_cache_read > 0:
        parts.append(f"{_format_token_count(tokens_cache_read)} cache")
    parts.append(f"{_format_token_count(tokens_output)} out")
    if tokens_reasoning > 0:
        parts.append(f"{_format_token_count(tokens_reasoning)} reasoning")
    return " / ".join(parts)


def _cache_hit_suffix(meta: dict[str, Any]) -> str:
    if _detail_bool(meta, CACHE_HIT_KEY):
        return " cache-hit"
    return ""


def _conversation_label(span: SpanRecord, meta: dict[str, Any]) -> str:
    purpose = _detail_str(meta, PURPOSE_KEY)
    if purpose:
        return purpose
    if span.description:
        return span.description
    return span.name


def _flow_step_label(span: SpanRecord, meta: dict[str, Any]) -> str:
    context = f"Span {span.span_id}"
    step = _detail_int(meta, STEP_KEY, context=context, field_name="meta_json")
    total_steps = _detail_int(meta, TOTAL_STEPS_KEY, context=context, field_name="meta_json")
    if step > 0 and total_steps > 0:
        return f"[{step}/{total_steps}]"
    if step > 0:
        return f"[{step}]"
    if span.sequence_no > 0:
        return f"[{span.sequence_no}]"
    return ""


def _round_label(span: SpanRecord, meta: dict[str, Any]) -> str:
    context = f"Span {span.span_id}"
    round_index = _detail_int(meta, ROUND_INDEX_KEY, context=context, field_name="meta_json")
    if round_index > 0:
        return f"[{round_index}]"
    if span.sequence_no > 0:
        return f"[{span.sequence_no}]"
    return ""


def _span_local_filename(span: SpanRecord, view: SpanTreeView) -> str | None:
    if span.kind in {SpanKind.LLM_ROUND, SpanKind.TOOL_CALL, SpanKind.ATTEMPT}:
        return None
    if span.kind == SpanKind.DEPLOYMENT and span.parent_span_id is None:
        return "deployment.json"
    if span.parent_span_id is None:
        return None
    siblings = [
        sibling_id
        for sibling_id in view.children_map.get(span.parent_span_id, [])
        if view.spans_by_id[sibling_id].kind not in {SpanKind.LLM_ROUND, SpanKind.TOOL_CALL, SpanKind.ATTEMPT}
    ]
    sibling_index = siblings.index(span.span_id) + 1
    return span_filename(span.kind, span.name, span.span_id, sibling_index)


def _format_attempt_line(span: SpanRecord, meta: dict[str, Any], view: SpanTreeView, *, indent: str, status: str, duration: str) -> str:
    context = f"Span {span.span_id}"
    attempt_num = _detail_int(meta, "attempt", context=context, field_name="meta_json")
    max_attempts = _detail_int(meta, "max_attempts", context=context, field_name="meta_json")
    label = f"{attempt_num}/{max_attempts}" if max_attempts > 0 else str(attempt_num)
    rendered = f"{indent}attempt: {label} {status} {duration}"
    cost = span.cost_usd + view.descendant_costs.get(span.span_id, 0.0)
    if cost > 0:
        rendered += f" ${cost:.4f}"
    return rendered


def _format_tree_line(span: SpanRecord, view: SpanTreeView, *, depth: int, include_filenames: bool) -> str:
    indent = _FOUR_SPACES * depth
    meta = view.meta_by_id[span.span_id]
    metrics = view.metrics_by_id[span.span_id]
    status = str(span.status)
    duration = _format_duration(span)
    cache_suffix = _cache_hit_suffix(meta)

    if span.kind == SpanKind.CONVERSATION:
        model = _detail_str(meta, MODEL_KEY) or UNKNOWN_MODEL_LABEL
        tokens = _format_token_parts(span, metrics, view)
        cost = span.cost_usd + view.descendant_costs.get(span.span_id, 0.0)
        chain_suffix = f" (continues {str(span.previous_conversation_id)[:8]}…)" if span.previous_conversation_id else ""
        line = f"{indent}conversation: {_conversation_label(span, meta)} {duration} {model}{chain_suffix}"
        if tokens:
            line += f" {tokens}"
        if cost > 0:
            line += f" ${cost:.4f}"
        rendered = f"{line}{cache_suffix}"
        filename = _span_local_filename(span, view) if include_filenames else None
        return f"{rendered}  -> {filename}" if filename is not None else rendered

    if span.kind == SpanKind.LLM_ROUND:
        model = _detail_str(meta, MODEL_KEY) or UNKNOWN_MODEL_LABEL
        tokens = _format_token_parts(span, metrics, view)
        line = f"{indent}llm_round{_round_label(span, meta)}: {model} {duration}"
        if tokens:
            line += f" {tokens}"
        if span.cost_usd > 0:
            line += f" ${span.cost_usd:.4f}"
        rendered = f"{line}{cache_suffix}"
        filename = _span_local_filename(span, view) if include_filenames else None
        return f"{rendered}  -> {filename}" if filename is not None else rendered

    if span.kind == SpanKind.TOOL_CALL:
        tool_name = _detail_str(meta, TOOL_NAME_KEY) or span.name
        return f"{indent}tool_call{_round_label(span, meta)}: {tool_name} {status} {duration}{cache_suffix}"

    if span.kind == SpanKind.FLOW:
        step_label = _flow_step_label(span, meta)
        rendered = f"{indent}flow{step_label}: {span.name} {status} {duration}{cache_suffix}"
        filename = _span_local_filename(span, view) if include_filenames else None
        return f"{rendered}  -> {filename}" if filename is not None else rendered

    if span.kind == SpanKind.ATTEMPT:
        return _format_attempt_line(span, meta, view, indent=indent, status=status, duration=duration)

    rendered = f"{indent}{span.kind}: {span.name} {status} {duration}{cache_suffix}"
    filename = _span_local_filename(span, view) if include_filenames else None
    return f"{rendered}  -> {filename}" if filename is not None else rendered


def format_span_tree_lines(view: SpanTreeView, *, include_filenames: bool = False) -> list[str]:
    """Render the full span execution tree as indented plain-text lines."""
    lines: list[str] = []
    visited: set[UUID] = set()

    def append_tree_lines(span_id: UUID, depth: int) -> None:
        if span_id in visited:
            lines.append(f"{_FOUR_SPACES * depth}[cycle detected while rendering execution tree]")
            return
        if depth > MAX_TREE_DEPTH:
            lines.append(f"{_FOUR_SPACES * depth}[execution tree depth limit reached]")
            return

        visited.add(span_id)
        span = view.spans_by_id[span_id]

        if span.kind == SpanKind.ATTEMPT:
            attempt_siblings = [c for c in view.children_map.get(span.parent_span_id, []) if view.spans_by_id[c].kind == SpanKind.ATTEMPT]
            if len(attempt_siblings) == 1 and span.status != SpanStatus.FAILED:
                # Single non-failed attempt — skip the ATTEMPT wrapper line and render children at same depth.
                # Failed attempts are always shown so timing and error context remain visible for debugging.
                for child_id in view.children_map.get(span_id, []):
                    append_tree_lines(child_id, depth)
                visited.remove(span_id)
                return

        lines.append(_format_tree_line(span, view, depth=depth, include_filenames=include_filenames))
        for child_id in view.children_map.get(span_id, []):
            append_tree_lines(child_id, depth + 1)
        visited.remove(span_id)

    append_tree_lines(view.root_span.span_id, 0)
    return lines


def format_span_overview_lines(view: SpanTreeView) -> list[str]:
    """Render a compact plain-text deployment overview for ai-trace show."""
    total_tokens = view.totals.tokens_input + view.totals.tokens_output + view.totals.tokens_cache_read + view.totals.tokens_reasoning
    return [
        f"Deployment {view.root_span.deployment_name or view.root_span.name} / {view.root_span.run_id}",
        (
            f"Status: {view.root_span.status}  Duration: {_format_duration(view.root_span)}  "
            f"Spans: {sum(view.counts_by_kind.values())}  Documents: {len(view.document_shas)}"
        ),
        (
            f"Flows: {view.counts_by_kind.get(SpanKind.FLOW, 0)}  "
            f"Tasks: {view.counts_by_kind.get(SpanKind.TASK, 0)}  "
            f"Attempts: {view.counts_by_kind.get(SpanKind.ATTEMPT, 0)}  "
            f"Operations: {view.counts_by_kind.get(SpanKind.OPERATION, 0)}  "
            f"Conversations: {view.counts_by_kind.get(SpanKind.CONVERSATION, 0)}  "
            f"LLM Rounds: {view.counts_by_kind.get(SpanKind.LLM_ROUND, 0)}  "
            f"Tool Calls: {view.counts_by_kind.get(SpanKind.TOOL_CALL, 0)}"
        ),
        f"Total Tokens: {total_tokens:,}  Total Cost: ${view.totals.cost_usd:.4f}",
    ]


def _build_failures_lines(view: SpanTreeView) -> list[str]:
    failed_spans = sorted(
        (span for span in view.spans_by_id.values() if span.status == SpanStatus.FAILED and span.kind != SpanKind.ATTEMPT),
        key=lambda span: (span.started_at, span.sequence_no, str(span.span_id)),
    )
    if not failed_spans:
        return []

    lines = ["## Failures", ""]
    for span in failed_spans:
        error_parts = [part for part in (span.error_type, span.error_message) if part]
        error_text = ": ".join(error_parts) if error_parts else "(no error recorded)"
        lines.append(f"- `{span.kind}: {span.name}`")
        lines.append(f"  `{error_text}`")
    lines.append("")
    return lines


def _build_flow_plan_lines(view: SpanTreeView) -> list[str]:
    meta = view.meta_by_id[view.root_span.span_id]
    flow_plan = meta.get(FLOW_PLAN_KEY)
    if not isinstance(flow_plan, list) or not flow_plan:
        return []

    flows_by_name: dict[str, list[SpanRecord]] = {}
    for span in view.spans_by_id.values():
        if span.kind != SpanKind.FLOW:
            continue
        flows_by_name.setdefault(span.name, []).append(span)

    for spans in flows_by_name.values():
        spans.sort(key=lambda span: (span.sequence_no, span.started_at, str(span.span_id)))

    lines = ["## Flow Plan", "", "| Step | Flow | Status | Duration | Cost |", "|---|---|---|---:|---:|"]
    planned_flow_span_ids: set[UUID] = set()
    for index, plan_entry in enumerate(flow_plan, start=1):
        if not isinstance(plan_entry, dict):
            msg = (
                f"Span {view.root_span.span_id} meta_json['flow_plan'] must contain only objects. "
                "Persist flow_plan as a JSON array of objects with at least a name field."
            )
            raise TypeError(msg)
        name = plan_entry.get("name")
        if not isinstance(name, str) or not name:
            name = f"Flow {index}"
        matches = flows_by_name.get(name, [])
        flow = matches.pop(0) if matches else None
        if flow is None:
            lines.append(f"| {index} | {name} | skipped | - | $0.0000 |")
            continue
        planned_flow_span_ids.add(flow.span_id)
        cost = flow.cost_usd + view.descendant_costs.get(flow.span_id, 0.0)
        lines.append(f"| {index} | {name} | {flow.status} | {_format_duration(flow)} | ${cost:.4f} |")

    unmatched_flows = sorted(
        (span for span in view.spans_by_id.values() if span.kind == SpanKind.FLOW and span.span_id not in planned_flow_span_ids),
        key=lambda span: (span.sequence_no, span.started_at, str(span.span_id)),
    )
    for flow in unmatched_flows:
        cost = flow.cost_usd + view.descendant_costs.get(flow.span_id, 0.0)
        lines.append(f"| {flow.sequence_no or '-'} | {flow.name} | {flow.status} | {_format_duration(flow)} | ${cost:.4f} |")
    lines.append("")
    return lines


def _build_conversation_chain_lines(view: SpanTreeView) -> list[str]:
    """Render conversation chains showing warmup → fork topology."""
    conversation_spans = [span for span in view.spans_by_id.values() if span.kind == SpanKind.CONVERSATION and span.previous_conversation_id is not None]
    if not conversation_spans:
        return []
    children_of: dict[UUID, list[SpanRecord]] = {}
    for span in conversation_spans:
        assert span.previous_conversation_id is not None
        children_of.setdefault(span.previous_conversation_id, []).append(span)
    roots: set[UUID] = set()
    for parent_id in children_of:
        parent = view.spans_by_id.get(parent_id)
        if parent is not None and parent.previous_conversation_id is None:
            roots.add(parent_id)
    if not roots:
        return []
    lines = ["## Conversation Chains", ""]
    for root_id in sorted(roots, key=lambda rid: view.spans_by_id[rid].started_at):
        root = view.spans_by_id[root_id]
        forks = sorted(children_of.get(root_id, []), key=lambda s: s.started_at)
        fork_names = ", ".join(s.name for s in forks)
        lines.append(f"- {root.name} → [{fork_names}]")
    lines.append("")
    return lines


def generate_summary_from_tree(tree: list[SpanRecord], root_deployment_id: UUID) -> str:
    """Render summary.md content for a span-era deployment tree."""
    view = build_span_tree_view(tree, root_deployment_id)
    if view is None:
        return "# No execution data found\n"

    total_tokens = view.totals.tokens_input + view.totals.tokens_output + view.totals.tokens_cache_read + view.totals.tokens_reasoning
    header = f"# {view.root_span.deployment_name or view.root_span.name} / {view.root_span.run_id}"
    lines = [header]
    if view.root_span.description:
        lines.append(f"\n> {view.root_span.description}")
    lines.extend([
        "",
        (
            f"**Status**: {view.root_span.status} | **Duration**: {_format_duration(view.root_span)} | "
            f"**Spans**: {sum(view.counts_by_kind.values())} | **Documents**: {len(view.document_shas)}"
        ),
        (
            f"**Flows**: {view.counts_by_kind.get(SpanKind.FLOW, 0)} | "
            f"**Tasks**: {view.counts_by_kind.get(SpanKind.TASK, 0)} | "
            f"**Attempts**: {view.counts_by_kind.get(SpanKind.ATTEMPT, 0)} | "
            f"**Operations**: {view.counts_by_kind.get(SpanKind.OPERATION, 0)} | "
            f"**Conversations**: {view.counts_by_kind.get(SpanKind.CONVERSATION, 0)} | "
            f"**LLM Rounds**: {view.counts_by_kind.get(SpanKind.LLM_ROUND, 0)} | "
            f"**Tool Calls**: {view.counts_by_kind.get(SpanKind.TOOL_CALL, 0)}"
        ),
        f"**Total Tokens**: {total_tokens:,} | **Total Cost**: ${view.totals.cost_usd:.4f}",
        "",
        "## Navigation",
        "",
        f"- `runs/{run_directory_name(view.root_span.started_at, view.root_span.name, view.root_span.span_id)}/` — hierarchical execution tree",
        "- `documents/` — content-addressed document metadata grouped by type",
        "- `blobs/` — flat binary blob storage",
        "- `logs.jsonl` — chronological execution logs",
        "- `costs.md` — cost breakdown by model",
        "- `llm_calls.jsonl` — chronological LLM round index",
        "- `errors.md` — failed span report when failures exist",
        "- `documents.md` — document catalog when documents exist",
        "",
        "## CLI Commands",
        "",
        "```bash",
        f"ai-trace show {root_deployment_id} --db-path <download-dir>",
        "",
        "# Replay a specific span",
        "ai-replay show --from-db <span-id> --db-path <download-dir>",
        "ai-replay run --from-db <span-id> --db-path <download-dir>",
        "```",
        "",
    ])
    lines.extend(_build_flow_plan_lines(view))
    lines.extend(_build_conversation_chain_lines(view))
    lines.extend(_build_failures_lines(view))
    lines.extend(["## Execution Tree", ""])
    lines.extend(format_span_tree_lines(view, include_filenames=True))
    lines.append("")
    return "\n".join(lines)


def generate_costs_from_tree(tree: list[SpanRecord], root_deployment_id: UUID) -> str:
    """Render costs.md content for a span-era deployment tree."""
    view = build_span_tree_view(tree, root_deployment_id)
    if view is None:
        return ""

    per_model: dict[str, CostTotals] = {}
    rounds_per_model: Counter[str] = Counter()
    for span in tree:
        if span.kind != SpanKind.LLM_ROUND:
            continue
        meta = view.meta_by_id[span.span_id]
        metrics = view.metrics_by_id[span.span_id]
        context = f"Span {span.span_id}"
        model = _detail_str(meta, MODEL_KEY) or UNKNOWN_MODEL_LABEL
        current = per_model.get(model, CostTotals())
        per_model[model] = CostTotals(
            cost_usd=current.cost_usd + span.cost_usd,
            tokens_input=current.tokens_input + _detail_int(metrics, TOKENS_INPUT_KEY, context=context, field_name="metrics_json"),
            tokens_output=current.tokens_output + _detail_int(metrics, TOKENS_OUTPUT_KEY, context=context, field_name="metrics_json"),
            tokens_cache_read=current.tokens_cache_read + _detail_int(metrics, TOKENS_CACHE_READ_KEY, context=context, field_name="metrics_json"),
            tokens_reasoning=current.tokens_reasoning + _detail_int(metrics, TOKENS_REASONING_KEY, context=context, field_name="metrics_json"),
        )
        rounds_per_model[model] += 1

    lines = ["# Cost by Model", ""]
    if per_model:
        ordered_models = sorted(
            per_model.items(),
            key=lambda item: (-item[1].cost_usd, item[0]),
        )
        lines.extend(["| Model | Rounds | Input | Cache Read | Output | Reasoning | Cost |", "|---|---:|---:|---:|---:|---:|---:|"])
        for model, totals in ordered_models:
            lines.append(
                f"| {model} | {rounds_per_model[model]} | {totals.tokens_input:,} | {totals.tokens_cache_read:,} | "
                f"{totals.tokens_output:,} | {totals.tokens_reasoning:,} | ${totals.cost_usd:.4f} |"
            )
        lines.append("")
    lines.extend([
        f"**Total**: ${view.totals.cost_usd:.4f}",
        "",
    ])
    return "\n".join(lines)
