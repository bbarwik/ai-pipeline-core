"""Render execution DAG summaries and cost reports as Markdown strings."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.database._types import ExecutionNode, NodeKind, NodeStatus

__all__ = [
    "generate_costs",
    "generate_summary",
]

_FOUR_SPACES = "    "
MAX_TREE_DEPTH = 100


@dataclass(frozen=True, slots=True)
class _TreeIndex:
    """Precomputed DAG lookups for summary and cost rendering."""

    nodes_by_id: dict[UUID, ExecutionNode]
    children_map: dict[UUID, list[UUID]]
    tasks_by_flow_id: dict[UUID, list[ExecutionNode]]
    turns_by_flow_id: dict[UUID, list[ExecutionNode]]
    flow_names_by_id: dict[UUID, str]
    descendant_turn_costs: dict[UUID, float]
    descendant_turn_counts: dict[UUID, int]


def _build_tree_index(tree: list[ExecutionNode]) -> _TreeIndex:
    """Build reusable DAG indexes once for a deployment tree."""
    nodes_by_id = {node.node_id: node for node in tree}
    children_map = _build_children_map(tree)
    tasks_by_flow_id: dict[UUID, list[ExecutionNode]] = {}
    turns_by_flow_id: dict[UUID, list[ExecutionNode]] = {}
    flow_names_by_id: dict[UUID, str] = {}

    for node in tree:
        if node.node_kind == NodeKind.FLOW:
            flow_names_by_id[node.node_id] = node.name
        if node.node_kind == NodeKind.TASK and node.flow_id is not None:
            tasks_by_flow_id.setdefault(node.flow_id, []).append(node)
        if node.node_kind == NodeKind.CONVERSATION_TURN and node.flow_id is not None:
            turns_by_flow_id.setdefault(node.flow_id, []).append(node)

    descendant_turn_costs: dict[UUID, float] = {}
    descendant_turn_counts: dict[UUID, int] = {}
    visiting: set[UUID] = set()

    def _visit(node_id: UUID) -> tuple[float, int]:
        if node_id in descendant_turn_costs:
            return descendant_turn_costs[node_id], descendant_turn_counts[node_id]
        if node_id in visiting:
            return 0.0, 0

        visiting.add(node_id)
        total_cost = 0.0
        total_turns = 0
        for child_id in children_map.get(node_id, []):
            child_cost, child_turns = _visit(child_id)
            child = nodes_by_id[child_id]
            if child.node_kind == NodeKind.CONVERSATION_TURN:
                total_cost += child.cost_usd
                total_turns += 1
            total_cost += child_cost
            total_turns += child_turns

        descendant_turn_costs[node_id] = total_cost
        descendant_turn_counts[node_id] = total_turns
        visiting.remove(node_id)
        return total_cost, total_turns

    for node in tree:
        _visit(node.node_id)

    return _TreeIndex(
        nodes_by_id=nodes_by_id,
        children_map=children_map,
        tasks_by_flow_id=tasks_by_flow_id,
        turns_by_flow_id=turns_by_flow_id,
        flow_names_by_id=flow_names_by_id,
        descendant_turn_costs=descendant_turn_costs,
        descendant_turn_counts=descendant_turn_counts,
    )


async def generate_summary(reader: DatabaseReader, deployment_id: UUID) -> str:
    """Generate summary.md content from the execution DAG."""
    tree = await reader.get_deployment_tree(deployment_id)
    if not tree:
        return "# No execution data found\n"

    index = _build_tree_index(tree)

    # Find deployment node
    deploy_node: ExecutionNode | None = None
    for n in tree:
        if n.node_kind == NodeKind.DEPLOYMENT:
            deploy_node = n
            break

    if deploy_node is None:
        return "# No deployment node found\n"

    # Gather stats
    flows = [n for n in tree if n.node_kind == NodeKind.FLOW]
    tasks = [n for n in tree if n.node_kind == NodeKind.TASK]
    turns = [n for n in tree if n.node_kind == NodeKind.CONVERSATION_TURN]
    failed = [n for n in tree if n.status == NodeStatus.FAILED]

    total_cost = sum(t.cost_usd for t in turns)
    total_tokens = sum(t.tokens_input + t.tokens_output for t in turns)
    completed_flows = sum(1 for f in flows if f.status == NodeStatus.COMPLETED)

    # Collect document SHA256s
    all_doc_shas: set[str] = set()
    for n in tree:
        all_doc_shas.update(n.input_document_shas)
        all_doc_shas.update(n.output_document_shas)
        all_doc_shas.update(n.context_document_shas)

    # Status
    if failed:
        status_text = f"Failed ({len(failed)} errors)"
    elif deploy_node.status == NodeStatus.COMPLETED:
        status_text = "Completed"
    else:
        status_text = str(deploy_node.status.value).title()
    duration_str = _format_duration(deploy_node)

    lines: list[str] = [
        f"# {deploy_node.deployment_name} / {deploy_node.run_id}",
        "",
        f"**Status**: {status_text} | **Duration**: {duration_str} | **Flows**: {completed_flows}/{len(flows)} completed | **Tasks**: {len(tasks)}",
        f"**LLM Turns**: {len(turns)} | **Documents**: {len(all_doc_shas)} | **Total Tokens**: {total_tokens:,} | **Total Cost**: ${total_cost:.4f}",
        "",
    ]

    # Navigation
    lines.extend([
        "## Navigation",
        "",
        "- `runs/` — hierarchical execution tree (browse JSON files directly)",
        "- `logs.jsonl` — chronological execution logs",
        "- `costs.md` — cost rollup by flow/task",
        "- `blobs/` — binary document content",
        "",
    ])

    # Flow Plan — merge payload flow_plan with actual flow nodes
    _build_flow_plan_lines(deploy_node, flows, index, lines)

    # Failures
    if failed:
        lines.extend(["## Failures", ""])
        for f in failed:
            parent_chain = _build_parent_chain(f, index.nodes_by_id)
            chain_str = "/".join(parent_chain) if parent_chain else ""
            error_str = f"{f.error_type}: {f.error_message}" if f.error_type else f.error_message
            if chain_str:
                lines.append(f"- `{chain_str}/{f.name}`")
            else:
                lines.append(f"- `{f.name}`")
            if error_str:
                lines.append(f"  `{error_str}`")
        lines.append("")

    # Execution Tree
    lines.extend(["## Execution Tree", ""])
    _build_tree_lines(deploy_node, index, 0, lines)
    lines.append("")

    return "\n".join(lines)


async def generate_costs(reader: DatabaseReader, deployment_id: UUID) -> str:
    """Generate costs.md content — cost aggregation by flow/task."""
    tree = await reader.get_deployment_tree(deployment_id)
    if not tree:
        return ""

    flows = [n for n in tree if n.node_kind == NodeKind.FLOW]
    if not flows:
        return ""

    index = _build_tree_index(tree)

    lines: list[str] = ["# Cost by Flow", ""]
    lines.append("| Flow | Step | Tasks | LLM Turns | Cost | Status |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for flow in sorted(flows, key=lambda f: f.sequence_no):
        flow_tasks = index.tasks_by_flow_id.get(flow.node_id, [])
        flow_turns = index.turns_by_flow_id.get(flow.node_id, [])
        flow_cost = sum(t.cost_usd for t in flow_turns)
        lines.append(f"| {flow.name} | {flow.sequence_no} | {len(flow_tasks)} | {len(flow_turns)} | ${flow_cost:.4f} | {flow.status.value} |")

    total_cost = sum(n.cost_usd for n in tree if n.node_kind == NodeKind.CONVERSATION_TURN)
    lines.extend(["", f"**Total**: ${total_cost:.4f}", ""])

    # Cost by task (for tasks with cost > 0)
    tasks_with_cost: list[tuple[ExecutionNode, float]] = []
    for task in (n for n in tree if n.node_kind == NodeKind.TASK):
        task_cost = _sum_turn_costs(task, index)
        if task_cost > 0:
            tasks_with_cost.append((task, task_cost))

    if tasks_with_cost:
        tasks_with_cost.sort(key=lambda x: x[1], reverse=True)
        lines.extend(["## Cost by Task", ""])
        lines.append("| Task | Flow | LLM Turns | Cost | Status |")
        lines.append("|---|---|---:|---:|---|")
        for task, cost in tasks_with_cost:
            flow_name = _find_flow_name(task, index)
            turn_count = _count_descendant_turns(task, index)
            lines.append(f"| {task.name} | {flow_name} | {turn_count} | ${cost:.4f} | {task.status.value} |")
        lines.append("")

    return "\n".join(lines)


def _build_children_map(tree: list[ExecutionNode]) -> dict[UUID, list[UUID]]:
    """Build parent_node_id -> [child_ids] map once for the whole tree."""
    children_map: dict[UUID, list[UUID]] = {}
    for n in tree:
        children_map.setdefault(n.parent_node_id, []).append(n.node_id)
    return children_map


def _format_duration(node: ExecutionNode) -> str:
    """Format node duration as human-readable string."""
    if node.ended_at is None:
        return "running..." if node.status == NodeStatus.RUNNING else "-"
    delta = node.ended_at - node.started_at
    return _format_timedelta(delta)


def _format_timedelta(delta: timedelta) -> str:
    """Format timedelta as human-readable string."""
    secs = delta.total_seconds()
    if secs < 1:
        return f"{int(secs * 1000)}ms"
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m {int(secs % 60)}s"
    return f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"


def _count_descendant_turns(node: ExecutionNode, index: _TreeIndex) -> int:
    """Count conversation_turn descendants."""
    return index.descendant_turn_counts.get(node.node_id, 0)


def _sum_turn_costs(parent: ExecutionNode, index: _TreeIndex) -> float:
    """Sum cost_usd from all conversation_turn descendants of a node."""
    return index.descendant_turn_costs.get(parent.node_id, 0.0)


def _find_flow_name(task: ExecutionNode, index: _TreeIndex) -> str:
    """Find the flow name for a task."""
    if task.flow_id is not None:
        flow_name = index.flow_names_by_id.get(task.flow_id)
        if flow_name:
            return flow_name
    # Fallback: walk parent chain
    parent = index.nodes_by_id.get(task.parent_node_id)
    if parent is not None and parent.node_kind == NodeKind.FLOW:
        return parent.name
    return ""


def _build_parent_chain(node: ExecutionNode, nodes_by_id: dict[UUID, ExecutionNode]) -> list[str]:
    """Build parent chain names from node to root."""
    chain: list[str] = []
    current_id = node.parent_node_id
    visited: set[UUID] = set()
    while current_id in nodes_by_id:
        if current_id in visited:
            break
        visited.add(current_id)
        parent = nodes_by_id[current_id]
        chain.append(parent.name)
        current_id = parent.parent_node_id
    return list(reversed(chain))


def _build_flow_plan_lines(
    deploy_node: ExecutionNode,
    actual_flows: list[ExecutionNode],
    index: _TreeIndex,
    lines: list[str],
) -> None:
    """Build Flow Plan table merging payload flow_plan with actual flow nodes."""
    # Build lookup of actual flows by name
    flows_by_name: dict[str, ExecutionNode] = {}
    for flow in actual_flows:
        flows_by_name[flow.name] = flow

    # Get planned flows from deployment payload
    flow_plan: list[dict[str, Any]] = deploy_node.payload.get("flow_plan", [])

    # Merge: planned flows first, then any actual flows not in the plan
    plan_entries: list[tuple[int, str, ExecutionNode | None]] = []
    planned_names: set[str] = set()

    for i, plan_entry in enumerate(flow_plan):
        name = plan_entry.get("name", f"Flow_{i}")
        planned_names.add(name)
        actual = flows_by_name.get(name)
        seq = actual.sequence_no if actual else i + 1
        plan_entries.append((seq, name, actual))

    # Add actual flows not in plan
    for flow in sorted(actual_flows, key=lambda f: f.sequence_no):
        if flow.name not in planned_names:
            plan_entries.append((flow.sequence_no, flow.name, flow))

    if not plan_entries:
        return

    lines.extend(["## Flow Plan", ""])
    lines.append("| Step | Flow | Status | Duration | Cost |")
    lines.append("|---|---|---|---:|---:|")
    for seq, name, flow in plan_entries:
        if flow is not None:
            flow_cost = _sum_turn_costs(flow, index)
            flow_dur = _format_duration(flow)
            lines.append(f"| {seq} | {name} | {flow.status.value} | {flow_dur} | ${flow_cost:.4f} |")
        else:
            lines.append(f"| {seq} | {name} | skipped | - | $0.0000 |")
    lines.append("")


def _build_tree_lines(
    node: ExecutionNode,
    index: _TreeIndex,
    depth: int,
    lines: list[str],
    visited: set[UUID] | None = None,
) -> None:
    """Build compact execution tree recursively."""
    if visited is None:
        visited = set()
    if node.node_id in visited:
        lines.append(f"{_FOUR_SPACES * depth}[cycle detected while rendering execution tree]")
        return
    if depth > MAX_TREE_DEPTH:
        lines.append(f"{_FOUR_SPACES * depth}[execution tree depth limit reached]")
        return

    visited.add(node.node_id)
    indent = _FOUR_SPACES * depth
    kind_label = node.node_kind.value
    dur = _format_duration(node)

    if node.node_kind == NodeKind.CONVERSATION_TURN:
        # turn[{seq}]: {model} {tokens_in} in / {tokens_out} out ${cost}
        token_info = _format_token_pair(node.tokens_input, node.tokens_output)
        cost_str = f"${node.cost_usd:.4f}" if node.cost_usd > 0 else ""
        model_str = node.model if node.model else ""
        parts = [f"{indent}turn[{node.sequence_no}]: {model_str}"]
        if token_info:
            parts.append(f" {token_info}")
        if cost_str:
            parts.append(f" {cost_str}")
        lines.append("".join(parts))
    else:
        parts = [f"{indent}{kind_label}"]
        if node.node_kind == NodeKind.FLOW:
            parts.append(f"[{node.sequence_no}]")
        parts.append(f": {node.name}")
        parts.append(f" {node.status.value} {dur}")

        if node.node_kind in {NodeKind.FLOW, NodeKind.TASK}:
            cost = _sum_turn_costs(node, index)
            if cost > 0:
                parts.append(f" ${cost:.4f}")

        lines.append("".join(parts))

    # Recurse into children
    child_ids = index.children_map.get(node.node_id, [])
    children = sorted(
        (index.nodes_by_id[cid] for cid in child_ids if cid in index.nodes_by_id),
        key=lambda n: (n.sequence_no, n.node_id),
    )
    for child in children:
        _build_tree_lines(child, index, depth + 1, lines, visited)
    visited.remove(node.node_id)


def _format_token_pair(input_tokens: int, output_tokens: int) -> str:
    """Format token counts."""
    if input_tokens == 0 and output_tokens == 0:
        return ""
    return f"{_format_token_count(input_tokens)} in / {_format_token_count(output_tokens)} out"


def _format_token_count(n: int) -> str:
    """Format token count: raw if <1000, K-suffix if >=1000."""
    if n >= 1000:
        return f"{round(n / 1000)}K"
    return str(n)
