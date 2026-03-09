"""Helpers for ClickHouse execution-node optimistic updates."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from clickhouse_connect.driver.summary import QuerySummary

from ai_pipeline_core.database._types import ExecutionNode, NodeKind, NodeStatus

UPDATE_NODE_MAX_ATTEMPTS = 5
_DOCUMENT_SHA_COLUMNS = frozenset({"input_document_shas", "output_document_shas", "context_document_shas"})


def build_node_update_command(
    *,
    node_id: UUID,
    updates: dict[str, Any],
    expected_version: int,
    next_version: int,
    node_columns: tuple[str, ...],
    clickhouse_param_type: Callable[[str, Any], str],
    as_list: Callable[[list[Any] | tuple[Any, ...]], list[str]],
) -> tuple[str, dict[str, Any]]:
    """Build the SELECT expression and parameters for a single optimistic update attempt."""
    params: dict[str, Any] = {"node_id": node_id, "expected_version": expected_version, "next_version": next_version}
    select_parts: list[str] = []
    param_idx = 0

    for col in node_columns:
        if col == "version":
            select_parts.append("{next_version:UInt64}")
            continue
        if col not in updates:
            select_parts.append(col)
            continue

        param_name = f"upd_{param_idx}"
        param_idx += 1
        value = updates[col]
        if isinstance(value, (NodeStatus, NodeKind)):
            value = value.value
        elif isinstance(value, dict) and col == "payload":
            value = json.dumps(value, default=str)
        elif isinstance(value, (list, tuple)) and col in _DOCUMENT_SHA_COLUMNS:
            value = as_list(cast("list[Any] | tuple[Any, ...]", value))
        params[param_name] = value
        select_parts.append(f"{{{param_name}:{clickhouse_param_type(col, value)}}}")

    return ", ".join(select_parts), params


def next_node_version(expected_version: int) -> int:
    """Generate a monotonic version for optimistic execution-node updates."""
    return max(expected_version + 1, time.time_ns())


def node_matches_updates(node: ExecutionNode, updates: dict[str, Any]) -> bool:
    """Check whether a read-back node reflects the requested semantic updates."""
    for field_name, expected_value in updates.items():
        actual = getattr(node, field_name)
        comparable_expected = expected_value
        if isinstance(expected_value, list):
            comparable_expected = tuple(cast("list[Any]", expected_value))
        if isinstance(expected_value, dict) and field_name == "payload":
            comparable_expected = dict(cast("dict[str, Any]", expected_value))
        if actual != comparable_expected:
            return False
    return True


def update_node_optimistically(
    *,
    node_id: UUID,
    updates: dict[str, Any],
    load_node: Callable[[UUID], ExecutionNode | None],
    write_attempt: Callable[[int, dict[str, Any]], int],
) -> None:
    """Apply an append-only optimistic node update with read-back verification."""
    requested_updates = dict(updates)
    for _attempt in range(UPDATE_NODE_MAX_ATTEMPTS):
        current_node = load_node(node_id)
        if current_node is None:
            raise KeyError(
                f"Node {node_id} not found in ClickHouse. Insert the node before calling update_node(), "
                f"or verify that the deployment writer successfully persisted the initial execution node."
            )
        write_updates = dict(requested_updates)
        if "updated_at" not in write_updates:
            write_updates["updated_at"] = datetime.now(UTC)
        if write_attempt(current_node.version, write_updates) == 0:
            continue
        refreshed_node = load_node(node_id)
        if refreshed_node is not None and node_matches_updates(refreshed_node, requested_updates):
            return

    raise RuntimeError(
        f"Failed to update node {node_id} after {UPDATE_NODE_MAX_ATTEMPTS} optimistic retries. "
        f"Concurrent writers kept changing the row version; retry the higher-level operation once the contention subsides."
    )


def command_written_rows(result: Any) -> int:
    """Read the written-row count from ClickHouse command() results."""
    if isinstance(result, QuerySummary):
        return result.written_rows
    return result if isinstance(result, int) else 0
