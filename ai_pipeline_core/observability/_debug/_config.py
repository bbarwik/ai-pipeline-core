"""Configuration and shared data types for local debug tracing."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceDebugConfig(BaseModel):
    """Configuration for local ``.trace/`` directory generation.

    Controls filesystem trace output. Enabled automatically in CLI mode.
    Each run overwrites previous ``.trace/`` contents.
    """

    model_config = ConfigDict(frozen=True)

    path: Path = Field(description="Directory for debug traces")
    max_file_bytes: int = Field(
        default=500_000,
        description="Max bytes for input.yaml or output.yaml. Truncated to stay under.",
    )

    # Span filtering
    verbose: bool = Field(
        default=False,
        description="When False (default), duplicate LLM spans are filtered from per-span directories but their cost/token metrics still count in totals.",
    )

    # Span optimization
    merge_wrapper_spans: bool = Field(
        default=True,
        description="Merge Prefect wrapper spans with inner traced function spans",
    )

    # Security - default redaction patterns for common secrets
    redact_patterns: tuple[str, ...] = Field(
        default=(
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
            r"sk-proj-[a-zA-Z0-9\-_]{20,}",  # OpenAI project keys
            r"AKIA[0-9A-Z]{16}",  # AWS access keys
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub personal tokens
            r"gho_[a-zA-Z0-9]{36}",  # GitHub OAuth tokens
            r"xoxb-[a-zA-Z0-9\-]+",  # Slack bot tokens
            r"xoxp-[a-zA-Z0-9\-]+",  # Slack user tokens
            r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]+",  # Passwords
            r"(?i)secret\s*[:=]\s*['\"]?[^\s'\"]+",  # Secrets
            r"(?i)api[_\-]?key\s*[:=]\s*['\"]?[^\s'\"]+",  # API keys
            r"(?i)bearer\s+[a-zA-Z0-9\-_\.]+",  # Bearer tokens
        ),
        description="Regex patterns for secrets to redact",
    )

    # Summary
    generate_summary: bool = Field(default=True, description="Generate summary.md")


# Maps replay payload_type (from replay.payload JSON) to the YAML filename on disk.
REPLAY_PAYLOAD_TO_FILENAME: dict[str, str] = {
    "conversation": "conversation.yaml",
    "pipeline_task": "task.yaml",
    "pipeline_flow": "flow.yaml",
}

# Inverse: filename -> display label (used by summary generation).
REPLAY_FILENAME_TO_LABEL: dict[str, str] = {v: k.removeprefix("pipeline_") for k, v in REPLAY_PAYLOAD_TO_FILENAME.items()}


@dataclass(slots=True)
class SpanInfo:
    """Information about a span for index building.

    Tracks execution details including timing, LLM metrics (tokens, cost, expected_cost, purpose),
    and Prefect context for observability and cost tracking across the trace hierarchy.
    """

    span_id: str
    parent_id: str | None
    name: str
    span_type: str
    status: str
    start_time: datetime
    path: Path  # Actual directory path for this span
    depth: int = 0  # Nesting depth (0 for root)
    order: int = 0  # Global execution order within trace
    end_time: datetime | None = None
    duration_ms: int = 0
    children: list[str] = field(default_factory=list)
    llm_info: dict[str, Any] | None = None
    prefect_info: dict[str, Any] | None = None
    description: str | None = None
    expected_cost: float | None = None
    input_type: str | None = None


@dataclass(slots=True)
class TraceState:
    """State for an active trace.

    Maintains trace metadata and span hierarchy with accumulated cost
    metrics (total_cost, total_expected_cost) for monitoring resource
    usage and budget tracking during trace execution.
    """

    trace_id: str
    name: str
    path: Path
    start_time: datetime
    spans: dict[str, SpanInfo] = field(default_factory=dict)
    root_span_id: str | None = None
    total_tokens: int = 0
    total_cost: float = 0.0
    total_expected_cost: float = 0.0
    llm_call_count: int = 0
    span_counter: int = 0  # Global counter for ordering span directories
    merged_wrapper_ids: set[str] = field(default_factory=set)  # IDs of merged wrappers
