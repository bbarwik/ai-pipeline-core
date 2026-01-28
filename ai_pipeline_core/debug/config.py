"""Configuration for local trace debugging."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TraceDebugConfig(BaseModel):
    """Configuration for local trace debugging.

    Controls how traces are written to the local filesystem for debugging.
    Enable by setting TRACE_DEBUG_PATH environment variable.
    """

    model_config = ConfigDict(frozen=True)

    path: Path = Field(description="Directory for debug traces")
    enabled: bool = Field(default=True, description="Whether debug tracing is enabled")

    # Content size limits (Issue #2)
    max_file_bytes: int = Field(
        default=50_000,
        description="Max bytes for input.yaml or output.yaml. Elements externalized to stay under.",
    )
    max_element_bytes: int = Field(
        default=10_000,
        description="Max bytes for single element. Above this, partial + artifact ref.",
    )
    element_excerpt_bytes: int = Field(
        default=2_000,
        description="Bytes of content to keep inline when element exceeds max_element_bytes.",
    )
    max_content_bytes: int = Field(
        default=10_000_000,
        description="Max bytes for any single artifact. Above this, truncate.",
    )

    # Image handling (Issue #7 - no changes per user)
    extract_base64_images: bool = Field(
        default=True,
        description="Extract base64 images to artifact files",
    )

    # Span optimization (Issue #4)
    merge_wrapper_spans: bool = Field(
        default=True,
        description="Merge Prefect wrapper spans with inner traced function spans",
    )

    # Events (Issue #12)
    events_file_mode: str = Field(
        default="errors_only",
        description="When to write events.yaml: 'all', 'errors_only', 'none'",
    )

    # Indexes (Issue #1)
    include_llm_index: bool = Field(
        default=True,
        description="Generate _llm_calls.yaml with LLM-specific details",
    )
    include_error_index: bool = Field(
        default=True,
        description="Generate _errors.yaml with failed span details",
    )

    # Cleanup
    max_traces: int | None = Field(
        default=None,
        description="Max number of traces to keep. None for unlimited.",
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
    generate_summary: bool = Field(default=True, description="Generate _summary.md")
