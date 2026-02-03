"""Document wrapper for agent results.

Pipeline tasks must return Document instances (framework rule).
AgentOutputDocument wraps AgentResult to comply with this requirement
while preserving provenance tracking.
"""

import re

from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.logging import get_pipeline_logger

from .base import AgentResult

__all__ = ["AgentOutputDocument"]

logger = get_pipeline_logger(__name__)

# Pattern for valid attachment names (alphanumeric, dots, dashes, underscores)
_VALID_NAME_PATTERN = re.compile(r"^[\w.\-]+$")


def _sanitize_name(name: str) -> str | None:
    """Sanitize artifact name for use as attachment.

    Returns None if name cannot be sanitized (should be skipped).
    """
    # Replace path separators with underscores
    sanitized = name.replace("/", "_").replace("\\", "_")

    # Check if result is valid
    if _VALID_NAME_PATTERN.match(sanitized):
        return sanitized

    return None


class AgentOutputDocument(Document):
    """Document wrapping agent execution results.

    Use this class when creating @pipeline_task functions that run agents.
    It converts AgentResult to a Document that:
    - Contains the primary artifact as content
    - Optionally includes other artifacts as attachments
    - Tracks provenance via origins parameter

    Usage: See tests/agents/test_documents.py for usage patterns.
    """

    @classmethod
    def from_result(
        cls,
        result: AgentResult,
        *,
        artifact_name: str = "output.md",
        name: str = "agent_output.md",
        description: str = "",
        origins: tuple[str, ...],  # Required! Enforces provenance
        include_artifacts_as_attachments: bool = False,
        max_attachment_size: int = 10 * 1024 * 1024,  # 10MB
    ) -> "AgentOutputDocument":
        """Create document from agent result.

        Args:
            result: AgentResult from run_agent()
            artifact_name: Name of artifact to use as primary content
            name: Document name (filename)
            description: Human-readable description
            origins: SHA256 hashes of documents that caused this execution.
                     REQUIRED - agents don't run in isolation, something
                     triggered them. Pass the triggering document's SHA256.
            include_artifacts_as_attachments: If True, include other artifacts
                as document attachments
            max_attachment_size: Skip attachments larger than this (bytes)

        Returns:
            AgentOutputDocument instance

        Raises:
            ValueError: If origins is empty (provenance required)
        """
        if not origins:
            raise ValueError(
                "origins is required for AgentOutputDocument. "
                "Pass the SHA256 of the document(s) that triggered this agent run. "
                "Example: origins=(input_doc.sha256,)"
            )

        # Extract primary content
        if result.success:
            content = result.get_artifact(artifact_name) or ""
            if not content:
                # Fallback: find any markdown file
                for art_name in result.artifacts:
                    if art_name.endswith(".md"):
                        content = result.get_artifact(art_name) or ""
                        break
                if not content:
                    content = f"Agent completed but no output artifact found.\nArtifacts: {list(result.artifacts.keys())}"
        else:
            # Build error content with all available info
            parts = [f"Agent failed: {result.error or 'Unknown error'}"]
            if result.traceback:
                parts.append(f"\n\n## Traceback\n```\n{result.traceback}\n```")
            if result.stderr:
                parts.append(f"\n\n## Stderr\n```\n{result.stderr}\n```")
            if result.exit_code is not None:
                parts.append(f"\n\nExit code: {result.exit_code}")
            content = "".join(parts)

        # Build attachments from other artifacts (optional)
        attachments: list[Attachment] | None = None
        if include_artifacts_as_attachments and result.artifacts:
            attachments = []
            for art_name, art_bytes in result.artifacts.items():
                # Skip primary content artifact
                if art_name == artifact_name:
                    continue

                # Skip large files
                if len(art_bytes) > max_attachment_size:
                    logger.warning(f"Skipping large artifact: {art_name} ({len(art_bytes)} > {max_attachment_size} bytes)")
                    continue

                # Sanitize name
                safe_name = _sanitize_name(art_name)
                if safe_name is None:
                    logger.warning(f"Skipping artifact with invalid name: {art_name}")
                    continue

                attachments.append(
                    Attachment(
                        name=safe_name,
                        content=art_bytes,
                        description=f"Agent artifact: {art_name}",
                    )
                )

        return cls.create(
            name=name,
            content=content,
            description=description or f"Output from agent: {result.agent_name or 'unknown'}",
            origins=origins,
            attachments=tuple(attachments) if attachments else (),
        )
