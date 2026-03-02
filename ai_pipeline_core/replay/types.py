"""Pydantic models for replay payloads.

Three payload types capture everything needed to re-execute an LLM call,
pipeline task, or pipeline flow. All use YAML for human-editable serialization.
Replay files are auto-captured by the trace writer during pipeline execution.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ConversationReplay",
    "DocumentRef",
    "FlowReplay",
    "HistoryEntry",
    "TaskReplay",
    "_infer_store_base",
]


def _infer_store_base(replay_file: Path) -> Path:
    """Walk up from replay_file to find .trace/ directory, return its parent.

    Used automatically by the CLI. Only needed programmatically when bypassing the CLI.
    Convention: .trace/ is always a direct child of the store base directory.
    """
    current = replay_file.resolve().parent
    while current != current.parent:
        if current.name == ".trace":
            return current.parent
        current = current.parent
    raise FileNotFoundError(
        f"Could not find .trace/ directory in any ancestor of {replay_file}. "
        f"The replay file must be inside a .trace/ directory tree, or use --store to specify the store base."
    )


class DocumentRef(BaseModel):
    """Reference to a document stored in LocalDocumentStore by SHA256.

    Documents are not inlined in replay YAML — they are referenced by SHA256 hash
    and resolved from the local store at execution time.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    doc_ref: str = Field(alias="$doc_ref")  # Full SHA256 hash of the document
    class_name: str  # Document subclass name for type resolution
    name: str  # Original document name


class HistoryEntry(BaseModel):
    """Single entry in a conversation's message history.

    Type determines which fields are populated:
    user_text/assistant_text use text, response uses content, document uses doc_ref.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    type: Literal["user_text", "assistant_text", "response", "document"]
    text: str | None = None  # For user_text and assistant_text entries
    content: str | None = None  # For response entries
    doc_ref: str | None = Field(None, alias="$doc_ref")  # SHA256 for document entries
    class_name: str | None = None  # Document class for document entries
    name: str | None = None  # Document name for document entries


class ConversationReplay(BaseModel):
    """Replay payload for a Conversation.send() / send_structured() call.

    Auto-captured in each span directory as ``conversation.yaml``.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    version: int = 1
    payload_type: Literal["conversation"] = "conversation"
    model: str
    model_options: dict[str, Any] = {}  # ModelOptions fields (reasoning_effort, cache_ttl, etc.)
    prompt: str
    response_format: str | None = None  # "module:ClassName" path to Pydantic model for send_structured()
    purpose: str | None = None  # Laminar span label for tracing
    context: tuple[DocumentRef, ...] = ()  # Context documents resolved by SHA256 at execution
    history: tuple[HistoryEntry, ...] = ()  # Prior conversation turns
    enable_substitutor: bool = True  # URL/token protection via URLSubstitutor
    extract_result_tags: bool = False  # Extract content between <result> tags
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, store_base: Path) -> Any:
        """Resolve document references and re-execute the LLM call."""
        from ._execute import execute_conversation

        return await execute_conversation(self, store_base)


class TaskReplay(BaseModel):
    """Replay payload for a @pipeline_task call.

    Auto-captured in each span directory as ``task.yaml``.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    version: int = 1
    payload_type: Literal["pipeline_task"] = "pipeline_task"
    function_path: str  # "module:qualname" path to the task function
    arguments: dict[str, Any] = {}  # Documents as $doc_ref, BaseModels as dicts, primitives as-is
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, store_base: Path) -> Any:
        """Resolve document references and re-execute the task function."""
        from ._execute import execute_task

        return await execute_task(self, store_base)


class FlowReplay(BaseModel):
    """Replay payload for a @pipeline_flow call.

    Auto-captured in each span directory as ``flow.yaml``.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    version: int = 1
    payload_type: Literal["pipeline_flow"] = "pipeline_flow"
    function_path: str  # "module:qualname" path to the flow function
    run_id: str  # Unique run identifier for document store scoping
    documents: tuple[DocumentRef, ...] = ()  # Input documents referenced by SHA256
    flow_options: dict[str, Any] = {}  # FlowOptions fields (filtered to known fields at execution)
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, store_base: Path) -> Any:
        """Resolve document references and re-execute the flow function."""
        from ._execute import execute_flow

        return await execute_flow(self, store_base)
