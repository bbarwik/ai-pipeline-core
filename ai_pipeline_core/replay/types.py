"""Pydantic models for replay payloads.

Three payload types capture everything needed to re-execute an LLM call,
pipeline task, or pipeline flow. All use YAML for human-editable serialization
when replay payloads are written to disk.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ai_pipeline_core.database._protocol import DatabaseReader
from ai_pipeline_core.documents._context import DocumentSha256

__all__ = [
    "ConversationReplay",
    "DocumentRef",
    "FlowReplay",
    "HistoryEntry",
    "TaskReplay",
    "ToolCallEntry",
    "_infer_db_path",
]


def _infer_db_path(replay_file: Path) -> Path:
    """Walk up from replay_file to find a FilesystemDatabase root."""
    current = replay_file.resolve().parent
    while current != current.parent:
        if (current / "runs").is_dir() or (current / "blobs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(
        f"Could not find a FilesystemDatabase root (directory with runs/ or blobs/) "
        f"in any ancestor of {replay_file}. Use --db-path to specify the database directory."
    )


class DocumentRef(BaseModel):
    """Reference to a document stored in a database backend by SHA256.

    Documents are not inlined in replay YAML — they are referenced by SHA256 hash
    and resolved from the database at execution time.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    doc_ref: DocumentSha256 = Field(alias="$doc_ref")  # Full SHA256 hash of the document
    class_name: str  # Document subclass name for type resolution
    name: str  # Original document name


class ToolCallEntry(BaseModel):
    """Typed tool call entry for replay serialization."""

    model_config = ConfigDict(frozen=True)

    id: str
    function_name: str
    arguments: str


class HistoryEntry(BaseModel):
    """Single entry in a conversation's message history.

    Type determines which fields are populated:
    user_text/assistant_text use text, response uses content (with optional tool_calls),
    tool_result uses tool_call_id/function_name/content, document uses doc_ref.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    type: Literal["user_text", "assistant_text", "response", "document", "tool_result"]
    text: str | None = None  # For user_text and assistant_text entries
    content: str | None = None  # For response and tool_result entries
    doc_ref: str | None = Field(None, alias="$doc_ref")  # SHA256 for document entries
    class_name: str | None = None  # Document class for document entries
    name: str | None = None  # Document name for document entries
    tool_call_id: str | None = None  # For tool_result entries
    function_name: str | None = None  # For tool_result entries
    tool_calls: list[ToolCallEntry] | None = None  # For response entries with tool calls


class ConversationReplay(BaseModel):
    """Replay payload for a Conversation.send() / send_structured() call.

    Serialized to YAML files such as ``conversation.yaml`` for replay and inspection.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    version: int = 1
    payload_type: Literal["conversation"] = "conversation"
    model: str
    model_options: dict[str, Any] = {}  # ModelOptions fields (reasoning_effort, cache_ttl, etc.)
    prompt: str = ""
    prompt_documents: tuple[DocumentRef, ...] = ()  # Prompt documents sent via send(Document) or send([Document, ...])
    response_format: str | None = None  # "module:ClassName" path to Pydantic model for send_structured()
    purpose: str | None = None  # Purpose label for the LLM call
    context: tuple[DocumentRef, ...] = ()  # Context documents resolved by SHA256 at execution
    history: tuple[HistoryEntry, ...] = ()  # Prior conversation turns
    enable_substitutor: bool = True  # URL/token protection via URLSubstitutor
    extract_result_tags: bool = False  # Extract content between <result> tags
    include_date: bool = True
    current_date: str | None = None
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

    async def execute(self, database: DatabaseReader) -> Any:
        """Resolve document references and re-execute the LLM call."""
        from ._execute import execute_conversation

        return await execute_conversation(self, database)


class TaskReplay(BaseModel):
    """Replay payload for a PipelineTask invocation.

    Serialized to YAML files such as ``task.yaml`` for replay and inspection.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    version: int = 1
    payload_type: Literal["pipeline_task"] = "pipeline_task"
    function_path: str  # "module:qualname" path to a PipelineTask class or task function
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

    async def execute(self, database: DatabaseReader) -> Any:
        """Resolve document references and re-execute the task."""
        from ._execute import execute_task

        return await execute_task(self, database)


class FlowReplay(BaseModel):
    """Replay payload for a PipelineFlow call.

    Serialized to YAML files such as ``flow.yaml`` for replay and inspection.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    version: int = 1
    payload_type: Literal["pipeline_flow"] = "pipeline_flow"
    function_path: str  # "module:qualname" path to the flow function
    run_id: str  # Unique run identifier for database-backed replay scoping
    documents: tuple[DocumentRef, ...] = ()  # Input documents referenced by SHA256
    flow_options: dict[str, Any] = {}  # FlowOptions fields (filtered to known fields at execution)
    flow_params: dict[str, Any] = {}  # PipelineFlow constructor kwargs for replay
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

    async def execute(self, database: DatabaseReader) -> Any:
        """Resolve document references and re-execute the flow function."""
        from ._execute import execute_flow

        return await execute_flow(self, database)
