# MODULE: replay
# CLASSES: DocumentRef, ToolCallEntry, HistoryEntry, ConversationReplay, TaskReplay, FlowReplay
# DEPENDS: BaseModel
# PURPOSE: First-class replay system for AI pipeline debugging.
# VERSION: 0.14.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core.replay import ConversationReplay, DocumentRef, FlowReplay, HistoryEntry, TaskReplay
```

## Public API

```python
class DocumentRef(BaseModel):
    """Reference to a document stored in a database backend by SHA256.

Documents are not inlined in replay YAML — they are referenced by SHA256 hash
and resolved from the database at execution time."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    doc_ref: DocumentSha256 = Field(alias='$doc_ref')  # Full SHA256 hash of the document
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
tool_result uses tool_call_id/function_name/content, document uses doc_ref."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal['user_text', 'assistant_text', 'response', 'document', 'tool_result']
    text: str | None = None  # For user_text and assistant_text entries
    content: str | None = None  # For response and tool_result entries
    doc_ref: str | None = Field(None, alias='$doc_ref')  # SHA256 for document entries
    class_name: str | None = None  # Document class for document entries
    name: str | None = None  # Document name for document entries
    tool_call_id: str | None = None  # For tool_result entries
    function_name: str | None = None  # For tool_result entries
    tool_calls: list[ToolCallEntry] | None = None  # For response entries with tool calls


class ConversationReplay(BaseModel):
    """Replay payload for a Conversation.send() / send_structured() call.

Serialized to YAML files such as ``conversation.yaml`` for replay and inspection."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    version: int = 1
    payload_type: Literal['conversation'] = 'conversation'
    model: str
    model_options: dict[str, Any] = {}  # ModelOptions fields (reasoning_effort, cache_ttl, etc.)
    prompt: str = ''
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

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, database: DatabaseReader) -> Any:
        """Resolve document references and re-execute the LLM call."""
        from ._execute import execute_conversation

        return await execute_conversation(self, database)

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


class TaskReplay(BaseModel):
    """Replay payload for a PipelineTask invocation.

Serialized to YAML files such as ``task.yaml`` for replay and inspection."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    version: int = 1
    payload_type: Literal['pipeline_task'] = 'pipeline_task'
    function_path: str  # "module:qualname" path to a PipelineTask class or task function
    arguments: dict[str, Any] = {}  # Documents as $doc_ref, BaseModels as dicts, primitives as-is
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, database: DatabaseReader) -> Any:
        """Resolve document references and re-execute the task."""
        from ._execute import execute_task

        return await execute_task(self, database)

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


class FlowReplay(BaseModel):
    """Replay payload for a PipelineFlow call.

Serialized to YAML files such as ``flow.yaml`` for replay and inspection."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    version: int = 1
    payload_type: Literal['pipeline_flow'] = 'pipeline_flow'
    function_path: str  # "module:qualname" path to the flow function
    run_id: str  # Unique run identifier for database-backed replay scoping
    documents: tuple[DocumentRef, ...] = ()  # Input documents referenced by SHA256
    flow_options: dict[str, Any] = {}  # FlowOptions fields (filtered to known fields at execution)
    flow_params: dict[str, Any] = {}  # PipelineFlow constructor kwargs for replay
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, database: DatabaseReader) -> Any:
        """Resolve document references and re-execute the flow function."""
        from ._execute import execute_flow

        return await execute_flow(self, database)

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


```

## Functions

```python
def main(argv: list[str] | None = None) -> int:
    """CLI entry point for replay operations."""
    parser = argparse.ArgumentParser(prog="ai-replay", description="Execute or inspect replay payloads")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute a replay payload")
    run_parser.add_argument("replay_file", nargs="?", help="Path to replay YAML file (conversation.yaml, task.yaml, flow.yaml)")
    run_parser.add_argument("--from-db", type=str, help="Load replay payload from an execution node ID in the database")
    run_parser.add_argument("--db-path", type=str, help="Use a FilesystemDatabase at this path instead of ClickHouse or auto-discovery")
    run_parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override a field before execution (repeatable)")
    run_parser.add_argument(
        "--import",
        dest="modules",
        action="append",
        metavar="MODULE",
        help="Import a module before replay (registers Document subclasses and functions)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for replay results",
    )

    show_parser = subparsers.add_parser("show", help="Pretty-print a replay YAML file")
    show_parser.add_argument("replay_file", help="Path to replay YAML file")

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.replay_file is None and args.from_db is None:
            parser.error("run requires either a replay_file or --from-db <node_id>")
        if args.replay_file is not None and args.from_db is not None:
            parser.error("run accepts a replay_file or --from-db <node_id>, not both")

    handlers: dict[str, Any] = {"run": _cmd_run, "show": _cmd_show}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)

```

## Examples

**Main show conversation** (`tests/replay/test_cli_usage.py:67`)

```python
def test_main_show_conversation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-pro")

    exit_code = main(["show", str(replay_file)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "ConversationReplay" in output
    assert "gemini-3-pro" in output
```

**Main run with db path** (`tests/replay/test_cli_usage.py:79`)

```python
def test_main_run_with_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_file = _conversation_yaml(tmp_path)
    db_path = tmp_path / "bundle"
    FilesystemDatabase(db_path)

    async def fake_execute(payload: object, database: object) -> _MockResult:
        _ = (payload, database)
        return _MockResult()

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)

    exit_code = main(["run", str(replay_file), "--db-path", str(db_path)])

    assert exit_code == 0
    output_dir = tmp_path / "conversation_replay"
    assert (output_dir / "output.yaml").exists()
```

**Main run with set override** (`tests/replay/test_cli_usage.py:98`)

```python
def test_main_run_with_set_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-flash")
    db_path = tmp_path / "bundle"
    FilesystemDatabase(db_path)

    captured_models: list[str] = []

    async def fake_execute(payload: object, database: object) -> _MockResult:
        _ = database
        captured_models.append(payload.model)
        return _MockResult()

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)

    exit_code = main(["run", str(replay_file), "--db-path", str(db_path), "--set", "model=grok-4.1-fast"])

    assert exit_code == 0
    assert captured_models == ["grok-4.1-fast"]
```

**Main run with output dir** (`tests/replay/test_cli_usage.py:119`)

```python
def test_main_run_with_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay_file = _conversation_yaml(tmp_path)
    db_path = tmp_path / "bundle"
    FilesystemDatabase(db_path)
    custom_dir = tmp_path / "my_output"

    async def fake_execute(payload: object, database: object) -> _MockResult:
        _ = (payload, database)
        return _MockResult()

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)

    exit_code = main(["run", str(replay_file), "--db-path", str(db_path), "--output-dir", str(custom_dir)])

    assert exit_code == 0
    assert (custom_dir / "output.yaml").exists()
```

**Main run from db** (`tests/replay/test_cli_usage.py:138`)

```python
def test_main_run_from_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.TASK,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="ReplayTask",
                sequence_no=0,
                started_at=datetime.now(UTC),
                payload={
                    "replay_payload": {
                        "version": 1,
                        "payload_type": "conversation",
                        "model": "gemini-3-flash",
                        "prompt": "Replay from db",
                        "context": [],
                        "history": [],
                    }
                },
            )
        )

    asyncio.run(seed())

    async def fake_execute(payload: object, database_obj: object) -> _MockResult:
        _ = (payload, database_obj)
        return _MockResult("DB replay")

    monkeypatch.setattr("ai_pipeline_core.replay.cli._execute_with_database", fake_execute)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 0
    assert (tmp_path / f"node_{str(node_id)[:8]}_replay" / "output.yaml").exists()
```

**Conversationreplay from yaml and override** (`tests/replay/test_cli_usage.py:317`)

```python
def test_ConversationReplay_from_yaml_and_override() -> None:
    yaml_text = yaml.dump({
        "version": 1,
        "payload_type": "conversation",
        "model": "gemini-3-flash",
        "prompt": "Analyze the report.",
        "context": [],
        "history": [],
    })

    replay = ConversationReplay.from_yaml(yaml_text)
    assert replay.model == "gemini-3-flash"

    modified = replay.model_copy(update={"model": "grok-4.1-fast"})
    assert modified.model == "grok-4.1-fast"
    assert modified.prompt == replay.prompt
    assert "grok-4.1-fast" in modified.to_yaml()
```

** infer db path from filesystem database root** (`tests/replay/test_cli_usage.py:337`)

```python
def test__infer_db_path_from_filesystem_database_root(tmp_path: Path) -> None:
    store_dir = tmp_path / "pipeline_output"
    replay_dir = store_dir / "runs" / "20260308_pipeline_abcd1234" / "flows" / "01_flow_1234"
    replay_dir.mkdir(parents=True)
    (store_dir / "blobs").mkdir(parents=True)
    replay_file = replay_dir / "conversation.yaml"
    replay_file.write_text("dummy")

    result = _infer_db_path(replay_file)
    assert result == store_dir
```

**Main run from db missing payload message does not claim failed nodes must succeed first** (`tests/replay/test_cli_usage.py:279`)

```python
def test_main_run_from_db_missing_payload_message_does_not_claim_failed_nodes_must_succeed_first(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.FLOW,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                parent_node_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="FailedFlow",
                sequence_no=1,
                status=NodeStatus.FAILED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={"error_message": "boom"},
            )
        )

    asyncio.run(seed())

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 1
    assert "executed normally" not in capsys.readouterr().err
```

**Main run from db rejects conversation parent node** (`tests/replay/test_cli_usage.py:185`)

```python
def test_main_run_from_db_rejects_conversation_parent_node(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.CONVERSATION,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="ConversationParent",
                sequence_no=0,
                started_at=datetime.now(UTC),
                payload={"turn_count": 1},
            )
        )

    asyncio.run(seed())

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 1
    assert "conversation_turn, task, and flow" in capsys.readouterr().err
```

**Main run from db rejects flow without replay payload** (`tests/replay/test_cli_usage.py:242`)

```python
def test_main_run_from_db_rejects_flow_without_replay_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "bundle"
    database = FilesystemDatabase(db_path)
    node_id = uuid4()
    deployment_id = uuid4()

    async def seed() -> None:
        await database.insert_node(
            ExecutionNode(
                node_id=node_id,
                node_kind=NodeKind.FLOW,
                deployment_id=deployment_id,
                root_deployment_id=deployment_id,
                parent_node_id=deployment_id,
                run_id="run-123",
                run_scope="run-123/scope",
                deployment_name="Replay Test",
                name="CachedFlow",
                sequence_no=1,
                status=NodeStatus.CACHED,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                payload={"skip_reason": "cached result"},
            )
        )

    asyncio.run(seed())

    exit_code = main(["run", "--from-db", str(node_id), "--db-path", str(db_path)])

    assert exit_code == 1
    assert "has no replay payload and cannot be replayed" in capsys.readouterr().err
```

**Flow replay empty flow params default** (`tests/replay/test_e2e_mocked.py:44`)

```python
def test_flow_replay_empty_flow_params_default() -> None:
    """FlowReplay with no flow_params defaults to empty dict (backward compatible)."""
    replay = FlowReplay(function_path="app.flows:MyFlow", run_id="run-1")
    assert replay.flow_params == {}
    yaml_text = replay.to_yaml()
    restored = FlowReplay.from_yaml(yaml_text)
    assert restored.flow_params == {}
```

**Empty documents** (`tests/replay/test_payload_roundtrip.py:191`)

```python
def test_empty_documents(self) -> None:
    payload = FlowReplay(
        function_path="my_pipeline.flows.noop",
        run_id="run-empty",
        documents=[],
        flow_options={},
    )
    yaml_text = payload.to_yaml()
    restored = FlowReplay.from_yaml(yaml_text)
    assert restored == payload
    assert restored.documents == ()
    assert restored.flow_options == {}
```


## Error Examples

**Resolve missing document raises** (`tests/replay/test_resolution.py:57`)

```python
@pytest.mark.asyncio
async def test_resolve_missing_document_raises(self, memory_database: MemoryDatabase) -> None:
    ref = DocumentRef.model_validate({
        "$doc_ref": "Z" * 52,
        "class_name": "ReplayTextDocument",
        "name": "ghost.txt",
    })
    with pytest.raises(FileNotFoundError, match="not found in database"):
        await resolve_document_ref(ref, memory_database)
```

** infer db path missing root raises** (`tests/replay/test_cli_usage.py:349`)

```python
def test__infer_db_path_missing_root_raises(tmp_path: Path) -> None:
    replay_file = tmp_path / "conversation.yaml"
    replay_file.write_text("dummy")

    with pytest.raises(FileNotFoundError):
        _infer_db_path(replay_file)
```

**Fixed tuple replay rejects length mismatch** (`tests/replay/test_deserialize.py:177`)

```python
@pytest.mark.asyncio
async def test_fixed_tuple_replay_rejects_length_mismatch(memory_database) -> None:
    function_path = f"{__name__}:_fixed_tuple_target"
    with pytest.raises(ValueError, match="expects 2 items but replay data has 3"):
        await resolve_task_kwargs(
            function_path,
            {"pair": [1, 2, 3]},
            memory_database,
        )
```

**Resolve missing blob raises** (`tests/replay/test_resolution.py:67`)

```python
@pytest.mark.asyncio
async def test_resolve_missing_blob_raises(self, memory_database: MemoryDatabase) -> None:
    record = DocumentRecord(
        document_sha256="A" * 64,
        content_sha256="B" * 64,
        deployment_id=uuid4(),
        producing_node_id=None,
        document_type="ReplayTextDocument",
        name="broken.txt",
    )
    await memory_database.save_document(record)
    ref = DocumentRef.model_validate({
        "$doc_ref": record.document_sha256,
        "class_name": "ReplayTextDocument",
        "name": record.name,
    })

    with pytest.raises(FileNotFoundError, match="Blob"):
        await resolve_document_ref(ref, memory_database)
```
