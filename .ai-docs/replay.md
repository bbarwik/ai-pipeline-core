# MODULE: replay
# CLASSES: DocumentRef, HistoryEntry, ConversationReplay, TaskReplay, FlowReplay
# DEPENDS: BaseModel
# PURPOSE: First-class replay system for AI pipeline debugging.
# VERSION: 0.12.0
# AUTO-GENERATED from source code — do not edit. Run: make docs-ai-build

## Imports

```python
from ai_pipeline_core.replay import ConversationReplay, DocumentRef, FlowReplay, HistoryEntry, TaskReplay, infer_store_base
```

## Public API

```python
class DocumentRef(BaseModel):
    """Reference to a document stored in LocalDocumentStore by SHA256.

Documents are not inlined in replay YAML — they are referenced by SHA256 hash
and resolved from the local store at execution time."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    doc_ref: str = Field(alias='$doc_ref')  # Full SHA256 hash of the document
    class_name: str  # Document subclass name for type resolution
    name: str  # Original document name


class HistoryEntry(BaseModel):
    """Single entry in a conversation's message history.

Type determines which fields are populated:
user_text/assistant_text use text, response uses content, document uses doc_ref."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    type: Literal['user_text', 'assistant_text', 'response', 'document']
    text: str | None = None  # For user_text and assistant_text entries
    content: str | None = None  # For response entries
    doc_ref: str | None = Field(None, alias='$doc_ref')  # SHA256 for document entries
    class_name: str | None = None  # Document class for document entries
    name: str | None = None  # Document name for document entries


class ConversationReplay(BaseModel):
    """Replay payload for a Conversation.send() / send_structured() call.

Auto-captured in each span directory as ``conversation.yaml``."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    version: int = 1
    payload_type: Literal['conversation'] = 'conversation'
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

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, store_base: Path) -> Any:
        """Resolve document references and re-execute the LLM call."""
        from ._execute import execute_conversation

        return await execute_conversation(self, store_base)

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


class TaskReplay(BaseModel):
    """Replay payload for a @pipeline_task call.

Auto-captured in each span directory as ``task.yaml``."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    version: int = 1
    payload_type: Literal['pipeline_task'] = 'pipeline_task'
    function_path: str  # "module:qualname" path to the task function
    arguments: dict[str, Any] = {}  # Documents as $doc_ref, BaseModels as dicts, primitives as-is
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, store_base: Path) -> Any:
        """Resolve document references and re-execute the task function."""
        from ._execute import execute_task

        return await execute_task(self, store_base)

    def to_yaml(self) -> str:
        """Serialize to human-editable YAML."""
        return yaml.dump(
            self.model_dump(mode="json", by_alias=True, exclude_defaults=False),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


class FlowReplay(BaseModel):
    """Replay payload for a @pipeline_flow call.

Auto-captured in each span directory as ``flow.yaml``."""
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    version: int = 1
    payload_type: Literal['pipeline_flow'] = 'pipeline_flow'
    function_path: str  # "module:qualname" path to the flow function
    run_id: str  # Unique run identifier for document store scoping
    documents: tuple[DocumentRef, ...] = ()  # Input documents referenced by SHA256
    flow_options: dict[str, Any] = {}  # FlowOptions fields (filtered to known fields at execution)
    original: dict[str, Any] = {}  # Cost/tokens from original execution, for comparison

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Deserialize from YAML text."""
        data = yaml.safe_load(text)
        return cls.model_validate(data)

    async def execute(self, store_base: Path) -> Any:
        """Resolve document references and re-execute the flow function."""
        from ._execute import execute_flow

        return await execute_flow(self, store_base)

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
    """CLI entry point for replay operations.

    ``--import MODULE`` also remaps ``__main__:X`` references to ``MODULE:X`` in replay
    payloads — required when replaying scripts originally run as ``python script.py``.

    Usage:
        ai-replay show conversation.yaml
        ai-replay run conversation.yaml --store ./output
        ai-replay run task.yaml --set model=grok-4.1-fast --import my_app
        ai-replay run flow.yaml --output-dir ./replay_out --import my_app
    """
    parser = argparse.ArgumentParser(prog="ai-replay", description="Execute or inspect replay YAML files")
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Execute a replay YAML file")
    run_parser.add_argument("replay_file", help="Path to replay YAML file (conversation.yaml, task.yaml, flow.yaml)")
    run_parser.add_argument("--store", type=str, help="Override store base path (default: inferred from .trace/ ancestor)")
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
        help="Output directory for results and traces (default: {replay_file_stem}_replay/ next to replay file)",
    )
    run_parser.add_argument(
        "--no-trace",
        action="store_true",
        default=False,
        help="Skip tracing setup — only print result without producing .trace/ output",
    )

    # show
    show_parser = subparsers.add_parser("show", help="Pretty-print a replay YAML file")
    show_parser.add_argument("replay_file", help="Path to replay YAML file")

    args = parser.parse_args(argv)

    handlers: dict[str, Any] = {"run": _cmd_run, "show": _cmd_show}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)

def infer_store_base(replay_file: Path) -> Path:
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

```

## Examples

**Main show conversation** (`tests/replay/test_cli_usage.py:55`)

```python
def test_main_show_conversation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Inspect a replay file without executing it."""
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-pro")

    exit_code = main(["show", str(replay_file)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "ConversationReplay" in output
    assert "gemini-3-pro" in output
```

**Main run with no trace** (`tests/replay/test_cli_usage.py:68`)

```python
def test_main_run_with_no_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute a replay file, saving output.yaml but skipping .trace/ generation."""
    replay_file = _conversation_yaml(tmp_path)
    monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: _MockResult())

    exit_code = main(["run", str(replay_file), "--store", str(tmp_path), "--no-trace"])

    assert exit_code == 0
    output_dir = tmp_path / "conversation_replay"
    assert (output_dir / "output.yaml").exists()
    assert not (output_dir / ".trace").exists()
```

**Main run with set override** (`tests/replay/test_cli_usage.py:82`)

```python
def test_main_run_with_set_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Override model before execution using --set flag."""
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-flash")
    monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: _MockResult())
    monkeypatch.setattr("ai_pipeline_core.replay.cli._init_replay_tracing", lambda _d: None)

    exit_code = main(["run", str(replay_file), "--store", str(tmp_path), "--set", "model=grok-4.1-fast"])

    assert exit_code == 0
```

**Main run with output dir** (`tests/replay/test_cli_usage.py:94`)

```python
def test_main_run_with_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Place replay output in a custom directory using --output-dir."""
    replay_file = _conversation_yaml(tmp_path)
    custom_dir = tmp_path / "my_output"
    monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: _MockResult())
    monkeypatch.setattr("ai_pipeline_core.replay.cli._init_replay_tracing", lambda _d: None)

    exit_code = main(["run", str(replay_file), "--store", str(tmp_path), "--output-dir", str(custom_dir)])

    assert exit_code == 0
    assert (custom_dir / "output.yaml").exists()
```

**Conversationreplay from yaml and override** (`tests/replay/test_cli_usage.py:108`)

```python
def test_ConversationReplay_from_yaml_and_override() -> None:
    """Load a replay payload from YAML, override a field, and serialize back."""
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

    # Override model for re-execution with a different provider
    modified = replay.model_copy(update={"model": "grok-4.1-fast"})
    assert modified.model == "grok-4.1-fast"
    assert modified.prompt == replay.prompt

    # Serialize back to YAML
    output = modified.to_yaml()
    assert "grok-4.1-fast" in output
```

**Infer store base from trace tree** (`tests/replay/test_cli_usage.py:133`)

```python
def test_infer_store_base_from_trace_tree(tmp_path: Path) -> None:
    """Find the store base directory by walking up to the .trace/ parent."""
    store_dir = tmp_path / "pipeline_output"
    trace_dir = store_dir / ".trace" / "001_flow" / "002_task"
    trace_dir.mkdir(parents=True)
    replay_file = trace_dir / "conversation.yaml"
    replay_file.write_text("dummy")

    result = infer_store_base(replay_file)
    assert result == store_dir
```

**Empty documents** (`tests/replay/test_payload_roundtrip.py:167`)

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

**Empty history** (`tests/replay/test_payload_roundtrip.py:20`)

```python
def test_empty_history(self) -> None:
    payload = ConversationReplay(
        model="gemini-3-flash",
        prompt="Summarize the document.",
        context=[],
        history=[],
    )
    yaml_text = payload.to_yaml()
    restored = ConversationReplay.from_yaml(yaml_text)
    assert restored == payload
    assert restored.model == "gemini-3-flash"
    assert restored.prompt == "Summarize the document."
    assert restored.context == ()
    assert restored.history == ()
```

**From nested span dir** (`tests/replay/test_resolution.py:148`)

```python
def test_from_nested_span_dir(self, tmp_path: Path) -> None:
    """Create a deeply nested path under .trace/ and verify infer_store_base
    finds the correct store base (parent of .trace/)."""
    # Layout: tmp_path/output/.trace/spans/task_abc/span_123/
    output_dir = tmp_path / "output"
    trace_dir = output_dir / ".trace"
    nested = trace_dir / "spans" / "task_abc" / "span_123"
    nested.mkdir(parents=True)

    replay_file = nested / "conversation.replay.yaml"
    replay_file.write_text("dummy")

    result = infer_store_base(replay_file)
    assert result == output_dir
```

**Primitives only** (`tests/replay/test_payload_roundtrip.py:123`)

```python
def test_primitives_only(self) -> None:
    payload = TaskReplay(
        function_path="my_pipeline.tasks.simple",
        arguments={
            "name": "hello",
            "count": 7,
            "ratio": 3.14,
            "enabled": True,
        },
    )
    yaml_text = payload.to_yaml()
    restored = TaskReplay.from_yaml(yaml_text)
    assert restored == payload
    assert restored.arguments["name"] == "hello"
    assert restored.arguments["count"] == 7
    assert restored.arguments["ratio"] == 3.14
    assert restored.arguments["enabled"] is True
```


## Error Examples

**Missing trace raises** (`tests/replay/test_resolution.py:163`)

```python
def test_missing_trace_raises(self, tmp_path: Path) -> None:
    """When no .trace/ directory exists in any ancestor, raise FileNotFoundError."""
    deep_path = tmp_path / "some" / "random" / "path"
    deep_path.mkdir(parents=True)
    replay_file = deep_path / "replay.yaml"
    replay_file.write_text("dummy")

    with pytest.raises(FileNotFoundError):
        infer_store_base(replay_file)
```

**Resolve missing document raises** (`tests/replay/test_resolution.py:66`)

```python
def test_resolve_missing_document_raises(self, store_base: Path) -> None:
    store_base.mkdir(parents=True, exist_ok=True)
    fake_sha = "Z" * 52
    ref = DocumentRef.model_validate({
        "$doc_ref": fake_sha,
        "class_name": "ReplayTextDocument",
        "name": "ghost.txt",
    })
    with pytest.raises(FileNotFoundError):
        resolve_document_ref(ref, store_base)
```

**Resolve wrong sha256 same prefix raises** (`tests/replay/test_resolution.py:78`)

```python
@pytest.mark.asyncio
async def test_resolve_wrong_sha256_same_prefix_raises(
    self,
    populated_store: Path,
    sample_text_doc: ReplayTextDocument,
) -> None:
    """Mutate SHA256 after the 6-char prefix so the filesystem file is found
    but the full SHA256 does not match any stored document."""
    real_sha = sample_text_doc.sha256
    # Keep first 6 chars (filename prefix), flip the rest
    mutated_sha = real_sha[:6] + ("A" if real_sha[6] != "A" else "B") + real_sha[7:]
    ref = DocumentRef.model_validate({
        "$doc_ref": mutated_sha,
        "class_name": "ReplayTextDocument",
        "name": "notes.txt",
    })
    with pytest.raises(FileNotFoundError):
        resolve_document_ref(ref, populated_store)
```

**Resolve missing class in registry raises** (`tests/replay/test_resolution.py:96`)

```python
def test_resolve_missing_class_in_registry_raises(self, store_base: Path) -> None:
    """Manually create store files with an unknown class_name and verify
    that resolution fails because the class is not in Document._class_name_registry."""
    fake_class = "NonExistentDocumentXYZ"
    class_dir = store_base / fake_class
    class_dir.mkdir(parents=True, exist_ok=True)

    fake_sha = "ABCDEF" + "0" * 46
    safe_name = _safe_filename("fake.txt", fake_sha)
    content_path = class_dir / safe_name
    meta_path = class_dir / f"{safe_name}.meta.json"

    content_path.write_bytes(b"fake content")
    meta = {
        "name": "fake.txt",
        "document_sha256": fake_sha,
        "content_sha256": "0" * 52,
        "class_name": fake_class,
        "description": "",
        "derived_from": [],
        "triggered_by": [],
        "mime_type": "text/plain",
        "attachments": [],
    }
    meta_path.write_text(json.dumps(meta))

    ref = DocumentRef.model_validate({
        "$doc_ref": fake_sha,
        "class_name": fake_class,
        "name": "fake.txt",
    })
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        resolve_document_ref(ref, store_base)
```
