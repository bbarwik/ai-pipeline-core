"""Tests for replay CLI output saving and tracing (_serialize_result, _write_output, _init_replay_tracing)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import yaml

from pydantic import BaseModel

from ai_pipeline_core.documents import Document
from ai_pipeline_core.replay.cli import _init_replay_tracing, _serialize_result, _write_output


class _MockUsage(BaseModel):
    """Mimics TokenUsage."""

    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


class _MockConversationResult:
    """Mimics a Conversation result with content, usage, cost, and optional parsed."""

    def __init__(self, content: str = "Hello", cost: float = 0.01, parsed: Any = None) -> None:
        self.content = content
        self.usage = _MockUsage()
        self.cost = cost
        self.parsed = parsed


class OutputDocument(Document):
    """Test document for output serialization."""


class TestSerializeResult:
    """Tests for _serialize_result."""

    def test_conversation_result(self) -> None:
        """Conversation results serialize with content, usage, and cost."""
        result = _MockConversationResult(content="LLM response text", cost=0.0042)
        data = _serialize_result(result)

        assert data["type"] == "conversation"
        assert data["content"] == "LLM response text"
        assert data["cost"] == 0.0042
        assert data["usage"]["prompt_tokens"] == 100
        assert data["usage"]["completion_tokens"] == 50
        assert data["usage"]["total_tokens"] == 150
        assert "timestamp" in data

    def test_conversation_with_parsed(self) -> None:
        """Conversation with structured output includes parsed type and data."""

        class _ParsedOutput(BaseModel):
            key: str = "value"

        result = _MockConversationResult(parsed=_ParsedOutput())
        data = _serialize_result(result)

        assert "_ParsedOutput" in data["parsed_type"]
        assert data["parsed"] == {"key": "value"}

    def test_conversation_parsed_without_model_dump(self) -> None:
        """Parsed objects without model_dump fall back to str()."""
        result = _MockConversationResult(parsed="plain string")
        data = _serialize_result(result)

        assert data["parsed_type"] == "str"
        assert data["parsed"] == "plain string"

    def test_document_result(self) -> None:
        """Single Document result serializes with metadata."""
        doc = OutputDocument(name="report.md", content=b"# Report content")
        data = _serialize_result(doc)

        assert data["type"] == "document"
        assert data["class_name"] == "OutputDocument"
        assert data["name"] == "report.md"
        assert data["content_bytes"] == len(b"# Report content")
        assert data["sha256"] == doc.sha256

    def test_document_list_result(self) -> None:
        """List of Documents serializes with per-document metadata."""
        docs = [
            OutputDocument(name="a.md", content=b"AAA"),
            OutputDocument(name="b.md", content=b"BBB"),
        ]
        data = _serialize_result(docs)

        assert data["type"] == "document_list"
        assert data["count"] == 2
        assert len(data["documents"]) == 2
        assert data["documents"][0]["name"] == "a.md"
        assert data["documents"][1]["name"] == "b.md"

    def test_mixed_list_result(self) -> None:
        """Lists with non-Document items use repr fallback."""
        items = [OutputDocument(name="x.md", content=b"X"), "not_a_doc"]
        data = _serialize_result(items)

        assert data["type"] == "document_list"
        assert data["documents"][0]["class_name"] == "OutputDocument"
        assert data["documents"][1]["value"] == "'not_a_doc'"

    def test_unknown_result(self) -> None:
        """Unknown result types use repr."""
        data = _serialize_result(42)

        assert data["type"] == "unknown"
        assert data["value"] == "42"


class TestWriteOutput:
    """Tests for _write_output."""

    def test_writes_to_output_dir(self, tmp_path: Path) -> None:
        """Output YAML file is written inside the output directory."""
        output_dir = tmp_path / "my_replay"
        result = _MockConversationResult(content="Hello world")
        output_path = _write_output(output_dir, result)

        assert output_path == output_dir / "output.yaml"
        assert output_path.exists()

        data = yaml.safe_load(output_path.read_text())
        assert data["type"] == "conversation"
        assert data["content"] == "Hello world"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "replay_output"
        assert not output_dir.exists()

        _write_output(output_dir, 42)
        assert output_dir.exists()
        assert (output_dir / "output.yaml").exists()

    def test_document_list_output(self, tmp_path: Path) -> None:
        """Document list results are properly written."""
        output_dir = tmp_path / "flow_replay"
        docs = [OutputDocument(name="a.md", content=b"A")]
        _write_output(output_dir, docs)

        data = yaml.safe_load((output_dir / "output.yaml").read_text())
        assert data["type"] == "document_list"
        assert data["count"] == 1


class TestInitReplayTracing:
    """Tests for _init_replay_tracing."""

    def test_creates_trace_dir(self, tmp_path: Path) -> None:
        """_init_replay_tracing creates a .trace/ directory inside the output dir."""
        output_dir = tmp_path / "replay_out"
        processor = _init_replay_tracing(output_dir)

        assert (output_dir / ".trace").is_dir()
        if processor is not None:
            processor.shutdown()

    def test_returns_processor(self, tmp_path: Path) -> None:
        """Returns a LocalDebugSpanProcessor (or None if no TracerProvider)."""
        from ai_pipeline_core.observability._debug import LocalDebugSpanProcessor

        output_dir = tmp_path / "replay_out"
        processor = _init_replay_tracing(output_dir)

        # In test environment we may or may not have a real TracerProvider
        assert processor is None or isinstance(processor, LocalDebugSpanProcessor)
        if processor is not None:
            processor.shutdown()

    def test_clears_existing_trace_dir(self, tmp_path: Path) -> None:
        """Existing .trace/ contents are cleared on re-init."""
        output_dir = tmp_path / "replay_out"
        trace_dir = output_dir / ".trace"
        trace_dir.mkdir(parents=True)
        (trace_dir / "old_file.txt").write_text("stale")

        processor = _init_replay_tracing(output_dir)

        assert not (trace_dir / "old_file.txt").exists()
        assert trace_dir.is_dir()
        if processor is not None:
            processor.shutdown()


class TestCmdRunWithTracing:
    """Tests for _cmd_run with the new output dir and tracing flags."""

    def test_no_trace_skips_tracing(self, tmp_path: Path, monkeypatch: MagicMock) -> None:
        """With --no-trace, no .trace/ directory is created in the output dir."""
        from ai_pipeline_core.replay.cli import _cmd_run

        replay_file = tmp_path / "conversation.yaml"
        replay_file.write_text(
            yaml.dump({
                "version": 1,
                "payload_type": "conversation",
                "model": "test-model",
                "prompt": "hello",
                "context": [],
                "history": [],
            })
        )

        mock_result = _MockConversationResult(content="test response")
        monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: mock_result)

        args = SimpleNamespace(
            replay_file=str(replay_file),
            store=str(tmp_path),
            set=None,
            modules=None,
            output_dir=str(tmp_path / "out"),
            no_trace=True,
        )

        result = _cmd_run(args)
        assert result == 0
        # Output YAML is still written
        assert (tmp_path / "out" / "output.yaml").exists()
        # But no .trace/ dir
        assert not (tmp_path / "out" / ".trace").exists()

    def test_output_dir_default_naming(self, tmp_path: Path, monkeypatch: MagicMock) -> None:
        """Default output dir is {stem}_replay/ next to the replay file."""
        from ai_pipeline_core.replay.cli import _cmd_run

        replay_file = tmp_path / "conversation.yaml"
        replay_file.write_text(
            yaml.dump({
                "version": 1,
                "payload_type": "conversation",
                "model": "test-model",
                "prompt": "hello",
                "context": [],
                "history": [],
            })
        )

        mock_result = _MockConversationResult(content="saved response")
        monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: mock_result)
        monkeypatch.setattr("ai_pipeline_core.replay.cli._init_replay_tracing", lambda _d: None)

        args = SimpleNamespace(
            replay_file=str(replay_file),
            store=str(tmp_path),
            set=None,
            modules=None,
            output_dir=None,
            no_trace=False,
        )

        result = _cmd_run(args)
        assert result == 0

        output_dir = tmp_path / "conversation_replay"
        assert (output_dir / "output.yaml").exists()
        data = yaml.safe_load((output_dir / "output.yaml").read_text())
        assert data["content"] == "saved response"

    def test_output_dir_override(self, tmp_path: Path, monkeypatch: MagicMock) -> None:
        """--output-dir places output in the specified directory."""
        from ai_pipeline_core.replay.cli import _cmd_run

        replay_file = tmp_path / "task.yaml"
        replay_file.write_text(
            yaml.dump({
                "version": 1,
                "payload_type": "pipeline_task",
                "function_path": "m:f",
                "arguments": {},
            })
        )

        mock_result = _MockConversationResult(content="custom dir")
        monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: mock_result)
        monkeypatch.setattr("ai_pipeline_core.replay.cli._init_replay_tracing", lambda _d: None)

        custom_dir = tmp_path / "custom_output"
        args = SimpleNamespace(
            replay_file=str(replay_file),
            store=str(tmp_path),
            set=None,
            modules=None,
            output_dir=str(custom_dir),
            no_trace=False,
        )

        result = _cmd_run(args)
        assert result == 0
        assert (custom_dir / "output.yaml").exists()
