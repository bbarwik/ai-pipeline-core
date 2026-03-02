"""CLI usage examples for the replay module (ai-replay / python -m ai_pipeline_core.replay).

Tests marked with @pytest.mark.ai_docs appear as priority examples in .ai-docs/replay.md.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from ai_pipeline_core.replay import ConversationReplay
from ai_pipeline_core.replay.types import _infer_store_base
from ai_pipeline_core.replay.cli import main


def _conversation_yaml(tmp_path: Path, **overrides: Any) -> Path:
    """Write a minimal conversation replay YAML and return its path."""
    data: dict[str, Any] = {
        "version": 1,
        "payload_type": "conversation",
        "model": "gemini-3-flash",
        "prompt": "Summarize the document.",
        "context": [],
        "history": [],
    }
    data.update(overrides)
    path = tmp_path / "conversation.yaml"
    path.write_text(yaml.dump(data))
    return path


def _task_yaml(tmp_path: Path) -> Path:
    """Write a minimal task replay YAML and return its path."""
    data = {
        "version": 1,
        "payload_type": "pipeline_task",
        "function_path": "my_app.tasks:extract_insights",
        "arguments": {"query": "market trends"},
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.dump(data))
    return path


class _MockResult:
    def __init__(self, content: str = "LLM response") -> None:
        self.content = content
        self.usage = SimpleNamespace(total_tokens=150, model_dump=lambda: {"total_tokens": 150})
        self.cost = 0.003
        self.parsed = None


@pytest.mark.ai_docs
def test_main_show_conversation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Inspect a replay file without executing it."""
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-pro")

    exit_code = main(["show", str(replay_file)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "ConversationReplay" in output
    assert "gemini-3-pro" in output


@pytest.mark.ai_docs
def test_main_run_with_no_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute a replay file, saving output.yaml but skipping .trace/ generation."""
    replay_file = _conversation_yaml(tmp_path)
    monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: (_coro.close(), _MockResult())[1])

    exit_code = main(["run", str(replay_file), "--store", str(tmp_path), "--no-trace"])

    assert exit_code == 0
    output_dir = tmp_path / "conversation_replay"
    assert (output_dir / "output.yaml").exists()
    assert not (output_dir / ".trace").exists()


@pytest.mark.ai_docs
def test_main_run_with_set_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Override model before execution using --set flag."""
    replay_file = _conversation_yaml(tmp_path, model="gemini-3-flash")
    monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: (_coro.close(), _MockResult())[1])
    monkeypatch.setattr("ai_pipeline_core.replay.cli._init_replay_tracing", lambda _d: None)

    exit_code = main(["run", str(replay_file), "--store", str(tmp_path), "--set", "model=grok-4.1-fast"])

    assert exit_code == 0


@pytest.mark.ai_docs
def test_main_run_with_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Place replay output in a custom directory using --output-dir."""
    replay_file = _conversation_yaml(tmp_path)
    custom_dir = tmp_path / "my_output"
    monkeypatch.setattr("ai_pipeline_core.replay.cli.asyncio.run", lambda _coro: (_coro.close(), _MockResult())[1])
    monkeypatch.setattr("ai_pipeline_core.replay.cli._init_replay_tracing", lambda _d: None)

    exit_code = main(["run", str(replay_file), "--store", str(tmp_path), "--output-dir", str(custom_dir)])

    assert exit_code == 0
    assert (custom_dir / "output.yaml").exists()


@pytest.mark.ai_docs
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


@pytest.mark.ai_docs
def test__infer_store_base_from_trace_tree(tmp_path: Path) -> None:
    """Find the store base directory by walking up to the .trace/ parent."""
    store_dir = tmp_path / "pipeline_output"
    trace_dir = store_dir / ".trace" / "001_flow" / "002_task"
    trace_dir.mkdir(parents=True)
    replay_file = trace_dir / "conversation.yaml"
    replay_file.write_text("dummy")

    result = _infer_store_base(replay_file)
    assert result == store_dir
