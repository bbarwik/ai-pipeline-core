"""Replay YAML round-trip tests for class-based pipeline runtime.

Verifies that replay payloads survive serialize → deserialize → re-serialize
and that all fields are preserved.
"""

from ai_pipeline_core.replay.types import FlowReplay, TaskReplay


def test_task_replay_yaml_round_trip() -> None:
    """TaskReplay survives to_yaml() → from_yaml() with all fields preserved."""
    original = TaskReplay(
        function_path="app.tasks:MyTask",
        arguments={"documents": [{"$doc_ref": "ABC123", "class_name": "InputDoc", "name": "in.txt"}], "label": "test"},
    )
    yaml_text = original.to_yaml()
    restored = TaskReplay.from_yaml(yaml_text)
    assert restored.function_path == original.function_path
    assert restored.arguments == original.arguments
    assert restored.payload_type == "pipeline_task"
    # Re-serialize should produce identical output
    assert restored.to_yaml() == yaml_text


def test_flow_replay_yaml_round_trip() -> None:
    """FlowReplay survives to_yaml() → from_yaml() with all fields including flow_params."""
    original = FlowReplay(
        function_path="app.flows:MyFlow",
        run_id="run-1",
        documents=({"$doc_ref": "DEF456", "class_name": "InputDoc", "name": "data.txt"},),
        flow_options={"custom_field": "value"},
        flow_params={"model_name": "gpt-5", "temperature": 0.7},
    )
    yaml_text = original.to_yaml()
    restored = FlowReplay.from_yaml(yaml_text)
    assert restored.function_path == original.function_path
    assert restored.run_id == original.run_id
    assert restored.flow_options == original.flow_options
    assert restored.flow_params == {"model_name": "gpt-5", "temperature": 0.7}
    assert restored.payload_type == "pipeline_flow"
    assert restored.to_yaml() == yaml_text


def test_flow_replay_empty_flow_params_default() -> None:
    """FlowReplay with no flow_params defaults to empty dict (backward compatible)."""
    replay = FlowReplay(function_path="app.flows:MyFlow", run_id="run-1")
    assert replay.flow_params == {}
    yaml_text = replay.to_yaml()
    restored = FlowReplay.from_yaml(yaml_text)
    assert restored.flow_params == {}
