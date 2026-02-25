"""Round-trip serialization tests for replay payload types.

Each test creates a payload, serializes to YAML via to_yaml(), deserializes
via from_yaml(), and asserts equality. The replay module does not exist yet —
these tests will fail with ImportError until it is implemented.
"""

from tests.replay.conftest import (
    ReplayArgsModel,
    ReplayBinaryDocument,
    ReplayMode,
    ReplayTextDocument,
    doc_ref_dict,
)

from ai_pipeline_core.replay import ConversationReplay, DocumentRef, FlowReplay, HistoryEntry, TaskReplay


class TestConversationReplayRoundTrip:
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

    def test_multi_turn(
        self,
        sample_text_doc: ReplayTextDocument,
        sample_binary_doc: ReplayBinaryDocument,
    ) -> None:
        text_ref = doc_ref_dict(sample_text_doc)
        binary_ref = doc_ref_dict(sample_binary_doc)

        payload = ConversationReplay(
            model="gpt-5.1",
            prompt="Analyze these documents.",
            response_format="AnalysisResult",
            context=[
                DocumentRef.model_validate(text_ref),
                DocumentRef.model_validate(binary_ref),
            ],
            history=[
                HistoryEntry(type="user_text", text="What do you see?"),
                HistoryEntry(type="response", content="I see a text document and an image."),
                HistoryEntry(
                    type="document",
                    doc_ref=sample_text_doc.sha256,
                    class_name="ReplayTextDocument",
                    name="notes.txt",
                ),
            ],
        )
        yaml_text = payload.to_yaml()
        restored = ConversationReplay.from_yaml(yaml_text)
        assert restored == payload
        assert len(restored.context) == 2
        assert len(restored.history) == 3
        assert restored.response_format == "AnalysisResult"

    def test_all_options(self, sample_text_doc: ReplayTextDocument) -> None:
        payload = ConversationReplay(
            model="gemini-3-pro",
            prompt="Deep analysis.",
            context=[],
            history=[],
            model_options={
                "reasoning_effort": "high",
                "cache_ttl": "300s",
                "retries": 3,
                "retry_delay_seconds": 2.0,
                "timeout": 120,
                "service_tier": "default",
                "search_context_size": "medium",
                "temperature": 0.7,
                "max_completion_tokens": 4096,
            },
            enable_substitutor=False,
            extract_result_tags=True,
            purpose="verification",
        )
        yaml_text = payload.to_yaml()
        restored = ConversationReplay.from_yaml(yaml_text)
        assert restored == payload
        assert restored.enable_substitutor is False
        assert restored.extract_result_tags is True
        assert restored.model_options["reasoning_effort"] == "high"
        assert restored.model_options["temperature"] == 0.7


class TestTaskReplayRoundTrip:
    def test_with_document_and_basemodel(
        self,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        ref = doc_ref_dict(sample_text_doc)
        args_model = ReplayArgsModel(max_items=10, label="test")

        payload = TaskReplay(
            function_path="my_pipeline.tasks.analyze",
            arguments={
                "source": ref,
                "config": args_model.model_dump(),
                "mode": ReplayMode.DEEP.value,
                "count": 42,
            },
        )
        yaml_text = payload.to_yaml()
        restored = TaskReplay.from_yaml(yaml_text)
        assert restored == payload
        assert restored.arguments["source"]["$doc_ref"] == sample_text_doc.sha256
        assert restored.arguments["config"]["max_items"] == 10
        assert restored.arguments["mode"] == "deep"

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


class TestFlowReplayRoundTrip:
    def test_with_documents(
        self,
        sample_text_doc: ReplayTextDocument,
        sample_binary_doc: ReplayBinaryDocument,
    ) -> None:
        text_ref = doc_ref_dict(sample_text_doc)
        binary_ref = doc_ref_dict(sample_binary_doc)

        payload = FlowReplay(
            function_path="my_pipeline.flows.process",
            run_id="run-abc-123",
            documents=[
                DocumentRef.model_validate(text_ref),
                DocumentRef.model_validate(binary_ref),
            ],
            flow_options={"replay_label": "baseline", "replay_mode": "fast"},
        )
        yaml_text = payload.to_yaml()
        restored = FlowReplay.from_yaml(yaml_text)
        assert restored == payload
        assert len(restored.documents) == 2
        assert restored.flow_options["replay_label"] == "baseline"
        assert restored.run_id == "run-abc-123"

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


class TestYamlSentinel:
    def test_yaml_contains_doc_ref_sentinel(
        self,
        sample_text_doc: ReplayTextDocument,
    ) -> None:
        ref = doc_ref_dict(sample_text_doc)
        payload = ConversationReplay(
            model="gemini-3-flash",
            prompt="Test.",
            context=[DocumentRef.model_validate(ref)],
            history=[],
        )
        yaml_text = payload.to_yaml()
        assert "$doc_ref" in yaml_text


class TestVersionDefault:
    def test_version_field_defaults_to_1(self) -> None:
        conv = ConversationReplay(
            model="gemini-3-flash",
            prompt="Test.",
            context=[],
            history=[],
        )
        task = TaskReplay(
            function_path="mod.func",
            arguments={},
        )
        flow = FlowReplay(
            function_path="mod.flow",
            run_id="run-1",
            documents=[],
            flow_options={},
        )
        assert conv.version == 1
        assert task.version == 1
        assert flow.version == 1
