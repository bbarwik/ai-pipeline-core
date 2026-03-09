"""Tests for RemoteDeployment class.

Covers __init_subclass__ validation, run() behavior, inline/remote mode,
deployment_class resolution, and edge cases.
"""

# pyright: reportPrivateUsage=false

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ai_pipeline_core import DeploymentResult, Document, FlowOptions
from ai_pipeline_core.database import MemoryDatabase
from ai_pipeline_core.database._filesystem import FilesystemDatabase
from ai_pipeline_core.deployment._helpers import class_name_to_deployment_name, extract_generic_params
from ai_pipeline_core.deployment._resolve import DocumentInput
from ai_pipeline_core.deployment.remote import RemoteDeployment, _derive_remote_run_id
from ai_pipeline_core.pipeline import pipeline_test_context


# ---------------------------------------------------------------------------
# Test document/result types
# ---------------------------------------------------------------------------


class AlphaDoc(Document):
    """First test document type."""


class BetaDoc(Document):
    """Second test document type."""


class GammaDoc(Document):
    """Third test document type."""


class SimpleResult(DeploymentResult):
    report: str = ""


class NestedDocResult(DeploymentResult):
    """Result containing a Document field — tests nested deserialization."""

    output_doc: AlphaDoc | None = None


# ===================================================================
# 1. __init_subclass__ validation
# ===================================================================


class TestNameDerivation:
    def test_auto_derives_kebab_case(self):
        class MyResearchPipeline(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert MyResearchPipeline.name == "my-research-pipeline"

    def test_single_word(self):
        class Research(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert Research.name == "research"

    def test_explicit_name_override(self):
        class CustomNamed(RemoteDeployment[FlowOptions, SimpleResult]):
            name = "my-explicit-name"

        assert CustomNamed.name == "my-explicit-name"

    def test_auto_derived_matches_pipeline_deployment_convention(self):
        class SamplePipeline(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert SamplePipeline.name == class_name_to_deployment_name("SamplePipeline")


class TestGenericExtraction:
    def test_extracts_options_type(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert Foo.options_type is FlowOptions

    def test_extracts_result_type(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert Foo.result_type is SimpleResult

    def test_two_args_returned_by_helper(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        args = extract_generic_params(Foo, RemoteDeployment)
        assert len(args) == 2
        assert args[0] is FlowOptions
        assert args[1] is SimpleResult


class TestTOptionsValidation:
    def test_rejects_non_flow_options(self):
        class NotFlowOptions(BaseModel):
            x: int = 1

        with pytest.raises(TypeError, match="FlowOptions subclass"):

            class Bad(RemoteDeployment[NotFlowOptions, SimpleResult]):  # type: ignore[type-var]
                pass

    def test_accepts_flow_options_subclass(self):
        class CustomOpts(FlowOptions):
            budget: float = 10.0

        class Good(RemoteDeployment[CustomOpts, SimpleResult]):
            pass

        assert Good.options_type is CustomOpts


class TestTResultValidation:
    def test_rejects_non_deployment_result(self):
        class NotAResult(BaseModel):
            x: int = 1

        with pytest.raises(TypeError, match="DeploymentResult subclass"):

            class Bad(RemoteDeployment[FlowOptions, NotAResult]):  # type: ignore[type-var]
                pass

    def test_accepts_deployment_result_subclass(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert Foo.result_type is SimpleResult


class TestMissingGenerics:
    def test_rejects_no_generic_params(self):
        with pytest.raises(TypeError, match="must specify 2 Generic parameters"):

            class Bad(RemoteDeployment):  # type: ignore[type-arg]
                pass

    def test_rejects_one_generic_param(self):
        with pytest.raises(TypeError):

            class Bad(RemoteDeployment[FlowOptions]):  # type: ignore[type-arg]
                pass


# ===================================================================
# 2. deployment_path property
# ===================================================================


class TestDeploymentPath:
    def test_auto_derived(self):
        class AiResearch(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert AiResearch().deployment_path == "ai-research/ai_research"

    def test_explicit_name(self):
        class CustomName(RemoteDeployment[FlowOptions, SimpleResult]):
            name = "my-pipeline"

        assert CustomName().deployment_path == "my-pipeline/my_pipeline"

    def test_path_format_matches_deployer(self):
        class SamplePipeline(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        path = SamplePipeline().deployment_path
        flow_name, deployment_name = path.split("/")
        assert flow_name == "sample-pipeline"
        assert deployment_name == "sample_pipeline"
        assert "-" not in deployment_name


# ===================================================================
# 3. deployment_class and _resolve_deployment_class
# ===================================================================


class TestDeploymentClass:
    def test_default_deployment_class_is_empty(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert Foo.deployment_class == ""

    def test_deployment_class_set_inline(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            deployment_class = "my_module:MyDeployment"

        assert Foo.deployment_class == "my_module:MyDeployment"


class TestResolveDeploymentClass:
    def test_raises_when_deployment_class_not_set(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        with pytest.raises(ValueError, match="deployment_class is not set"):
            Foo()._resolve_deployment_class()

    def test_imports_class_from_module_path(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            deployment_class = "ai_pipeline_core.deployment.base:DeploymentResult"

        resolved = Foo()._resolve_deployment_class()
        assert resolved is DeploymentResult

    def test_raises_on_invalid_module(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            deployment_class = "nonexistent.module:SomeClass"

        with pytest.raises(ModuleNotFoundError):
            Foo()._resolve_deployment_class()

    def test_raises_on_invalid_class_name(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            deployment_class = "ai_pipeline_core.deployment.base:NonexistentClass"

        with pytest.raises(AttributeError):
            Foo()._resolve_deployment_class()


# ===================================================================
# 4. run() — inline vs remote mode detection
# ===================================================================


_EXEC_CTX_PATH = "ai_pipeline_core.deployment.remote.get_execution_context"


class TestInlineModeDetection:
    """run() uses inline mode when the active backend does not support remote execution."""

    async def test_inline_mode_with_memory_database(self):
        """MemoryDatabase triggers inline mode."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        mock_ctx = MagicMock()
        mock_ctx.database = MemoryDatabase()
        mock_ctx.deployment_id = uuid4()
        mock_ctx.root_deployment_id = uuid4()
        mock_ctx.current_node_id = uuid4()
        mock_ctx.deployment_name = "test"
        mock_ctx.run_scope = "test"
        mock_ctx.next_child_sequence.return_value = 0

        foo = Foo()
        with (
            patch(_EXEC_CTX_PATH, return_value=mock_ctx),
            patch.object(Foo, "_run_inline", new_callable=AsyncMock) as mock_inline,
            patch.object(Foo, "_run_remote", new_callable=AsyncMock),
        ):
            mock_inline.return_value = SimpleResult(success=True)
            with pipeline_test_context(run_id="project"):
                await foo.run((), FlowOptions())

        mock_inline.assert_awaited_once()

    async def test_inline_mode_with_filesystem_database(self, tmp_path):
        """FilesystemDatabase triggers inline mode via supports_remote=False."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        mock_ctx = MagicMock()
        mock_ctx.database = FilesystemDatabase(tmp_path)
        mock_ctx.deployment_id = uuid4()
        mock_ctx.root_deployment_id = uuid4()
        mock_ctx.current_node_id = uuid4()
        mock_ctx.deployment_name = "test"
        mock_ctx.run_scope = "test"
        mock_ctx.next_child_sequence.return_value = 0

        foo = Foo()
        with (
            patch(_EXEC_CTX_PATH, return_value=mock_ctx),
            patch.object(Foo, "_run_inline", new_callable=AsyncMock) as mock_inline,
            patch.object(Foo, "_run_remote", new_callable=AsyncMock) as mock_remote,
        ):
            mock_inline.return_value = SimpleResult(success=True)
            with pipeline_test_context(run_id="project"):
                await foo.run((), FlowOptions())

        mock_inline.assert_awaited_once()
        mock_remote.assert_not_awaited()

    async def test_remote_mode_with_remote_capable_database(self):
        """Backends advertising supports_remote=True use Prefect remote execution."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        class RemoteCapableDatabase:
            supports_remote = True

            async def insert_node(self, _node):
                return None

            async def update_node(self, _node_id, **_updates):
                return None

        mock_ctx = MagicMock()
        mock_ctx.database = RemoteCapableDatabase()
        mock_ctx.deployment_id = uuid4()
        mock_ctx.root_deployment_id = uuid4()
        mock_ctx.current_node_id = uuid4()
        mock_ctx.deployment_name = "test"
        mock_ctx.run_scope = "test"
        mock_ctx.next_child_sequence.return_value = 0

        foo = Foo()
        with (
            patch(_EXEC_CTX_PATH, return_value=mock_ctx),
            patch.object(Foo, "_run_inline", new_callable=AsyncMock) as mock_inline,
            patch.object(Foo, "_run_remote", new_callable=AsyncMock) as mock_remote,
        ):
            mock_remote.return_value = SimpleResult(success=True)
            with pipeline_test_context(run_id="project"):
                await foo.run((), FlowOptions())

        mock_remote.assert_awaited_once()
        mock_inline.assert_not_awaited()

    async def test_inline_mode_with_no_context_logs_warning(self, caplog):
        """A remote deployment falling back to inline execution must emit a warning."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        foo = Foo()
        with (
            patch(_EXEC_CTX_PATH, return_value=None),
            patch.object(Foo, "_run_inline", new_callable=AsyncMock) as mock_inline,
        ):
            mock_inline.return_value = SimpleResult(success=True)
            with pipeline_test_context(run_id="project"):
                await foo.run((), FlowOptions())

        assert "inline" in caplog.text.lower()
        mock_inline.assert_awaited_once()

    async def test_inline_mode_with_no_context(self):
        """When no execution context exists, database is None => inline mode."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        foo = Foo()
        with (
            patch(_EXEC_CTX_PATH, return_value=None),
            patch.object(Foo, "_run_inline", new_callable=AsyncMock) as mock_inline,
        ):
            mock_inline.return_value = SimpleResult(success=True)
            with pipeline_test_context(run_id="project"):
                await foo.run((), FlowOptions())

        mock_inline.assert_awaited_once()

    async def test_inline_mode_propagates_publisher_and_parent_execution_id(self):
        """Inline remote execution should preserve lineage and publisher from the active context."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        publisher = MagicMock()
        execution_id = uuid4()
        mock_ctx = MagicMock()
        mock_ctx.database = MemoryDatabase()
        mock_ctx.deployment_id = uuid4()
        mock_ctx.root_deployment_id = uuid4()
        mock_ctx.current_node_id = uuid4()
        mock_ctx.deployment_name = "test"
        mock_ctx.run_scope = "test"
        mock_ctx.publisher = publisher
        mock_ctx.execution_id = execution_id
        mock_ctx.next_child_sequence.return_value = 0

        foo = Foo()
        with (
            patch(_EXEC_CTX_PATH, return_value=mock_ctx),
            patch.object(Foo, "_run_inline", new_callable=AsyncMock) as mock_inline,
        ):
            mock_inline.return_value = SimpleResult(success=True)
            with pipeline_test_context(run_id="project"):
                await foo.run((), FlowOptions())

        assert mock_inline.await_args.kwargs["publisher"] is publisher
        assert mock_inline.await_args.kwargs["parent_execution_id"] == execution_id


# ===================================================================
# 5. run() validation
# ===================================================================


class TestRunValidation:
    async def test_rejects_outside_execution_context(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        with pytest.raises(RuntimeError, match="pipeline_test_context"):
            await Foo().run((), FlowOptions())

    async def test_rejects_invalid_run_id(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        with pytest.raises(ValueError, match="contains invalid characters"):
            with pipeline_test_context(run_id="invalid run id with spaces"):
                await Foo().run((), FlowOptions())

    async def test_rejects_empty_run_id(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        with pytest.raises(ValueError, match="must not be empty"):
            with pipeline_test_context(run_id=""):
                await Foo().run((), FlowOptions())

    async def test_rejects_too_long_derived_run_id(self):
        """A 92+ char base run_id produces a derived run_id exceeding 100 chars."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
        long_run_id = "a" * 92  # 92 + 1 ('-') + 8 (fingerprint) = 101 > 100
        with pytest.raises(ValueError, match="Shorten the base run_id"):
            with pipeline_test_context(run_id=long_run_id):
                await Foo().run((doc,), FlowOptions())


# ===================================================================
# 6. run() result handling
# ===================================================================


_RUN_REMOTE = "ai_pipeline_core.deployment.remote._run_remote_deployment"


class TestRunResultDeserialization:
    async def test_run_remote_uses_document_inputs_parameter_name(self):
        """Remote calls must match the Prefect wrapper's document_inputs parameter."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test input")
        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.return_value = {"success": True, "report": "from dict"}
            await foo._run_remote(
                "project",
                [doc],
                FlowOptions(),
                child_deployment_id=uuid4(),
                root_deployment_id=uuid4(),
                parent_deployment_task_id=uuid4(),
            )

        parameters = mock_prefect.await_args.args[1]
        assert "document_inputs" in parameters
        assert "documents" not in parameters

    async def test_run_remote_normalizes_document_payloads_to_document_input_schema(self):
        """Remote parameters must be serialized as DocumentInput-compatible payloads."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test input")
        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.return_value = {"success": True, "report": "from dict"}
            await foo._run_remote(
                "project",
                [doc],
                FlowOptions(),
                child_deployment_id=uuid4(),
                root_deployment_id=uuid4(),
                parent_deployment_task_id=uuid4(),
            )

        parameters = mock_prefect.await_args.args[1]
        serialized = parameters["document_inputs"][0]
        validated = DocumentInput.model_validate(serialized)
        assert validated.class_name == "AlphaDoc"
        assert "sha256" not in serialized

    async def test_run_remote_uses_parent_execution_id_in_parameters(self):
        """Remote Prefect calls should receive the caller execution_id, not the root deployment id."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        parent_execution_id = uuid4()
        root_deployment_id = uuid4()
        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.return_value = {"success": True, "report": "from dict"}
            await foo._run_remote(
                "project",
                [],
                FlowOptions(),
                child_deployment_id=uuid4(),
                root_deployment_id=root_deployment_id,
                parent_deployment_task_id=uuid4(),
                parent_execution_id=parent_execution_id,
            )

        parameters = mock_prefect.await_args.args[1]
        assert parameters["parent_execution_id"] == str(parent_execution_id)
        assert parameters["root_deployment_id"] == str(root_deployment_id)

    async def test_dict_result_deserialized_in_run_remote(self):
        """_run_remote deserializes dict results via result_type.model_validate."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.return_value = {"success": True, "report": "from dict"}
            result = await foo._run_remote(
                "project",
                [],
                FlowOptions(),
                child_deployment_id=uuid4(),
                root_deployment_id=uuid4(),
                parent_deployment_task_id=uuid4(),
            )

        assert isinstance(result, SimpleResult)
        assert result.report == "from dict"

    async def test_deployment_result_returned_directly_in_run_remote(self):
        """_run_remote returns DeploymentResult instances directly."""

        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        expected = SimpleResult(success=True, report="direct")
        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.return_value = expected
            result = await foo._run_remote(
                "project",
                [],
                FlowOptions(),
                child_deployment_id=uuid4(),
                root_deployment_id=uuid4(),
                parent_deployment_task_id=uuid4(),
            )

        assert result is expected

    async def test_invalid_result_type_raises_in_run_remote(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.return_value = "invalid string"
            with pytest.raises(TypeError, match="unexpected type"):
                await foo._run_remote(
                    "project",
                    [],
                    FlowOptions(),
                    child_deployment_id=uuid4(),
                    root_deployment_id=uuid4(),
                    parent_deployment_task_id=uuid4(),
                )


class TestRunErrorPropagation:
    async def test_error_propagates_from_run_remote(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        foo = Foo()
        with patch(_RUN_REMOTE) as mock_prefect:
            mock_prefect.side_effect = ValueError("deployment not found")
            with pytest.raises(ValueError, match="deployment not found"):
                await foo._run_remote(
                    "project",
                    [],
                    FlowOptions(),
                    child_deployment_id=uuid4(),
                    root_deployment_id=uuid4(),
                    parent_deployment_task_id=uuid4(),
                )


# ===================================================================
# 7. Serialization round-trip
# ===================================================================


class TestSerializationRoundTrip:
    def test_serialize_model_produces_reconstructable_dict(self):
        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test input")
        serialized = doc.serialize_model()

        assert serialized["class_name"] == "AlphaDoc"
        assert serialized["sha256"] == doc.sha256

        restored = AlphaDoc.from_dict(serialized)
        assert isinstance(restored, AlphaDoc)
        assert restored.sha256 == doc.sha256

    def test_from_dict_round_trip_with_known_types(self):
        alpha = AlphaDoc.create_root(name="a.txt", content="alpha content", reason="test input")
        beta = BetaDoc.create_root(name="b.txt", content="beta content", reason="test input")

        type_map = {cls.__name__: cls for cls in [AlphaDoc, BetaDoc]}
        for original in [alpha, beta]:
            serialized = original.serialize_model()
            cls = type_map[serialized["class_name"]]
            restored = cls.from_dict(serialized)
            assert type(restored) is type(original)
            assert restored.sha256 == original.sha256


# ===================================================================
# 8. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_module_level_instantiation_no_event_loop(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        instance = Foo()
        assert instance.name == "foo"
        assert instance.deployment_path == "foo/foo"

    def test_multiple_instances_share_class_state(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        a = Foo()
        b = Foo()
        assert a.name == b.name
        assert a.deployment_path == b.deployment_path
        assert a.result_type is b.result_type


# ===================================================================
# 9. extract_generic_params helper
# ===================================================================


class TestExtractGenericParams:
    def test_two_params_from_remote_deployment(self):
        class Foo(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        result = extract_generic_params(Foo, RemoteDeployment)
        assert len(result) == 2
        assert result[0] is FlowOptions
        assert result[1] is SimpleResult

    def test_no_match_returns_empty_tuple(self):
        result = extract_generic_params(SimpleResult, RemoteDeployment)
        assert result == ()


# ===================================================================
# 10. Deterministic remote run_id
# ===================================================================


class TestDeriveRemoteRunId:
    """Tests for deterministic run_id generation from inputs."""

    def test_deterministic_same_inputs(self):
        """Same documents + options produce same derived run_id."""
        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
        opts = FlowOptions()
        id1 = _derive_remote_run_id("base-run", [doc], opts)
        id2 = _derive_remote_run_id("base-run", [doc], opts)
        assert id1 == id2

    def test_different_docs_different_id(self):
        """Different document content produces different derived run_id."""
        doc_a = AlphaDoc.create_root(name="a.txt", content="aaa", reason="test")
        doc_b = AlphaDoc.create_root(name="b.txt", content="bbb", reason="test")
        opts = FlowOptions()
        id_a = _derive_remote_run_id("base-run", [doc_a], opts)
        id_b = _derive_remote_run_id("base-run", [doc_b], opts)
        assert id_a != id_b

    def test_different_options_different_id(self):
        """Different options produce different derived run_id."""
        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")

        class CustomOptions(FlowOptions):
            budget: float = 10.0

        opts1 = CustomOptions(budget=10.0)
        opts2 = CustomOptions(budget=20.0)
        id1 = _derive_remote_run_id("base-run", [doc], opts1)
        id2 = _derive_remote_run_id("base-run", [doc], opts2)
        assert id1 != id2

    @pytest.mark.ai_docs
    def test_format_starts_with_base_run_id(self):
        """Derived run_id starts with the user's base run_id."""
        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
        derived = _derive_remote_run_id("my-project", [doc], FlowOptions())
        assert derived.startswith("my-project-")

    def test_format_alphanumeric_hyphen(self):
        """Derived run_id matches the allowed pattern."""
        import re

        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
        derived = _derive_remote_run_id("run-123", [doc], FlowOptions())
        assert re.match(r"^[a-zA-Z0-9_-]+$", derived)

    def test_empty_docs_deterministic(self):
        """Empty documents list still produces deterministic run_id."""
        opts = FlowOptions()
        id1 = _derive_remote_run_id("base", [], opts)
        id2 = _derive_remote_run_id("base", [], opts)
        assert id1 == id2
        assert id1.startswith("base-")

    def test_different_base_run_ids(self):
        """Different base run_id produces different derived run_id even with same inputs."""
        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
        opts = FlowOptions()
        id_a = _derive_remote_run_id("project-a", [doc], opts)
        id_b = _derive_remote_run_id("project-b", [doc], opts)
        assert id_a != id_b

    def test_cli_fields_excluded_from_fingerprint(self):
        """CLI-specific fields (working_directory, start, end) don't affect the derived run_id."""
        from pathlib import Path

        class CliOptions(FlowOptions):
            working_directory: Path = Path(".")
            start: str = ""
            budget: float = 10.0

        doc = AlphaDoc.create_root(name="test.txt", content="hello", reason="test")
        id_a = _derive_remote_run_id("run", [doc], CliOptions(working_directory=Path("/home/user/project-a")))
        id_b = _derive_remote_run_id("run", [doc], CliOptions(working_directory=Path("/other/path")))
        assert id_a == id_b, "working_directory is a CLI field and must not affect the fingerprint"

        id_c = _derive_remote_run_id("run", [doc], CliOptions(budget=20.0))
        assert id_a != id_c, "non-CLI fields must affect the fingerprint"


# ===================================================================
# 11. Bug reproduction tests
# ===================================================================


class TestReportedBugs:
    """Tests to prove/disprove bugs found during code review."""

    def test_typing_union_syntax(self):
        """Bug report: typing.Union[A, B] is rejected by _validate_document_type.

        Project bans Union syntax per CLAUDE.md, but should it raise a clear error?
        """
        # RemoteDeployment now takes 2 params (options, result), no doc type param
        # This test documents that the class accepts valid 2-param generics

        class UnionSyntax(RemoteDeployment[FlowOptions, SimpleResult]):
            pass

        assert UnionSyntax.options_type is FlowOptions
        assert UnionSyntax.result_type is SimpleResult
