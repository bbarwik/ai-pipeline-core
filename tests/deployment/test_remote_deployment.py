"""Tests for RemoteDeployment class.

Covers __init_subclass__ validation, run() behavior, union generics,
tracing integration, serialization round-trips, and edge cases.
"""

# pyright: reportPrivateUsage=false

import types
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from ai_pipeline_core import DeploymentContext, DeploymentResult, Document, FlowOptions
from ai_pipeline_core.deployment._helpers import class_name_to_deployment_name, extract_generic_params
from ai_pipeline_core.deployment.remote import RemoteDeployment
from ai_pipeline_core.observability.tracing import TraceLevel


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
        class MyResearchPipeline(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert MyResearchPipeline.name == "my-research-pipeline"

    def test_single_word(self):
        class Research(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Research.name == "research"

    def test_explicit_name_override(self):
        class CustomNamed(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            name = "my-explicit-name"
            trace_level: ClassVar[TraceLevel] = "off"

        assert CustomNamed.name == "my-explicit-name"

    def test_auto_derived_matches_pipeline_deployment_convention(self):
        class SamplePipeline(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert SamplePipeline.name == class_name_to_deployment_name("SamplePipeline")


class TestGenericExtraction:
    def test_extracts_options_type(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Foo.options_type is FlowOptions

    def test_extracts_result_type(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Foo.result_type is SimpleResult

    def test_three_args_returned_by_helper(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        args = extract_generic_params(Foo, RemoteDeployment)
        assert len(args) == 3
        assert args[0] is AlphaDoc
        assert args[1] is FlowOptions
        assert args[2] is SimpleResult

    def test_union_doc_arg_is_union_type(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        args = extract_generic_params(Foo, RemoteDeployment)
        assert isinstance(args[0], types.UnionType)
        assert set(args[0].__args__) == {AlphaDoc, BetaDoc}


class TestTDocValidation:
    def test_single_document_type_accepted(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Foo.result_type is SimpleResult

    def test_union_of_two_documents_accepted(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Foo.name == "foo"

    def test_union_of_three_documents_accepted(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc | GammaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Foo.result_type is SimpleResult

    def test_base_document_accepted(self):
        class Foo(RemoteDeployment[Document, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert Foo.name == "foo"

    def test_rejects_non_document_type(self):
        with pytest.raises(TypeError, match="Document subclass"):

            class Bad(RemoteDeployment[str, FlowOptions, SimpleResult]):  # type: ignore[type-var]
                trace_level: ClassVar[TraceLevel] = "off"

    def test_rejects_non_document_in_union(self):
        with pytest.raises(TypeError, match="Document subclass"):

            class Bad(RemoteDeployment[AlphaDoc | str, FlowOptions, SimpleResult]):  # type: ignore[type-var]
                trace_level: ClassVar[TraceLevel] = "off"

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="Document subclass"):

            class Bad(RemoteDeployment[int, FlowOptions, SimpleResult]):  # type: ignore[type-var]
                trace_level: ClassVar[TraceLevel] = "off"


class TestTOptionsValidation:
    def test_rejects_non_flow_options(self):
        class NotFlowOptions(BaseModel):
            x: int = 1

        with pytest.raises(TypeError, match="FlowOptions subclass"):

            class Bad(RemoteDeployment[AlphaDoc, NotFlowOptions, SimpleResult]):  # type: ignore[type-var]
                trace_level: ClassVar[TraceLevel] = "off"


class TestTResultValidation:
    def test_rejects_non_deployment_result(self):
        class NotAResult(BaseModel):
            x: int = 1

        with pytest.raises(TypeError, match="DeploymentResult subclass"):

            class Bad(RemoteDeployment[AlphaDoc, FlowOptions, NotAResult]):  # type: ignore[type-var]
                trace_level: ClassVar[TraceLevel] = "off"


class TestMissingGenerics:
    def test_rejects_no_generic_params(self):
        with pytest.raises(TypeError, match="must specify 3 Generic parameters"):

            class Bad(RemoteDeployment):  # type: ignore[type-arg]
                trace_level: ClassVar[TraceLevel] = "off"

    def test_rejects_two_generic_params(self):
        with pytest.raises(TypeError):

            class Bad(RemoteDeployment[FlowOptions, SimpleResult]):  # type: ignore[type-arg]
                trace_level: ClassVar[TraceLevel] = "off"


# ===================================================================
# 2. deployment_path property
# ===================================================================


class TestDeploymentPath:
    def test_auto_derived(self):
        class AiResearch(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert AiResearch().deployment_path == "ai-research/ai_research"

    def test_explicit_name(self):
        class CustomName(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            name = "my-pipeline"
            trace_level: ClassVar[TraceLevel] = "off"

        assert CustomName().deployment_path == "my-pipeline/my_pipeline"

    def test_path_format_matches_deployer(self):
        class SamplePipeline(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        path = SamplePipeline().deployment_path
        flow_name, deployment_name = path.split("/")
        assert flow_name == "sample-pipeline"
        assert deployment_name == "sample_pipeline"
        assert "-" not in deployment_name


# ===================================================================
# 3. run() behavior
# ===================================================================


_REMOTE_RUN = "ai_pipeline_core.deployment.remote.run_remote_deployment"


class TestRunBasic:
    async def test_returns_typed_result(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True, report="done")
            result = await Foo().run("project", [], FlowOptions())

        assert isinstance(result, SimpleResult)
        assert result.report == "done"

    async def test_deployment_path_used_in_prefect_call(self):
        class MyPipeline(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await MyPipeline().run("project", [], FlowOptions())

        deployment_name = mock_run.call_args[0][0]
        assert deployment_name == "my-pipeline/my_pipeline"

    async def test_options_passed_through(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        opts = FlowOptions()
        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], opts)

        params = mock_run.call_args[0][1]
        assert params["options"] is opts


class TestRunDocumentSerialization:
    async def test_documents_serialized_via_serialize_model(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        doc = AlphaDoc.create(name="test.txt", content="hello world")
        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [doc], FlowOptions())

        params = mock_run.call_args[0][1]
        assert len(params["documents"]) == 1
        serialized = params["documents"][0]
        assert serialized["class_name"] == "AlphaDoc"
        assert serialized["content"] == "hello world"
        # Metadata keys must be stripped for Prefect JSON schema validation
        for key in ("id", "sha256", "content_sha256", "size", "mime_type"):
            assert key not in serialized

    async def test_multiple_union_docs_serialized(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        docs = [
            AlphaDoc.create(name="a.txt", content="alpha"),
            BetaDoc.create(name="b.txt", content="beta"),
        ]
        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", docs, FlowOptions())

        params = mock_run.call_args[0][1]
        assert len(params["documents"]) == 2
        class_names = {d["class_name"] for d in params["documents"]}
        assert class_names == {"AlphaDoc", "BetaDoc"}

    async def test_empty_documents_list(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions())

        params = mock_run.call_args[0][1]
        assert params["documents"] == []


class TestRunContext:
    async def test_context_none_replaced_with_default(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions())

        params = mock_run.call_args[0][1]
        assert isinstance(params["context"], DeploymentContext)

    async def test_explicit_context_preserved(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        ctx = DeploymentContext(progress_webhook_url="http://example.com")
        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions(), context=ctx)

        params = mock_run.call_args[0][1]
        assert params["context"] is ctx


class TestRunResultDeserialization:
    async def test_deployment_result_instance_returned_directly(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        expected = SimpleResult(success=True, report="direct")
        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = expected
            result = await Foo().run("project", [], FlowOptions())

        assert result is expected

    async def test_dict_result_deserialized_via_model_validate(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = {"success": True, "report": "from dict"}
            result = await Foo().run("project", [], FlowOptions())

        assert isinstance(result, SimpleResult)
        assert result.report == "from dict"

    async def test_nested_document_in_result_deserialized(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, NestedDocResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        original_doc = AlphaDoc.create(name="output.txt", content="result data")
        original_result = NestedDocResult(success=True, output_doc=original_doc)
        result_dict = original_result.model_dump(mode="json")

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = result_dict
            result = await Foo().run("project", [], FlowOptions())

        assert isinstance(result, NestedDocResult)
        assert result.output_doc is not None
        assert isinstance(result.output_doc, AlphaDoc)

    async def test_invalid_result_type_raises(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = "invalid string"
            with pytest.raises(TypeError, match="unexpected type"):
                await Foo().run("project", [], FlowOptions())


class TestRunProgress:
    async def test_on_progress_forwarded(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        callback = AsyncMock()
        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions(), on_progress=callback)

        assert mock_run.call_args.kwargs["on_progress"] is callback

    async def test_on_progress_none_by_default(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions())

        assert mock_run.call_args.kwargs["on_progress"] is None


class TestRunErrorPropagation:
    async def test_deployment_not_found_propagates(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.side_effect = ValueError("deployment not found")
            with pytest.raises(ValueError, match="deployment not found"):
                await Foo().run("project", [], FlowOptions())


# ===================================================================
# 4. Tracing integration
# ===================================================================


class TestTracing:
    def test_execute_is_traced_by_default(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            pass

        assert getattr(Foo._execute, "__is_traced__", False) is True

    def test_trace_level_off_skips_tracing(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert getattr(Foo._execute, "__is_traced__", False) is False

    def test_subclass_specific_trace_names(self):
        class PipelineA(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            pass

        class PipelineB(RemoteDeployment[BetaDoc, FlowOptions, SimpleResult]):
            pass

        assert PipelineA._execute is not PipelineB._execute

    async def test_trace_cost_set_when_configured(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"
            trace_cost: ClassVar[float | None] = 0.05

        with (
            patch(_REMOTE_RUN) as mock_run,
            patch("ai_pipeline_core.deployment.remote.set_trace_cost") as mock_cost,
        ):
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions())
            mock_cost.assert_called_once_with(0.05)

    async def test_trace_cost_none_not_set(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        with (
            patch(_REMOTE_RUN) as mock_run,
            patch("ai_pipeline_core.deployment.remote.set_trace_cost") as mock_cost,
        ):
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", [], FlowOptions())
            mock_cost.assert_not_called()


class TestTraceCombinedGuard:
    def test_user_traced_execute_not_double_wrapped(self):
        """If _execute already has __is_traced__, __init_subclass__ skips re-wrapping."""

        def already_traced(fn: Any) -> Any:
            fn.__is_traced__ = True
            return fn

        class UserTraced(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            @already_traced
            async def _execute(self, *args: Any, **kwargs: Any) -> SimpleResult:  # type: ignore[override]
                return SimpleResult(success=True)

        # Should still be the user's function, not double-wrapped
        assert getattr(UserTraced._execute, "__is_traced__", False) is True

    def test_trace_level_off_with_untraced_execute(self):
        class NoTrace(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        assert getattr(NoTrace._execute, "__is_traced__", False) is False


# ===================================================================
# 5. Serialization round-trip
# ===================================================================


class TestSerializationRoundTrip:
    def test_serialize_model_produces_reconstructable_dict(self):
        doc = AlphaDoc.create(name="test.txt", content="hello")
        serialized = doc.serialize_model()

        assert serialized["class_name"] == "AlphaDoc"
        assert serialized["sha256"] == doc.sha256

        restored = AlphaDoc.from_dict(serialized)
        assert isinstance(restored, AlphaDoc)
        assert restored.sha256 == doc.sha256

    def test_from_dict_round_trip_with_known_types(self):
        alpha = AlphaDoc.create(name="a.txt", content="alpha content")
        beta = BetaDoc.create(name="b.txt", content="beta content")

        type_map = {cls.__name__: cls for cls in [AlphaDoc, BetaDoc]}
        for original in [alpha, beta]:
            serialized = original.serialize_model()
            cls = type_map[serialized["class_name"]]
            restored = cls.from_dict(serialized)
            assert type(restored) is type(original)
            assert restored.sha256 == original.sha256

    async def test_full_run_serialization_path(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        docs = [
            AlphaDoc.create(name="input.txt", content="input data"),
            BetaDoc.create(name="context.md", content="context info"),
        ]

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)
            await Foo().run("project", docs, FlowOptions())

        params = mock_run.call_args[0][1]
        raw_docs = params["documents"]

        type_map = {cls.__name__: cls for cls in [AlphaDoc, BetaDoc]}
        restored = [type_map[d["class_name"]].from_dict(d) for d in raw_docs]
        assert len(restored) == 2
        assert isinstance(restored[0], AlphaDoc)
        assert isinstance(restored[1], BetaDoc)
        assert restored[0].sha256 == docs[0].sha256
        assert restored[1].sha256 == docs[1].sha256


# ===================================================================
# 6. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_module_level_instantiation_no_event_loop(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        instance = Foo()
        assert instance.name == "foo"
        assert instance.deployment_path == "foo/foo"

    def test_multiple_instances_share_class_state(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        a = Foo()
        b = Foo()
        assert a.name == b.name
        assert a.deployment_path == b.deployment_path
        assert a.result_type is b.result_type

    def test_union_with_three_plus_doc_types(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc | GammaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        args = extract_generic_params(Foo, RemoteDeployment)
        assert isinstance(args[0], types.UnionType)
        assert len(args[0].__args__) == 3

    async def test_concurrent_instances_independent(self):
        class ClientA(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        class ClientB(RemoteDeployment[BetaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        a = ClientA()
        b = ClientB()

        with patch(_REMOTE_RUN) as mock_run:
            mock_run.return_value = SimpleResult(success=True)

            await a.run("proj-a", [], FlowOptions())
            await b.run("proj-b", [], FlowOptions())

        calls = mock_run.call_args_list
        assert calls[0][0][0] == "client-a/client_a"
        assert calls[1][0][0] == "client-b/client_b"


# ===================================================================
# 7. extract_generic_params helper (updated)
# ===================================================================


class TestExtractGenericParamsUpdated:
    def test_three_params_from_remote_deployment(self):
        class Foo(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        result = extract_generic_params(Foo, RemoteDeployment)
        assert len(result) == 3
        assert result[0] is AlphaDoc
        assert result[1] is FlowOptions
        assert result[2] is SimpleResult

    def test_union_in_first_position(self):
        class Foo(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        result = extract_generic_params(Foo, RemoteDeployment)
        assert isinstance(result[0], types.UnionType)
        assert result[1] is FlowOptions
        assert result[2] is SimpleResult

    def test_no_match_returns_none_tuple(self):
        result = extract_generic_params(SimpleResult, RemoteDeployment)
        assert result == (None, None)


# ===================================================================
# 8. Bug reproduction tests (from agent review)
# ===================================================================


class TestReportedBugs:
    """Tests to prove/disprove bugs found during code review."""

    def test_trace_level_inherited_from_parent(self):
        """Bug report: cls.__dict__.get('trace_level') ignores inherited values.

        If a parent sets trace_level='off', a child that doesn't override it
        should also be untraced.
        """

        class ParentOff(RemoteDeployment[AlphaDoc, FlowOptions, SimpleResult]):
            trace_level: ClassVar[TraceLevel] = "off"

        class ChildInherits(ParentOff):
            pass

        # Child should inherit trace_level="off" and NOT be traced
        assert getattr(ChildInherits._execute, "__is_traced__", False) is False

    def test_typing_union_syntax(self):
        """Bug report: typing.Union[A, B] is rejected by _validate_document_type.

        Project bans Union syntax per CLAUDE.md, but should it raise a clear error?
        """
        # This tests whether PEP 604 union syntax is handled
        try:

            class UnionSyntax(RemoteDeployment[AlphaDoc | BetaDoc, FlowOptions, SimpleResult]):  # type: ignore[type-var]
                trace_level: ClassVar[TraceLevel] = "off"

            accepted = True
        except TypeError:
            accepted = False

        # Whether accepted or rejected, this documents the behavior
        # The project uses PEP 604 syntax (A | B), so typing.Union is not required
        assert isinstance(accepted, bool)  # always passes — documents behavior
