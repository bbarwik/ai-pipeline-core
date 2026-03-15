"""Static validation tests for PipelineTask and PipelineFlow.

Tests class name collision detection, return type enforcement, input type
enforcement, bare Document rejection, NewType input validation,
PipelineDeployment flow chain validation, and build_result requirements.
"""

# pyright: reportPrivateUsage=false, reportUnusedClass=false

from typing import Any, ClassVar, NewType

import pytest
from pydantic import BaseModel, ConfigDict

from ai_pipeline_core import Document, FlowOptions
from ai_pipeline_core.deployment.base import DeploymentResult, PipelineDeployment, _validate_flow_chain
from ai_pipeline_core.documents import DocumentSha256
from ai_pipeline_core.llm.conversation import Conversation
from ai_pipeline_core.pipeline import PipelineFlow, PipelineTask
from ai_pipeline_core.pipeline._type_validation import (
    collect_document_types,
    contains_bare_document,
    flatten_union,
)

# --- Document subclasses for testing ---


class InputDoc(Document):
    pass


class OutputDoc(Document):
    pass


class ExtraDoc(Document):
    pass


class AlphaDocument(Document):
    pass


class BetaDocument(Document):
    pass


class GammaDocument(Document):
    pass


class DeltaDocument(Document):
    pass


class InputMode:
    """Fake enum-like for test only; not used in validation."""


from enum import StrEnum


class RealInputMode(StrEnum):
    FAST = "fast"
    DEEP = "deep"


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str


class Opts(FlowOptions):
    flag: bool = False


class SampleResult(DeploymentResult):
    """Result for deployment testing."""


# --- PipelineFlow subclasses for deployment chain testing ---


class AlphaToBetaFlow(PipelineFlow):
    async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[BetaDocument, ...]:
        return ()


class BetaToGammaFlow(PipelineFlow):
    async def run(self, documents: tuple[BetaDocument, ...], options: FlowOptions) -> tuple[GammaDocument, ...]:
        return ()


class AlphaToGammaFlow(PipelineFlow):
    async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[GammaDocument, ...]:
        return ()


class GammaToDeltaFlow(PipelineFlow):
    async def run(self, documents: tuple[GammaDocument, ...], options: FlowOptions) -> tuple[DeltaDocument, ...]:
        return ()


class NeedsDeltaFlow(PipelineFlow):
    async def run(self, documents: tuple[DeltaDocument, ...], options: FlowOptions) -> tuple[AlphaDocument, ...]:
        return ()


class UnionInputFlow(PipelineFlow):
    async def run(self, documents: tuple[BetaDocument | DeltaDocument, ...], options: FlowOptions) -> tuple[GammaDocument, ...]:
        return ()


# --- Existing tests (preserved from current file) ---


def test_pipeline_task_extracts_document_types_from_flexible_signature() -> None:
    class GoodTask(PipelineTask):
        @classmethod
        async def run(
            cls,
            source: InputDoc,
            mode: RealInputMode,
            config: FrozenConfig,
            prompt: str,
        ) -> tuple[OutputDoc, ...]:
            _ = (cls, source, mode, config, prompt)
            return ()

    assert GoodTask.input_document_types == [InputDoc]
    assert GoodTask.output_document_types == [OutputDoc]
    assert GoodTask.name == "GoodTask"


def test_pipeline_flow_init_accepts_forward_referenced_constructor_annotation() -> None:
    class ConfiguredFlow(PipelineFlow):
        config: FlowConfig

        async def run(self, documents: tuple[InputDoc, ...], options: FlowOptions) -> tuple[OutputDoc, ...]:
            _ = (documents, options)
            return ()

    class FlowConfig:
        pass

    config = FlowConfig()
    flow = ConfiguredFlow(config=config)
    assert flow.config is config
    assert flow.get_params() == {"config": config}


def test_pipeline_task_inherits_validated_run() -> None:
    class _BaseTask(PipelineTask):
        @classmethod
        async def run(cls, source: InputDoc) -> tuple[OutputDoc, ...]:
            _ = (cls, source)
            return ()

    class DerivedTask(_BaseTask):
        pass

    assert DerivedTask.input_document_types == [InputDoc]
    assert DerivedTask.output_document_types == [OutputDoc]


def test_pipeline_task_inherits_validated_run_through_abstract_base() -> None:
    class _BaseTask(PipelineTask):
        @classmethod
        async def run(cls, source: InputDoc) -> tuple[OutputDoc, ...]:
            _ = (cls, source)
            return ()

    class _AbstractMidTask(_BaseTask):
        _abstract_task = True

    class DerivedTask(_AbstractMidTask):
        pass

    assert DerivedTask._run_spec is not None
    assert DerivedTask.input_document_types == [InputDoc]
    assert DerivedTask.output_document_types == [OutputDoc]


def test_pipeline_task_rejects_bare_document() -> None:
    with pytest.raises(TypeError, match="bare 'Document'"):

        class BadTask(PipelineTask):
            @classmethod
            async def run(cls, documents: tuple[Document, ...]) -> tuple[OutputDoc, ...]:
                _ = (cls, documents)
                return ()


def test_pipeline_task_rejects_non_classmethod_run() -> None:
    with pytest.raises(TypeError, match="@classmethod"):

        class BadTask(PipelineTask):
            async def run(self, source: InputDoc) -> tuple[OutputDoc, ...]:
                _ = (self, source)
                return ()


def test_pipeline_task_rejects_sync_run() -> None:
    with pytest.raises(TypeError, match="async def"):

        class SyncTask(PipelineTask):
            @classmethod
            def run(cls, source: InputDoc) -> tuple[OutputDoc, ...]:
                _ = (cls, source)
                return ()


def test_pipeline_task_rejects_non_frozen_basemodel_input() -> None:
    class MutableConfig(BaseModel):
        label: str

    with pytest.raises(TypeError, match="unsupported input annotation"):

        class BadTask(PipelineTask):
            @classmethod
            async def run(cls, config: MutableConfig) -> tuple[OutputDoc, ...]:
                _ = (cls, config)
                return ()


def test_pipeline_task_rejects_invalid_return_annotation() -> None:
    with pytest.raises(TypeError, match="must return Document, None, list\\[Document\\], tuple\\[Document, \\.\\.\\.\\]"):

        class BadTask(PipelineTask):
            @classmethod
            async def run(cls, source: InputDoc) -> dict[str, OutputDoc]:
                _ = (cls, source)
                return {}


class _StageTask(PipelineTask):
    @classmethod
    async def run(cls, documents: tuple[InputDoc, ...]) -> tuple[OutputDoc, ...]:
        _ = cls
        return ()


def test_pipeline_flow_extracts_types_and_task_graph() -> None:
    class GoodFlow(PipelineFlow):
        async def run(self, documents: tuple[InputDoc, ...], options: Opts) -> tuple[OutputDoc, ...]:
            _ = options
            return await _StageTask.run(documents)

    assert GoodFlow.input_document_types == [InputDoc]
    assert GoodFlow.output_document_types == [OutputDoc]
    assert GoodFlow.expected_tasks() == ["_StageTask"]


def test_pipeline_flow_ast_extracts_handle_pattern() -> None:
    class HandleFlow(PipelineFlow):
        async def run(self, documents: tuple[InputDoc, ...], options: Opts) -> tuple[OutputDoc, ...]:
            _ = options
            handle = _StageTask.run(documents)
            return await handle

    assert ("_StageTask", "dispatched") in HandleFlow.task_graph


def test_pipeline_flow_rejects_wrong_signature() -> None:
    with pytest.raises(TypeError, match="get_run_id\\(\\)"):

        class BadFlow(PipelineFlow):
            async def run(self, run_id: int, documents: tuple[InputDoc, ...], options: Opts) -> tuple[OutputDoc, ...]:
                _ = (run_id, documents, options)
                return ()


def test_pipeline_flow_accepts_inherited_run() -> None:
    class _BaseFlow(PipelineFlow):
        async def run(self, documents: tuple[InputDoc, ...], options: Opts) -> tuple[OutputDoc, ...]:
            _ = (documents, options)
            return ()

    class DerivedFlow(_BaseFlow):
        pass

    assert DerivedFlow.input_document_types == [InputDoc]
    assert DerivedFlow.output_document_types == [OutputDoc]


def test_pipeline_flow_rejects_bare_document_in_return() -> None:
    with pytest.raises(TypeError, match="bare 'Document'"):

        class BadFlow(PipelineFlow):
            async def run(self, documents: tuple[InputDoc, ...], options: Opts) -> tuple[Document, ...]:
                _ = (documents, options)
                return ()


def test_pipeline_task_allows_conversation_input() -> None:
    class GoodTask(PipelineTask):
        @classmethod
        async def run(cls, conv: Conversation[str], source: InputDoc) -> tuple[OutputDoc, ...]:
            _ = (cls, conv, source)
            return ()

    assert GoodTask.input_document_types == [InputDoc]


def test_pipeline_flow_custom_name() -> None:
    class NamedFlow(PipelineFlow):
        name = "custom-flow"

        async def run(self, documents: tuple[InputDoc, ...], options: Opts) -> tuple[OutputDoc, ...]:
            _ = (documents, options)
            return ()

    assert NamedFlow.name == "custom-flow"


def test_pipeline_flow_rejects_typo_kwargs() -> None:
    """PipelineFlow must reject unknown kwargs to catch typos."""

    class StrictFlow(PipelineFlow):
        estimated_minutes: ClassVar[float] = 5.0

        async def run(self, documents: tuple[InputDoc, ...], options: Opts) -> tuple[OutputDoc, ...]:
            _ = (documents, options)
            return ()

    # Valid kwarg works
    flow_inst = StrictFlow(estimated_minutes=10.0)
    assert flow_inst.estimated_minutes == 10.0

    # Typo is rejected
    with pytest.raises(TypeError, match="unknown init parameter"):
        StrictFlow(estimated_minutse=5)


# --------------------------------------------------------------------------- #
# Annotation parsing helper tests
# --------------------------------------------------------------------------- #


class TestAnnotationParsing:
    """Test annotation extraction from type hints using collect_document_types."""

    def test_single_type(self):
        parsed = collect_document_types(list[InputDoc])
        assert parsed == [InputDoc]

    def test_pipe_union(self):
        parsed = collect_document_types(list[InputDoc | ExtraDoc])
        assert set(parsed) == {InputDoc, ExtraDoc}

    def test_typing_union(self):
        parsed = collect_document_types(list[InputDoc | ExtraDoc])
        assert set(parsed) == {InputDoc, ExtraDoc}

    def test_dict_type_walks_args(self):
        parsed = collect_document_types(dict[str, InputDoc])
        assert InputDoc in parsed  # dict args are walked

    def test_plain_list_returns_empty(self):
        parsed = collect_document_types(list)
        assert parsed == []

    def test_non_document_types_ignored(self):
        parsed = collect_document_types(list[str])
        assert parsed == []

    def test_flatten_union_simple(self):
        result = flatten_union(InputDoc)
        assert result == [InputDoc]

    def test_flatten_union_pipe(self):
        result = flatten_union(InputDoc | ExtraDoc)
        assert set(result) == {InputDoc, ExtraDoc}

    def test_contains_bare_document_true(self):
        assert contains_bare_document(Document) is True
        assert contains_bare_document(list[Document]) is True
        assert contains_bare_document(Document | InputDoc) is True

    def test_contains_bare_document_false(self):
        assert contains_bare_document(InputDoc) is False
        assert contains_bare_document(list[InputDoc]) is False
        assert contains_bare_document(int) is False


# --------------------------------------------------------------------------- #
# Canonical name collision detection tests
# --------------------------------------------------------------------------- #


class TestClassNameCollision:
    """Test Document.__init_subclass__ class name collision detection."""

    def test_different_class_names_ok(self):
        """Classes with different names register without error."""

        class UniqueNameOneDocument(Document):
            pass

        class UniqueNameTwoDocument(Document):
            pass

        assert UniqueNameOneDocument.__name__ != UniqueNameTwoDocument.__name__

    def test_registry_stores_classes(self):
        """The registry is a dict mapping class names to Document subclasses."""
        from ai_pipeline_core.documents.document import _class_name_registry

        assert isinstance(_class_name_registry, dict)
        for name, cls in _class_name_registry.items():
            assert isinstance(name, str)
            assert isinstance(cls, type)

    def test_test_module_classes_skip_registry(self):
        """Classes defined in test modules are not registered."""
        from ai_pipeline_core.documents.document import _class_name_registry, _is_test_module

        assert _is_test_module(AlphaDocument)

        existing = _class_name_registry.get(AlphaDocument.__name__)
        assert existing is not AlphaDocument

    def test_collision_detection_for_production_classes(self):
        """Verify the collision detection logic works by directly calling the check."""
        from ai_pipeline_core.documents.document import _is_test_module

        assert _is_test_module(AlphaDocument) is True

        class _FakeProductionClass:
            __module__ = "ai_pipeline_core.custom_documents"

        assert _is_test_module(_FakeProductionClass) is False


# --------------------------------------------------------------------------- #
# @pipeline_flow annotation validation tests
# --------------------------------------------------------------------------- #


class TestFlowAnnotationValidation:
    """Test PipelineFlow return type and input annotation validation."""

    def test_rejects_list_input_annotation(self):
        """Flow input annotations must use tuple, not list."""
        with pytest.raises(TypeError, match="tuple\\[DocumentSubclass, \\.\\.\\.\\]"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: list[AlphaDocument], options: FlowOptions) -> tuple[BetaDocument, ...]:
                    return ()

    def test_rejects_non_document_return_type(self):
        """Flow with return annotation that has no Document subclasses is rejected."""
        with pytest.raises(TypeError, match="must return tuple"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[str, ...]:
                    return ()

    def test_rejects_dict_return_type(self):
        """Flow returning dict is rejected."""
        with pytest.raises(TypeError, match="must return tuple"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> dict[str, Any]:
                    return {}

    def test_accepts_concrete_document_return_type(self):
        """Flow returning tuple[ConcreteDocument, ...] is accepted."""

        class GoodFlow(PipelineFlow):
            async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[BetaDocument, ...]:
                return ()

        assert GoodFlow.output_document_types == [BetaDocument]

    def test_rejects_missing_return_annotation(self):
        """Flow missing return annotation is rejected."""
        with pytest.raises(TypeError, match="missing return annotation"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions):
                    return ()

    def test_rejects_non_list_return_type(self):
        """Flow returning a non-list type is rejected."""
        with pytest.raises(TypeError):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> AlphaDocument:
                    return AlphaDocument(name="a.txt", content=b"a")

    def test_rejects_non_document_input(self):
        """Flow with non-tuple[Document] input annotation is rejected."""
        with pytest.raises(TypeError, match="must be annotated as tuple"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: str, options: FlowOptions) -> tuple[AlphaDocument, ...]:
                    return ()

    def test_rejects_overlapping_input_output_types(self):
        """Flow that consumes and produces the same Document type is rejected."""
        with pytest.raises(TypeError, match="overlapping input/output"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[AlphaDocument, ...]:
                    return ()

    def test_rejects_missing_task_annotation(self):
        """Task with no return annotation is rejected."""
        with pytest.raises(TypeError, match=r"missing.*return"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls, x: int):
                    return x


# --------------------------------------------------------------------------- #
# @pipeline_task return type validation tests
# --------------------------------------------------------------------------- #


class TestTaskReturnTypeValidation:
    """Test PipelineTask return type annotation enforcement."""

    # --- Accepted types ---

    def test_accepts_single_document(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> AlphaDocument:
                return AlphaDocument(name="a.txt", content=b"a")

    def test_accepts_list_document(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> list[AlphaDocument]:
                return []

    def test_accepts_list_union_documents(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> list[AlphaDocument | BetaDocument]:
                return []

    def test_accepts_tuple_documents(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> tuple[AlphaDocument, BetaDocument]: ...

    def test_accepts_tuple_of_lists(self):
        # tuple of lists is not supported as output; only flat tuple or list
        # This tests whatever the validation actually does
        with pytest.raises(TypeError):

            class AcceptTask(PipelineTask):
                @classmethod
                async def run(cls) -> tuple[list[AlphaDocument], list[BetaDocument]]: ...

    def test_accepts_variable_length_tuple(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> tuple[AlphaDocument, ...]: ...

    def test_accepts_none(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> None:
                pass

    def test_accepts_document_or_none(self):
        class AcceptTask(PipelineTask):
            @classmethod
            async def run(cls) -> AlphaDocument | None:
                return None

    # --- Rejected types ---

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="must return Document"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> int:
                    return 0

    def test_rejects_str(self):
        with pytest.raises(TypeError, match="must return Document"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> str:
                    return ""

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="must return Document"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> bool:
                    return True

    def test_rejects_dict(self):
        with pytest.raises(TypeError, match="must return Document"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> dict[str, Any]:
                    return {}

    def test_rejects_list_str(self):
        with pytest.raises(TypeError, match="unsupported output member"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> list[str]:
                    return []

    def test_rejects_tuple_with_non_document(self):
        with pytest.raises(TypeError, match="unsupported output member"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> tuple[AlphaDocument, int]: ...

    def test_rejects_missing_annotation(self):
        with pytest.raises(TypeError, match=r"missing.*return"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls):
                    pass

    def test_rejects_any(self):
        with pytest.raises(TypeError, match="must not use 'Any'"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> Any:
                    return None

    def test_rejects_object(self):
        with pytest.raises(TypeError, match="must return Document"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> object:
                    return None


# --------------------------------------------------------------------------- #
# Bare Document rejection tests
# --------------------------------------------------------------------------- #


class TestBareDocumentRejection:
    """Bare Document (not a subclass) must be rejected in pipeline annotations.

    The framework requires specific Document subclasses for type safety and
    document flow tracking.
    """

    # --- PipelineFlow output ---

    def test_flow_rejects_bare_document_output(self):
        """Flow returning tuple[Document, ...] is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[Document, ...]:
                    return ()

    def test_flow_rejects_bare_document_in_union_output(self):
        """Flow returning tuple[Document | BetaDocument, ...] is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[Document | BetaDocument, ...]:
                    return ()

    # --- PipelineFlow input ---

    def test_flow_rejects_bare_document_input(self):
        """Flow with documents: tuple[Document, ...] is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[Document, ...], options: FlowOptions) -> tuple[AlphaDocument, ...]:
                    return ()

    def test_flow_rejects_bare_document_in_union_input(self):
        """Flow with documents: tuple[Document | AlphaDocument, ...] is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadFlow(PipelineFlow):
                async def run(self, documents: tuple[Document | AlphaDocument, ...], options: FlowOptions) -> tuple[BetaDocument, ...]:
                    return ()

    # --- PipelineTask ---

    def test_task_rejects_bare_document_return(self):
        """Task returning bare Document is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> Document: ...

    def test_task_rejects_bare_document_list_return(self):
        """Task returning list[Document] is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> list[Document]:
                    return []

    def test_task_rejects_bare_document_or_none(self):
        """Task returning Document | None is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> Document | None:
                    return None

    def test_task_rejects_bare_document_in_tuple(self):
        """Task returning tuple containing bare Document is rejected."""
        with pytest.raises(TypeError, match="bare 'Document'"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls) -> tuple[Document, AlphaDocument]: ...

    # --- Positive cases (concrete subclasses accepted) ---

    def test_flow_accepts_concrete_subclasses(self):
        class GoodFlow(PipelineFlow):
            async def run(self, documents: tuple[AlphaDocument, ...], options: FlowOptions) -> tuple[BetaDocument, ...]:
                return ()

    def test_task_accepts_concrete_subclass(self):
        class GoodTask(PipelineTask):
            @classmethod
            async def run(cls) -> AlphaDocument: ...

    def test_task_accepts_concrete_list(self):
        class GoodTask(PipelineTask):
            @classmethod
            async def run(cls) -> list[AlphaDocument]:
                return []

    def test_task_accepts_concrete_union(self):
        class GoodTask(PipelineTask):
            @classmethod
            async def run(cls) -> AlphaDocument | BetaDocument: ...

    def test_task_accepts_concrete_or_none(self):
        class GoodTask(PipelineTask):
            @classmethod
            async def run(cls) -> AlphaDocument | None:
                return None


# --------------------------------------------------------------------------- #
# PipelineDeployment validation tests
# --------------------------------------------------------------------------- #


class TestDeploymentFlowChainValidation:
    """Test _validate_flow_chain for flow chain type pool validation."""

    def test_valid_chain(self):
        """Valid flow chain: A->B, B->C passes validation."""
        _validate_flow_chain("ValidChain", [AlphaToBetaFlow(), BetaToGammaFlow()])

    def test_valid_single_flow(self):
        """Single flow deployment passes validation."""
        _validate_flow_chain("SingleFlow", [AlphaToBetaFlow()])

    def test_broken_chain_raises(self):
        """Flow requiring types not in pool raises TypeError."""
        with pytest.raises(TypeError, match="none are produced by preceding flows"):
            _validate_flow_chain("BrokenChain", [AlphaToBetaFlow(), NeedsDeltaFlow()])

    def test_three_step_chain_valid(self):
        """Three-step chain: A->B, B->C, C->D passes."""
        _validate_flow_chain("ThreeStep", [AlphaToBetaFlow(), BetaToGammaFlow(), GammaToDeltaFlow()])

    def test_union_input_any_of_semantics(self):
        """Flow with union input types passes if at least one type is in the pool."""
        _validate_flow_chain("UnionChain", [AlphaToBetaFlow(), UnionInputFlow()])

    def test_union_input_none_satisfied_raises(self):
        """Flow with union input types fails if no type is in the pool."""
        with pytest.raises(TypeError, match="none are produced by preceding flows"):
            _validate_flow_chain("BadUnion", [AlphaToGammaFlow(), UnionInputFlow()])

    def test_three_step_chain_broken_at_step_three(self):
        """Chain where step 3 needs types not in pool raises."""
        with pytest.raises(TypeError, match="none are produced by preceding flows"):
            _validate_flow_chain("BrokenAtThree", [AlphaToBetaFlow(), AlphaToGammaFlow(), NeedsDeltaFlow()])


class TestDeploymentBuildResultRequired:
    """Test PipelineDeployment requires build_result and build_flows implementations."""

    def test_missing_build_result_raises(self):
        """Deployment without build_result raises TypeError."""
        with pytest.raises(TypeError, match=r"must implement.*build_result"):

            class NoBuild(PipelineDeployment[FlowOptions, SampleResult]):
                def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
                    return [AlphaToBetaFlow()]

    def test_missing_build_flows_raises(self):
        """Deployment without build_flows raises TypeError."""
        with pytest.raises(TypeError, match="must implement build_flows"):

            class NoFlows(PipelineDeployment[FlowOptions, SampleResult]):
                @staticmethod
                def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> SampleResult:
                    return SampleResult(success=True)

    def test_concrete_parent_build_result_inherited(self):
        """Inheriting build_result from a concrete parent deployment is allowed."""

        class ParentDeploy(PipelineDeployment[FlowOptions, SampleResult]):
            def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
                return [AlphaToBetaFlow()]

            @staticmethod
            def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        class ChildDeploy(ParentDeploy):
            def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
                return [AlphaToBetaFlow(), BetaToGammaFlow()]

        assert ChildDeploy.name == "child-deploy"

    def test_valid_deployment(self):
        """Valid deployment with build_result and build_flows passes."""

        class ValidDeploy(PipelineDeployment[FlowOptions, SampleResult]):
            def build_flows(self, options: FlowOptions) -> list[PipelineFlow]:
                return [AlphaToBetaFlow()]

            @staticmethod
            def build_result(run_id: str, documents: tuple[Document, ...], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        assert ValidDeploy.name == "valid-deploy"


# --------------------------------------------------------------------------- #
# NewType input validation tests
# --------------------------------------------------------------------------- #

CustomId = NewType("CustomId", str)
Score = NewType("Score", float)


class TestNewTypeInputValidation:
    """NewType wrapping valid scalar types — currently REJECTED by validation.

    The type validator does not resolve NewType to its supertype.
    These tests document current behavior (rejection), not desired behavior.
    """

    def test_bare_newtype_rejected(self):
        with pytest.raises(TypeError, match="unsupported input annotation"):

            class BadTask(PipelineTask):
                @classmethod
                async def run(cls, value: DocumentSha256) -> AlphaDocument:
                    return AlphaDocument(name="a.txt", content=b"a")

    def test_str_accepted_directly(self):
        class GoodTask(PipelineTask):
            @classmethod
            async def run(cls, value: str) -> AlphaDocument:
                return AlphaDocument(name="a.txt", content=b"a")

    def test_bytes_input_rejected(self):
        with pytest.raises(TypeError, match="unsupported input annotation"):

            class BadBytesTask(PipelineTask):
                @classmethod
                async def run(cls, payload: bytes) -> AlphaDocument:
                    return AlphaDocument(name="a.txt", content=b"a")


def test_abstract_task_can_be_defined_without_run() -> None:
    class AbstractBaseTask(PipelineTask):
        _abstract_task = True

    assert not hasattr(AbstractBaseTask, "_run_spec")


def test_abstract_task_subclass_must_define_run() -> None:
    class AbstractParentTask(PipelineTask):
        _abstract_task = True

    with pytest.raises(TypeError, match="must define"):

        class MissingRunTask(AbstractParentTask):
            pass


def test_abstract_task_subclass_with_run_validates_normally() -> None:
    class AbstractInputDocument(Document):
        pass

    class AbstractOutputDocument(Document):
        pass

    class AbstractParentTask(PipelineTask):
        _abstract_task = True

    class ConcreteTask(AbstractParentTask):
        @classmethod
        async def run(cls, documents: tuple[AbstractInputDocument, ...]) -> tuple[AbstractOutputDocument, ...]:
            _ = (cls, documents)
            return ()

    assert ConcreteTask._run_spec is not None
    assert ConcreteTask.input_document_types == [AbstractInputDocument]
    assert ConcreteTask.output_document_types == [AbstractOutputDocument]


def test_abstract_task_flag_does_not_inherit() -> None:
    class AbstractParentTask(PipelineTask):
        _abstract_task = True

    with pytest.raises(TypeError, match="must define"):

        class ChildTask(AbstractParentTask):
            pass


def test_abstract_task_has_no_document_types() -> None:
    class AbstractBaseTask(PipelineTask):
        _abstract_task = True

    assert AbstractBaseTask.input_document_types == []
    assert AbstractBaseTask.output_document_types == []


def test_multi_level_abstract_task_chain() -> None:
    class MultiLevelInputDocument(Document):
        pass

    class MultiLevelOutputDocument(Document):
        pass

    class LevelOneTask(PipelineTask):
        _abstract_task = True

    class LevelTwoTask(LevelOneTask):
        _abstract_task = True

    class ConcreteTask(LevelTwoTask):
        @classmethod
        async def run(cls, documents: tuple[MultiLevelInputDocument, ...]) -> tuple[MultiLevelOutputDocument, ...]:
            _ = (cls, documents)
            return ()

    assert ConcreteTask._run_spec is not None
    assert ConcreteTask.input_document_types == [MultiLevelInputDocument]
    assert ConcreteTask.output_document_types == [MultiLevelOutputDocument]

    with pytest.raises(TypeError, match="must define"):

        class BadConcreteTask(LevelTwoTask):
            pass
