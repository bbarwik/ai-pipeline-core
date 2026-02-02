"""Tests for Phase 7: Static validation at definition/decoration time.

Tests canonical_name collision detection, @pipeline_flow annotation validation,
@pipeline_task DocumentList rejection, return type enforcement, and
PipelineDeployment flow chain validation.
"""

# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportPrivateUsage=false, reportUnusedClass=false

from typing import Any

import pytest

from ai_pipeline_core import (
    DeploymentResult,
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
    pipeline_task,
)
from ai_pipeline_core.pipeline.decorators import _find_non_document_leaves

# --- Document subclasses for testing ---


class AlphaDocument(Document):
    """Alpha document for testing."""


class BetaDocument(Document):
    """Beta document for testing."""


class GammaDocument(Document):
    """Gamma document for testing."""


class DeltaDocument(Document):
    """Delta document for testing."""


class SampleResult(DeploymentResult):
    """Result for deployment testing."""


# --- Flows for deployment testing ---


@pipeline_flow()
async def alpha_to_beta(project_name: str, documents: list[AlphaDocument], flow_options: FlowOptions) -> list[BetaDocument]:
    return [BetaDocument(name="beta.txt", content=b"beta")]


@pipeline_flow()
async def beta_to_gamma(project_name: str, documents: list[BetaDocument], flow_options: FlowOptions) -> list[GammaDocument]:
    return [GammaDocument(name="gamma.txt", content=b"gamma")]


@pipeline_flow()
async def alpha_to_gamma(project_name: str, documents: list[AlphaDocument], flow_options: FlowOptions) -> list[GammaDocument]:
    return [GammaDocument(name="gamma.txt", content=b"gamma")]


@pipeline_flow()
async def gamma_to_delta(project_name: str, documents: list[GammaDocument], flow_options: FlowOptions) -> list[DeltaDocument]:
    return [DeltaDocument(name="delta.txt", content=b"delta")]


@pipeline_flow()
async def needs_delta(project_name: str, documents: list[DeltaDocument], flow_options: FlowOptions) -> list[AlphaDocument]:
    return [AlphaDocument(name="alpha.txt", content=b"alpha")]


@pipeline_flow()
async def union_input_flow(project_name: str, documents: list[BetaDocument | DeltaDocument], flow_options: FlowOptions) -> list[GammaDocument]:
    return [GammaDocument(name="gamma.txt", content=b"gamma")]


# --------------------------------------------------------------------------- #
# Canonical name collision detection tests
# --------------------------------------------------------------------------- #


class TestCanonicalNameCollision:
    """Test Document.__init_subclass__ canonical name collision detection."""

    def test_different_canonical_names_ok(self):
        """Classes with different canonical names register without error."""

        class UniqueNameOneDocument(Document):
            pass

        class UniqueNameTwoDocument(Document):
            pass

        # No error raised
        assert UniqueNameOneDocument.canonical_name() != UniqueNameTwoDocument.canonical_name()

    def test_registry_stores_classes(self):
        """The registry contains registered production classes (e.g., Document itself)."""
        from ai_pipeline_core.documents.document import _canonical_name_registry

        assert isinstance(_canonical_name_registry, dict)
        assert len(_canonical_name_registry) > 0, "Registry should contain production Document subclasses"
        for name, cls in _canonical_name_registry.items():
            assert isinstance(name, str)
            assert isinstance(cls, type)

    def test_test_module_classes_skip_registry(self):
        """Classes defined in test modules are not registered."""
        from ai_pipeline_core.documents.document import _canonical_name_registry, _is_test_module

        # AlphaDocument is defined in this test file
        assert _is_test_module(AlphaDocument)

        # Its canonical name should NOT be in the registry
        canonical = AlphaDocument.canonical_name()
        # If it's there, it's from a different class (unlikely for "alpha")
        existing = _canonical_name_registry.get(canonical)
        assert existing is not AlphaDocument

    def test_collision_detection_for_production_classes(self):
        """Verify the collision detection logic works by directly calling the check."""
        from ai_pipeline_core.documents.document import _is_test_module

        # We can't easily create production-class collisions in a test file
        # (since test modules are skipped), so we test the helper function
        assert _is_test_module(AlphaDocument) is True

        class _FakeProductionClass:
            __module__ = "ai_pipeline_core.custom_documents"

        assert _is_test_module(_FakeProductionClass) is False


# --------------------------------------------------------------------------- #
# @pipeline_flow annotation validation tests
# --------------------------------------------------------------------------- #


class TestFlowAnnotationValidation:
    """Test @pipeline_flow return type and input annotation validation."""

    def test_rejects_non_document_return_type(self):
        """Flow with return annotation that has no Document subclasses is rejected."""
        with pytest.raises(TypeError, match="does not contain Document subclasses"):

            @pipeline_flow()
            async def bad_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[str]:
                return []

    def test_rejects_dict_return_type(self):
        """Flow returning dict is rejected."""
        with pytest.raises(TypeError, match="does not contain Document subclasses"):

            @pipeline_flow()
            async def bad_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> dict[str, Any]:
                return {}

    def test_accepts_document_return_type(self):
        """Flow returning list[Document] is accepted."""

        @pipeline_flow()
        async def good_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            return []

        assert good_flow.output_document_types == [Document]  # type: ignore[attr-defined]

    def test_accepts_concrete_document_return_type(self):
        """Flow returning list[ConcreteDocument] is accepted."""

        @pipeline_flow()
        async def good_flow(project_name: str, documents: list[AlphaDocument], flow_options: FlowOptions) -> list[BetaDocument]:
            return []

        assert good_flow.output_document_types == [BetaDocument]  # type: ignore[attr-defined]

    def test_no_return_annotation_allowed(self):
        """Flow with no return annotation is allowed (no validation triggered)."""

        @pipeline_flow()
        async def flow_without_return(project_name: str, documents: list[Document], flow_options: FlowOptions):
            return []

        # No error — missing annotation means no validation
        assert flow_without_return.output_document_types == []  # type: ignore[attr-defined]

    def test_rejects_non_document_return_annotation(self):
        """When return annotation has no Document subclasses, raise TypeError."""
        with pytest.raises(TypeError, match="return annotation does not contain"):

            @pipeline_flow()
            async def my_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[str]:
                return []


# --------------------------------------------------------------------------- #
# @pipeline_task DocumentList rejection tests
# --------------------------------------------------------------------------- #


class TestTaskDocumentListRejection:
    """Test @pipeline_task rejects stale DocumentList references."""

    def test_rejects_document_list_in_parameter(self):
        """Task with DocumentList parameter annotation is rejected."""
        with pytest.raises(TypeError, match=r"DocumentList.*removed"):

            @pipeline_task
            async def bad_task(docs: "DocumentList") -> None:  # type: ignore  # noqa: F821
                pass

    def test_rejects_document_list_in_return(self):
        """Task with DocumentList return annotation is rejected."""
        with pytest.raises(TypeError, match=r"DocumentList.*removed"):

            @pipeline_task
            async def bad_task() -> "DocumentList":  # type: ignore  # noqa: F821
                pass

    def test_accepts_list_document(self):
        """Task with list[Document] annotation is accepted."""

        @pipeline_task
        async def good_task(docs: list[Document]) -> list[Document]:
            return docs

        # No error

    def test_accepts_no_annotations(self):
        """Task with no annotations is accepted when persist=False."""

        @pipeline_task(persist=False)
        async def simple_task(x: int) -> int:
            return x

        # No error


# --------------------------------------------------------------------------- #
# @pipeline_task return type validation tests
# --------------------------------------------------------------------------- #


class TestTaskReturnTypeValidation:
    """Test @pipeline_task return type annotation enforcement when persist=True."""

    # --- Accepted types ---

    def test_accepts_single_document(self):
        @pipeline_task
        async def t() -> AlphaDocument:
            return AlphaDocument(name="a.txt", content=b"a")

    def test_accepts_base_document(self):
        @pipeline_task
        async def t() -> Document:
            return Document(name="a.txt", content=b"a")

    def test_accepts_list_document(self):
        @pipeline_task
        async def t() -> list[AlphaDocument]:
            return []

    def test_accepts_list_union_documents(self):
        @pipeline_task
        async def t() -> list[AlphaDocument | BetaDocument]:
            return []

    def test_accepts_tuple_documents(self):
        @pipeline_task
        async def t() -> tuple[AlphaDocument, BetaDocument]: ...

    def test_accepts_tuple_of_lists(self):
        @pipeline_task
        async def t() -> tuple[list[AlphaDocument], list[BetaDocument]]: ...

    def test_accepts_mixed_tuple(self):
        @pipeline_task
        async def t() -> tuple[AlphaDocument, list[BetaDocument]]: ...

    def test_accepts_variable_length_tuple(self):
        @pipeline_task
        async def t() -> tuple[AlphaDocument, ...]: ...

    def test_accepts_none(self):
        @pipeline_task
        async def t() -> None:
            pass

    def test_accepts_document_or_none(self):
        @pipeline_task
        async def t() -> AlphaDocument | None:
            return None

    # --- Rejected types ---

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="non-Document types: int"):

            @pipeline_task
            async def t() -> int:
                return 0

    def test_rejects_str(self):
        with pytest.raises(TypeError, match="non-Document types: str"):

            @pipeline_task
            async def t() -> str:
                return ""

    def test_rejects_bool(self):
        with pytest.raises(TypeError, match="non-Document types: bool"):

            @pipeline_task
            async def t() -> bool:
                return True

    def test_rejects_dict(self):
        with pytest.raises(TypeError, match="non-Document types: dict"):

            @pipeline_task
            async def t() -> dict[str, Any]:
                return {}

    def test_rejects_list_str(self):
        with pytest.raises(TypeError, match="non-Document types: str"):

            @pipeline_task
            async def t() -> list[str]:
                return []

    def test_rejects_tuple_with_non_document(self):
        with pytest.raises(TypeError, match="non-Document types: int"):

            @pipeline_task
            async def t() -> tuple[AlphaDocument, int]: ...

    def test_rejects_tuple_list_of_non_document(self):
        with pytest.raises(TypeError, match="non-Document types: str"):

            @pipeline_task
            async def t() -> tuple[list[AlphaDocument], list[str]]: ...

    def test_rejects_missing_annotation(self):
        with pytest.raises(TypeError, match="missing return type annotation"):

            @pipeline_task
            async def t():
                pass

    def test_rejects_any(self):
        with pytest.raises(TypeError, match="non-Document types"):

            @pipeline_task
            async def t() -> Any:
                return None

    def test_rejects_object(self):
        with pytest.raises(TypeError, match="non-Document types: object"):

            @pipeline_task
            async def t() -> object:
                return None

    # --- persist=False opts out ---

    def test_persist_false_allows_int(self):
        @pipeline_task(persist=False)
        async def t() -> int:
            return 0

    def test_persist_false_allows_no_annotation(self):
        @pipeline_task(persist=False)
        async def t():
            pass

    def test_persist_false_allows_str(self):
        @pipeline_task(persist=False)
        async def t() -> str:
            return ""


# --------------------------------------------------------------------------- #
# _find_non_document_leaves unit tests
# --------------------------------------------------------------------------- #


class TestFindNonDocumentLeaves:
    """Direct unit tests for the _find_non_document_leaves helper."""

    def test_none_type(self):
        assert _find_non_document_leaves(type(None)) == []

    def test_document_subclass(self):
        assert _find_non_document_leaves(AlphaDocument) == []

    def test_base_document(self):
        assert _find_non_document_leaves(Document) == []

    def test_int(self):
        assert _find_non_document_leaves(int) == [int]

    def test_str(self):
        assert _find_non_document_leaves(str) == [str]

    def test_list_document(self):
        assert _find_non_document_leaves(list[AlphaDocument]) == []

    def test_list_str(self):
        assert _find_non_document_leaves(list[str]) == [str]

    def test_bare_list(self):
        assert _find_non_document_leaves(list) != []

    def test_bare_tuple(self):
        assert _find_non_document_leaves(tuple) != []

    def test_union_all_documents(self):
        assert _find_non_document_leaves(AlphaDocument | BetaDocument) == []

    def test_union_with_none(self):
        assert _find_non_document_leaves(AlphaDocument | None) == []

    def test_union_mixed(self):
        result = _find_non_document_leaves(AlphaDocument | int)
        assert result == [int]

    def test_tuple_fixed(self):
        assert _find_non_document_leaves(tuple[AlphaDocument, BetaDocument]) == []

    def test_tuple_variable_length(self):
        assert _find_non_document_leaves(tuple[AlphaDocument, ...]) == []

    def test_tuple_of_lists(self):
        assert _find_non_document_leaves(tuple[list[AlphaDocument], list[BetaDocument]]) == []

    def test_tuple_mixed_invalid(self):
        result = _find_non_document_leaves(tuple[AlphaDocument, int])
        assert result == [int]

    def test_any_rejected(self):
        assert _find_non_document_leaves(Any) != []


# --------------------------------------------------------------------------- #
# PipelineDeployment validation tests
# --------------------------------------------------------------------------- #


class TestDeploymentFlowChainValidation:
    """Test PipelineDeployment.__init_subclass__ flow chain validation."""

    def test_valid_chain(self):
        """Valid flow chain: A->B, B->C passes validation."""

        class ValidChain(PipelineDeployment[FlowOptions, SampleResult]):
            flows = [alpha_to_beta, beta_to_gamma]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        assert ValidChain.name == "valid-chain"

    def test_valid_single_flow(self):
        """Single flow deployment passes validation."""

        class SingleFlow(PipelineDeployment[FlowOptions, SampleResult]):
            flows = [alpha_to_beta]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        assert SingleFlow.name == "single-flow"

    def test_broken_chain_raises(self):
        """Flow requiring types not in pool raises TypeError."""
        with pytest.raises(TypeError, match="none are produced by preceding flows"):

            class BrokenChain(PipelineDeployment[FlowOptions, SampleResult]):
                # alpha_to_beta outputs BetaDocument, but needs_delta requires DeltaDocument
                flows = [alpha_to_beta, needs_delta]  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                    return SampleResult(success=True)

    def test_three_step_chain_valid(self):
        """Three-step chain: A->B, B->C, C->D passes."""

        class ThreeStepChain(PipelineDeployment[FlowOptions, SampleResult]):
            flows = [alpha_to_beta, beta_to_gamma, gamma_to_delta]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        assert ThreeStepChain.name == "three-step-chain"

    def test_union_input_any_of_semantics(self):
        """Flow with union input types passes if at least one type is in the pool."""

        class UnionChain(PipelineDeployment[FlowOptions, SampleResult]):
            # alpha_to_beta outputs BetaDocument; union_input_flow accepts Beta|Delta
            # BetaDocument is in pool, so this should pass even though DeltaDocument is not
            flows = [alpha_to_beta, union_input_flow]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        assert UnionChain.name == "union-chain"

    def test_union_input_none_satisfied_raises(self):
        """Flow with union input types fails if no type is in the pool."""
        with pytest.raises(TypeError, match="none are produced by preceding flows"):

            class BadUnion(PipelineDeployment[FlowOptions, SampleResult]):
                # alpha_to_gamma outputs GammaDocument; union_input_flow needs Beta|Delta
                # Neither is in pool
                flows = [alpha_to_gamma, union_input_flow]  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                    return SampleResult(success=True)

    def test_three_step_chain_broken_at_step_three(self):
        """Chain where step 3 needs types not in pool raises."""
        with pytest.raises(TypeError, match="none are produced by preceding flows"):

            class BrokenAtThree(PipelineDeployment[FlowOptions, SampleResult]):
                # alpha_to_beta -> alpha_to_gamma: gamma needs alpha (available), OK
                # but then needs_delta needs DeltaDocument (not available)
                flows = [alpha_to_beta, alpha_to_gamma, needs_delta]  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                    return SampleResult(success=True)


class TestDeploymentDuplicateFlows:
    """Test PipelineDeployment rejects duplicate flows."""

    def test_duplicate_flow_rejected(self):
        """Same flow object appearing twice is rejected."""
        with pytest.raises(TypeError, match="duplicate flow"):

            class DupDeployment(PipelineDeployment[FlowOptions, SampleResult]):
                flows = [alpha_to_beta, alpha_to_beta]  # type: ignore[reportAssignmentType]

                @staticmethod
                def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                    return SampleResult(success=True)

    def test_different_flows_accepted(self):
        """Different flow objects are accepted."""

        class DiffFlows(PipelineDeployment[FlowOptions, SampleResult]):
            flows = [alpha_to_beta, beta_to_gamma]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        assert len(DiffFlows.flows) == 2


class TestDeploymentBuildResultRequired:
    """Test PipelineDeployment requires build_result implementation."""

    def test_missing_build_result_raises(self):
        """Deployment without build_result raises TypeError."""
        with pytest.raises(TypeError, match=r"must implement.*build_result"):

            class NoBuild(PipelineDeployment[FlowOptions, SampleResult]):
                flows = [alpha_to_beta]  # type: ignore[reportAssignmentType]

    def test_abstract_build_result_not_sufficient(self):
        """Inheriting only the abstract build_result from PipelineDeployment raises."""
        with pytest.raises(TypeError, match=r"must implement.*build_result"):

            class InheritedAbstract(PipelineDeployment[FlowOptions, SampleResult]):
                flows = [alpha_to_beta]  # type: ignore[reportAssignmentType]
                # No build_result — only abstract from PipelineDeployment

    def test_concrete_parent_build_result_inherited(self):
        """Inheriting build_result from a concrete parent deployment is allowed."""

        class ParentDeployment(PipelineDeployment[FlowOptions, SampleResult]):
            flows = [alpha_to_beta]  # type: ignore[reportAssignmentType]

            @staticmethod
            def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
                return SampleResult(success=True)

        class ChildDeployment(ParentDeployment):
            flows = [alpha_to_beta, beta_to_gamma]  # type: ignore[reportAssignmentType]
            # Inherits build_result from ParentDeployment — should be fine

        assert ChildDeployment.name == "child-deployment"

    def test_subclass_without_flows_skips_all_validation(self):
        """Intermediate class without flows skips all validation including build_result."""

        class AbstractMiddle(PipelineDeployment[FlowOptions, SampleResult]):
            pass

        # Should not raise — no flows attribute means skip validation
        assert not hasattr(AbstractMiddle, "name")
