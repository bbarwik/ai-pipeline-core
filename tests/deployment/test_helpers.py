"""Tests for deployment helper functions."""

# pyright: reportPrivateUsage=false

from ai_pipeline_core import (
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)
from ai_pipeline_core.deployment import DeploymentResult
from ai_pipeline_core.deployment._helpers import (
    class_name_to_deployment_name,
    extract_generic_params,
)


class TestClassNameToDeploymentName:
    """Test PascalCase to kebab-case conversion."""

    def test_simple_name(self):
        """Test simple two-word name."""
        assert class_name_to_deployment_name("ResearchPipeline") == "research-pipeline"

    def test_single_word(self):
        """Test single word name."""
        assert class_name_to_deployment_name("Pipeline") == "pipeline"

    def test_multi_word(self):
        """Test multi-word name."""
        assert class_name_to_deployment_name("MyLongPipelineName") == "my-long-pipeline-name"

    def test_acronym(self):
        """Test name with consecutive capitals."""
        assert class_name_to_deployment_name("AIResearch") == "a-i-research"


class TestExtractGenericParams:
    """Test extraction of generic type parameters from PipelineDeployment subclasses."""

    def test_extracts_options_and_result(self):
        """Test correct extraction from concrete subclass."""
        options_type, result_type = extract_generic_params(SampleDeployment, PipelineDeployment)
        assert options_type is FlowOptions
        assert result_type is SampleResult

    def test_returns_empty_for_non_generic(self):
        """Test returns empty tuple for class without generic params."""

        class Plain:
            """Plain class."""

        result = extract_generic_params(Plain, PipelineDeployment)
        assert result == ()


# --- Module-level test infrastructure ---


class SampleResult(DeploymentResult):
    """Test result type."""

    report: str = ""


class SampleInputDoc(Document):
    """Input for testing."""


class SampleOutputDoc(Document):
    """Output for testing."""


@pipeline_flow()
async def sample_flow(run_id: str, documents: list[SampleInputDoc], flow_options: FlowOptions) -> list[SampleOutputDoc]:
    """Sample flow."""
    return []


class SampleDeployment(PipelineDeployment[FlowOptions, SampleResult]):
    """Deployment for testing."""

    flows = [sample_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(run_id: str, documents: list[Document], options: FlowOptions) -> SampleResult:
        """Build result."""
        return SampleResult(success=True)


# ---------------------------------------------------------------------------
# Tests for init_observability_best_effort (coverage for _helpers.py lines 13-23)
# ---------------------------------------------------------------------------

from unittest.mock import patch


class TestInitObservabilityBestEffort:
    @patch("ai_pipeline_core.observability._initialization.initialize_observability")
    def test_success(self, mock_init):
        from ai_pipeline_core.deployment._helpers import init_observability_best_effort

        init_observability_best_effort()
        mock_init.assert_called_once()

    @patch("ai_pipeline_core.observability.tracing._initialise_laminar")
    @patch("ai_pipeline_core.observability._initialization.initialize_observability", side_effect=RuntimeError("fail"))
    def test_fallback_to_laminar(self, mock_init, mock_laminar):
        from ai_pipeline_core.deployment._helpers import init_observability_best_effort

        init_observability_best_effort()
        mock_laminar.assert_called_once()

    @patch("ai_pipeline_core.observability.tracing._initialise_laminar", side_effect=ImportError("no lmnr"))
    @patch("ai_pipeline_core.observability._initialization.initialize_observability", side_effect=RuntimeError("fail"))
    def test_both_fail_swallowed(self, mock_init, mock_laminar):
        from ai_pipeline_core.deployment._helpers import init_observability_best_effort

        init_observability_best_effort()  # should not raise
