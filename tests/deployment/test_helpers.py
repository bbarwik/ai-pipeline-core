"""Tests for deployment helper functions."""

# pyright: reportPrivateUsage=false

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx
import pytest

from ai_pipeline_core import (
    Document,
    FlowOptions,
    PipelineDeployment,
    pipeline_flow,
)
from ai_pipeline_core.deployment import DeploymentResult
from ai_pipeline_core.deployment.contract import ProgressRun
from ai_pipeline_core.deployment._helpers import (
    class_name_to_deployment_name,
    download_documents,
    extract_generic_params,
    send_webhook,
    upload_documents,
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

    def test_returns_none_for_non_generic(self):
        """Test returns None for class without generic params."""

        class Plain:
            """Plain class."""

        options_type, result_type = extract_generic_params(Plain, PipelineDeployment)
        assert options_type is None
        assert result_type is None


class TestSendWebhook:
    """Test webhook sending with retries."""

    async def test_success(self):
        """Test successful webhook delivery."""
        payload = ProgressRun(
            flow_run_id=UUID(int=0),
            project_name="test",
            state="RUNNING",
            timestamp=datetime.now(UTC),
            step=1,
            total_steps=3,
            flow_name="flow1",
            status="started",
            progress=0.0,
            step_progress=0.0,
            message="Starting",
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None

        with patch("ai_pipeline_core.deployment._helpers.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await send_webhook("http://example.com/hook", payload)
            mock_client.post.assert_called_once()

    async def test_retries_on_failure(self):
        """Test webhook retries on transient failure."""
        payload = ProgressRun(
            flow_run_id=UUID(int=0),
            project_name="test",
            state="RUNNING",
            timestamp=datetime.now(UTC),
            step=1,
            total_steps=1,
            flow_name="flow",
            status="started",
            progress=0.0,
            step_progress=0.0,
            message="msg",
        )

        with patch("ai_pipeline_core.deployment._helpers.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = httpx.HTTPError("timeout")
            mock_client_cls.return_value = mock_client

            with patch("ai_pipeline_core.deployment._helpers.asyncio.sleep"), pytest.raises(httpx.HTTPError):
                await send_webhook("http://example.com/hook", payload, max_retries=2, retry_delay=0)

            assert mock_client.post.call_count == 2


class TestDownloadDocuments:
    """Test document downloading from URLs."""

    async def test_downloads_and_creates_documents(self):
        """Test successful document download."""
        mock_response = AsyncMock()
        mock_response.content = b"test content"
        mock_response.raise_for_status = lambda: None

        with patch("ai_pipeline_core.deployment._helpers.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            docs = await download_documents(["http://example.com/file.txt"])
            assert len(docs) == 1
            assert docs[0].name == "file.txt"


class TestUploadDocuments:
    """Test document uploading to URLs."""

    async def test_uploads_matching_documents(self):
        """Test upload with URL mapping."""

        class UploadDoc(Document):
            """Document to upload."""

        doc = UploadDoc(name="output.txt", content=b"result data")
        docs = [doc]
        url_mapping = {"output.txt": "http://example.com/upload/output.txt"}

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None

        with patch("ai_pipeline_core.deployment._helpers.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.put.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await upload_documents(docs, url_mapping)
            mock_client.put.assert_called_once()


# --- Module-level test infrastructure ---


class SampleResult(DeploymentResult):
    """Test result type."""

    report: str = ""


class SampleInputDoc(Document):
    """Input for testing."""


class SampleOutputDoc(Document):
    """Output for testing."""


@pipeline_flow()
async def sample_flow(project_name: str, documents: list[SampleInputDoc], flow_options: FlowOptions) -> list[SampleOutputDoc]:
    """Sample flow."""
    return []


class SampleDeployment(PipelineDeployment[FlowOptions, SampleResult]):
    """Deployment for testing."""

    flows = [sample_flow]  # type: ignore[reportAssignmentType]

    @staticmethod
    def build_result(project_name: str, documents: list[Document], options: FlowOptions) -> SampleResult:
        """Build result."""
        return SampleResult(success=True)
