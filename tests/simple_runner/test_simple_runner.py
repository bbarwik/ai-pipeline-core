"""Tests for the simple_runner module with Pydantic Settings CLI."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import Field

from ai_pipeline_core.documents import DocumentList, FlowDocument
from ai_pipeline_core.flow.config import FlowConfig
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.pipeline import pipeline_flow
from ai_pipeline_core.simple_runner.cli import run_cli
from ai_pipeline_core.simple_runner.simple_runner import (
    run_pipeline,
    run_pipelines,
)


class SimpleInputDocument(FlowDocument):
    """Test document for simple runner tests."""

    pass


class OutputDocument(FlowDocument):
    """Output document for simple runner tests."""

    pass


class SimpleFlowConfig(FlowConfig):
    """Test flow configuration."""

    INPUT_DOCUMENT_TYPES = [SimpleInputDocument]
    OUTPUT_DOCUMENT_TYPE = OutputDocument


class CustomFlowOptions(FlowOptions):
    """Custom flow options for testing."""

    batch_size: int = Field(100, ge=1, description="Batch size for processing")
    temperature: float = Field(0.7, ge=0, le=2, description="Model temperature")
    enable_cache: bool = Field(True, description="Enable caching")


class TestDocumentOperations:
    """Test document loading and saving operations."""

    async def test_save_documents_to_directory(self, tmp_path: Path):
        """Test saving documents to directory."""
        doc1 = SimpleInputDocument(name="test1.txt", content=b"content1")
        doc2 = SimpleInputDocument(
            name="test2.txt", content=b"content2", description="Test description"
        )
        documents = DocumentList([doc1, doc2])

        await SimpleFlowConfig.save_documents(str(tmp_path), documents, validate_output_type=False)

        # Check files were created
        doc_dir = tmp_path / "simple_input"
        assert doc_dir.exists()
        assert (doc_dir / "test1.txt").exists()
        assert (doc_dir / "test2.txt").exists()
        assert (doc_dir / "test2.txt.description.md").exists()

        # Verify content
        assert (doc_dir / "test1.txt").read_bytes() == b"content1"
        assert (doc_dir / "test2.txt").read_bytes() == b"content2"
        assert (doc_dir / "test2.txt.description.md").read_text() == "Test description"

    async def test_load_documents_from_directory(self, tmp_path: Path):
        """Test loading documents from directory."""
        # Create test files
        doc_dir = tmp_path / "simple_input"
        doc_dir.mkdir()
        (doc_dir / "test1.txt").write_bytes(b"content1")
        (doc_dir / "test2.txt").write_bytes(b"content2")
        (doc_dir / "test2.txt.description.md").write_text("Test description")

        # Load documents
        documents = await SimpleFlowConfig.load_documents(str(tmp_path))

        assert len(documents) == 2
        doc1 = next((d for d in documents if d.name == "test1.txt"), None)
        doc2 = next((d for d in documents if d.name == "test2.txt"), None)

        assert doc1 is not None
        assert doc1.content == b"content1"
        assert doc1.description is None

        assert doc2 is not None
        assert doc2.content == b"content2"
        assert doc2.description == "Test description"

    async def test_load_documents_missing_directory(self, tmp_path: Path):
        """Test loading documents when directory doesn't exist."""
        documents = await SimpleFlowConfig.load_documents(str(tmp_path / "nonexistent"))
        assert len(documents) == 0

    async def test_load_documents_with_invalid_file(self, tmp_path: Path, caplog):
        """Test loading documents handles invalid files gracefully."""
        doc_dir = tmp_path / "simple_input"
        doc_dir.mkdir()
        (doc_dir / "valid.txt").write_bytes(b"valid content")

        # Create a file that will cause an error when loading
        with patch.object(
            SimpleInputDocument, "__init__", side_effect=ValueError("Invalid content")
        ):
            documents = await SimpleFlowConfig.load_documents(str(tmp_path))

        assert len(documents) == 0
        assert "Failed to load" in caplog.text


class TestRunPipeline:
    """Test the single run_pipeline function."""

    async def test_run_single_pipeline(self, tmp_path: Path):
        """Test running a single pipeline flow."""

        @pipeline_flow(config=SimpleFlowConfig)
        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            assert project_name == "test-project"
            assert len(documents) == 1
            assert documents[0].name == "input.txt"
            return DocumentList([OutputDocument(name="output.txt", content=b"result")])

        # Prepare input documents
        input_doc = SimpleInputDocument(name="input.txt", content=b"input")
        await SimpleFlowConfig.save_documents(
            str(tmp_path), DocumentList([input_doc]), validate_output_type=False
        )

        # Run single pipeline
        result = await run_pipeline(
            flow_func=test_flow,
            project_name="test-project",
            output_dir=tmp_path,
            flow_options=FlowOptions(),
        )

        # Verify result
        assert len(result) == 1
        assert result[0].name == "output.txt"
        assert result[0].content == b"result"

        # Verify output was saved
        assert (tmp_path / "output" / "output.txt").exists()


class TestRunPipelines:
    """Test the run_pipelines function for multiple flows."""

    async def test_run_pipelines_single_flow(self, tmp_path: Path):
        """Test running a pipeline with a single flow."""

        @pipeline_flow(config=SimpleFlowConfig)
        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            assert project_name == "test-project"
            assert len(documents) == 1
            assert isinstance(flow_options, FlowOptions)
            return DocumentList([OutputDocument(name="output.txt", content=b"output")])

        # Prepare input documents
        input_doc = SimpleInputDocument(name="input.txt", content=b"input")
        await SimpleFlowConfig.save_documents(
            str(tmp_path), DocumentList([input_doc]), validate_output_type=False
        )

        # Run pipeline
        await run_pipelines(
            project_name="test-project",
            output_dir=tmp_path,
            flows=[test_flow],
            flow_options=FlowOptions(),
            start_step=1,
            end_step=1,
        )

        # Verify output
        output_dir = tmp_path / "output"
        assert output_dir.exists()
        assert (output_dir / "output.txt").exists()
        assert (output_dir / "output.txt").read_bytes() == b"output"

    async def test_run_pipeline_multiple_flows(self, tmp_path: Path):
        """Test running a pipeline with multiple flows."""

        class Flow1Config(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        class Flow2Config(FlowConfig):
            INPUT_DOCUMENT_TYPES = [SimpleInputDocument]
            OUTPUT_DOCUMENT_TYPE = OutputDocument

        @pipeline_flow(config=Flow1Config)
        async def flow1(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([
                SimpleInputDocument(name="intermediate.txt", content=b"intermediate")
            ])

        @pipeline_flow(config=Flow2Config)
        async def flow2(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            assert len(documents) == 1
            assert documents[0].name == "intermediate.txt"
            return DocumentList([OutputDocument(name="final.txt", content=b"final")])

        await run_pipelines(
            project_name="test-project",
            output_dir=tmp_path,
            flows=[flow1, flow2],
            flow_options=FlowOptions(),
        )

        # Verify final output
        assert (tmp_path / "output" / "final.txt").exists()
        assert (tmp_path / "output" / "final.txt").read_bytes() == b"final"

    async def test_run_pipeline_with_start_end_steps(self, tmp_path: Path):
        """Test running pipeline with specific start and end steps."""
        flow_calls = []

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def flow1(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            flow_calls.append(1)
            return DocumentList([SimpleInputDocument(name="flow1.txt", content=b"flow1")])

        @pipeline_flow(config=DummyConfig)
        async def flow2(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            flow_calls.append(2)
            return DocumentList([SimpleInputDocument(name="flow2.txt", content=b"flow2")])

        @pipeline_flow(config=DummyConfig)
        async def flow3(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            flow_calls.append(3)
            return DocumentList([SimpleInputDocument(name="flow3.txt", content=b"flow3")])

        # Run only flow 2
        await run_pipelines(
            project_name="test",
            output_dir=tmp_path,
            flows=[flow1, flow2, flow3],
            flow_options=FlowOptions(),
            start_step=2,
            end_step=2,
        )

        assert flow_calls == [2]

    async def test_run_pipeline_invalid_steps(self, tmp_path: Path):
        """Test run_pipeline with invalid step ranges."""

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def dummy_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with pytest.raises(ValueError, match="Invalid start/end steps"):
            await run_pipelines(
                project_name="test",
                output_dir=tmp_path,
                flows=[dummy_flow],
                flow_options=FlowOptions(),
                start_step=2,
                end_step=1,
            )

    async def test_run_pipeline_missing_input_documents(self, tmp_path: Path):
        """Test pipeline fails when required input documents are missing."""

        @pipeline_flow(config=SimpleFlowConfig)
        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with pytest.raises(RuntimeError, match="Missing input documents"):
            await run_pipelines(
                project_name="test",
                output_dir=tmp_path,
                flows=[test_flow],
                flow_options=FlowOptions(),
            )

    async def test_run_pipeline_flow_exception_propagates(self, tmp_path: Path):
        """Test that exceptions in flows propagate correctly."""

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def failing_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            raise ValueError("Flow failed")

        with pytest.raises(ValueError, match="Flow failed"):
            await run_pipelines(
                project_name="test",
                output_dir=tmp_path,
                flows=[failing_flow],
                flow_options=FlowOptions(),
            )


class TestCLI:
    """Test the CLI interface."""

    def test_run_cli_basic(self, tmp_path: Path):
        """Test basic CLI execution."""
        flow_executed = False

        @pipeline_flow(config=SimpleFlowConfig)
        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: CustomFlowOptions
        ) -> DocumentList:
            nonlocal flow_executed
            flow_executed = True
            # Project name should be the directory name from tmp_path
            assert project_name == tmp_path.name
            assert isinstance(flow_options, CustomFlowOptions)
            assert flow_options.batch_size == 200
            assert flow_options.temperature == 0.5
            assert flow_options.enable_cache is False
            return DocumentList([OutputDocument(name="output.txt", content=b"result")])

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            # Project name from initializer is ignored now, directory name is used
            return "ignored", DocumentList([SimpleInputDocument(name="init.txt", content=b"init")])

        # Mock sys.argv
        with patch.object(
            sys,
            "argv",
            [
                "test",
                str(tmp_path),
                "--batch-size",
                "200",
                "--temperature",
                "0.5",
                "--enable-cache",
                "false",
                "--start",
                "1",
                "--end",
                "1",
            ],
        ):
            run_cli(
                flows=[test_flow],
                options_cls=CustomFlowOptions,  # type: ignore[arg-type]
                initializer=initializer,
            )

        assert flow_executed
        assert (tmp_path / "output" / "output.txt").exists()

    def test_run_cli_positional_argument(self, tmp_path: Path):
        """Test that working_directory is handled as positional argument."""
        initializer_called = False

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            nonlocal initializer_called
            initializer_called = True
            # Check that working_directory was parsed
            assert hasattr(opts, "working_directory")
            assert Path(opts.working_directory) == tmp_path  # type: ignore
            return "test", DocumentList([])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def dummy_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with patch.object(sys, "argv", ["test", str(tmp_path)]):
            run_cli(
                flows=[dummy_flow],
                options_cls=FlowOptions,  # type: ignore[arg-type]
                initializer=initializer,
            )

        assert initializer_called
        assert tmp_path.exists()

    def test_run_cli_with_initial_documents(self, tmp_path: Path):
        """Test CLI saves initial documents when start=1."""

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            return "test", DocumentList([
                SimpleInputDocument(name="init1.txt", content=b"initial1"),
                SimpleInputDocument(name="init2.txt", content=b"initial2"),
            ])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def dummy_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with patch.object(sys, "argv", ["test", str(tmp_path), "--start", "1"]):
            run_cli(
                flows=[dummy_flow],
                options_cls=FlowOptions,  # type: ignore[arg-type]
                initializer=initializer,
            )

        # Check initial documents were saved
        assert (tmp_path / "simple_input" / "init1.txt").exists()
        assert (tmp_path / "simple_input" / "init2.txt").exists()
        assert (tmp_path / "simple_input" / "init1.txt").read_bytes() == b"initial1"

    def test_run_cli_skip_initial_documents_when_start_not_1(self, tmp_path: Path):
        """Test CLI skips initial documents when start > 1."""

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            return "test", DocumentList([SimpleInputDocument(name="init.txt", content=b"initial")])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def dummy_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with patch.object(sys, "argv", ["test", str(tmp_path), "--start", "2", "--end", "2"]):

            async def mock_pipeline(*args, **kwargs):
                pass

            with patch(
                "ai_pipeline_core.simple_runner.simple_runner.run_pipelines", new=mock_pipeline
            ):
                run_cli(
                    flows=[dummy_flow, dummy_flow],
                    options_cls=FlowOptions,  # type: ignore[arg-type]
                    initializer=initializer,
                )

        # Initial documents should NOT be saved when start > 1
        assert not (tmp_path / "simple_input" / "init.txt").exists()

    def test_run_cli_environment_setup(self):
        """Test that CLI initializes environment correctly."""
        with patch("ai_pipeline_core.simple_runner.cli.setup_logging") as mock_setup:
            # Mock both places where Laminar.initialize is called
            with patch("ai_pipeline_core.simple_runner.cli.Laminar.initialize") as mock_lmnr_cli:
                with patch("ai_pipeline_core.tracing.Laminar.initialize") as mock_lmnr_tracing:
                    with patch.object(sys, "argv", ["test", "/tmp/test"]):
                        with tempfile.TemporaryDirectory() as tmpdir:
                            sys.argv[1] = tmpdir

                            class DummyConfig(FlowConfig):
                                INPUT_DOCUMENT_TYPES = []
                                OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

                            @pipeline_flow(config=DummyConfig)
                            async def dummy_flow(
                                project_name: str,
                                documents: DocumentList,
                                flow_options: FlowOptions,
                            ) -> DocumentList:
                                return DocumentList([])

                            run_cli(
                                flows=[dummy_flow],
                                options_cls=FlowOptions,  # type: ignore[arg-type]
                                initializer=lambda opts: ("test", DocumentList([])),
                            )

                    mock_setup.assert_called_once()
                    # Either CLI or tracing might call it
                    assert mock_lmnr_cli.called or mock_lmnr_tracing.called

    def test_run_cli_handles_lmnr_initialization_failure(self):
        """Test CLI handles LMNR initialization failure gracefully."""

        # Create the flow outside of the mocked context to avoid issues
        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def dummy_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        # Now mock Laminar.initialize to fail in CLI
        with patch("ai_pipeline_core.simple_runner.cli.Laminar.initialize") as mock_lmnr_cli:
            mock_lmnr_cli.side_effect = Exception("LMNR init failed")

            # Capture the logger to verify warning
            with patch("ai_pipeline_core.simple_runner.cli.logger") as mock_logger:
                with patch.object(sys, "argv", ["test", "/tmp/test"]):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        sys.argv[1] = tmpdir

                        # Should not raise, just log warning
                        run_cli(
                            flows=[dummy_flow],
                            options_cls=FlowOptions,  # type: ignore[arg-type]
                            initializer=lambda opts: ("test", DocumentList([])),
                        )

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                assert "Failed to initialize LMNR tracing" in str(mock_logger.warning.call_args)

    def test_run_cli_with_custom_options_inheritance(self, tmp_path: Path):
        """Test CLI with inherited FlowOptions."""

        class ProjectOptions(CustomFlowOptions):
            """Project-specific options."""

            api_key: str = Field("", description="API key")
            max_retries: int = Field(3, ge=0, description="Max retries")

        options_received = None

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: ProjectOptions
        ) -> DocumentList:
            nonlocal options_received
            options_received = flow_options
            return DocumentList([])

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            return "test", DocumentList([])

        with patch.object(
            sys,
            "argv",
            [
                "test",
                str(tmp_path),
                "--api-key",
                "test-key",
                "--max-retries",
                "5",
                "--batch-size",
                "50",
            ],
        ):
            run_cli(
                flows=[test_flow],
                options_cls=ProjectOptions,  # type: ignore[arg-type]
                initializer=initializer,
            )

        assert options_received is not None
        assert options_received.api_key == "test-key"  # type: ignore
        assert options_received.max_retries == 5  # type: ignore
        assert options_received.batch_size == 50  # type: ignore

    def test_run_cli_error_handling(self, tmp_path: Path):
        """Test CLI error handling doesn't use sys.exit."""

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            raise ValueError("Initializer failed")

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def dummy_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with patch.object(sys, "argv", ["test", str(tmp_path)]):
            # Should raise exception, not call sys.exit
            with pytest.raises(ValueError, match="Initializer failed"):
                run_cli(
                    flows=[dummy_flow],
                    options_cls=FlowOptions,  # type: ignore[arg-type]
                    initializer=initializer,
                )


class TestFlowSequenceValidation:
    """Test validation of flows and configs."""

    async def test_mismatched_flows_and_configs(self, tmp_path: Path):
        """Test that mismatched flows and configs raise error."""

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SimpleInputDocument

        @pipeline_flow(config=DummyConfig)
        async def flow1(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        # flow2 is not decorated, so it doesn't have config attribute
        async def flow2(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([])

        with pytest.raises(RuntimeError, match="does not have a config attribute"):
            await run_pipelines(
                project_name="test",
                output_dir=tmp_path,
                flows=[flow1, flow2],  # flow2 not decorated
                flow_options=FlowOptions(),
            )
