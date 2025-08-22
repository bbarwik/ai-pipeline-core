"""Edge case tests for simple_runner to achieve 100% coverage."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import Field

from ai_pipeline_core.documents import DocumentList, FlowDocument, TaskDocument
from ai_pipeline_core.flow.config import FlowConfig
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.simple_runner.cli import run_cli
from ai_pipeline_core.simple_runner.simple_runner import (
    load_documents_from_directory,
    run_pipeline,
    run_pipelines,
    save_documents_to_directory,
)


class SampleDocument(FlowDocument):
    """Sample document for testing."""

    pass


class OutputDocument(FlowDocument):
    """Output document for testing."""

    pass


class NonFlowDoc(TaskDocument):
    """Task document for testing - should be skipped in save."""

    pass


class EdgeCaseFlowConfig(FlowConfig):
    """Test flow configuration for edge cases."""

    INPUT_DOCUMENT_TYPES = [SampleDocument]
    OUTPUT_DOCUMENT_TYPE = OutputDocument


class TestEdgeCases:
    """Test edge cases for simple_runner."""

    def test_save_documents_skips_non_flow_documents(self, tmp_path: Path):
        """Test that save_documents_to_directory skips non-FlowDocument types."""
        flow_doc = SampleDocument(name="flow.txt", content=b"flow content")
        task_doc = NonFlowDoc(name="task.txt", content=b"task content")
        # Document.create is abstract, just skip this test for regular Document

        documents = DocumentList([flow_doc, task_doc])
        save_documents_to_directory(tmp_path, documents)

        # Only flow document should be saved (task documents are skipped)
        assert (tmp_path / "sample" / "flow.txt").exists()
        assert not (tmp_path / "non_flow" / "task.txt").exists()

    def test_project_name_from_empty_directory_name(self, tmp_path: Path):
        """Test error when directory name results in empty project name."""
        # Create a directory with just "/" which would result in empty basename

        with patch.object(sys, "argv", ["test", "/"]):
            with patch("ai_pipeline_core.simple_runner.cli.Path.mkdir"):

                async def dummy_flow(*args, **kwargs) -> DocumentList:
                    return DocumentList([])

                class DummyConfig(FlowConfig):
                    INPUT_DOCUMENT_TYPES = []
                    OUTPUT_DOCUMENT_TYPE = SampleDocument

                with pytest.raises(ValueError, match="Project name cannot be empty"):
                    run_cli(
                        flows=[dummy_flow],
                        flow_configs=[DummyConfig],
                        options_cls=FlowOptions,
                    )

    def test_initializer_returns_tuple(self, tmp_path: Path):
        """Test that initializer returns tuple with project name and documents."""

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            # Return tuple with project name and documents
            return "ignored", DocumentList([SampleDocument(name="init.txt", content=b"init")])

        async def dummy_flow(*args, **kwargs) -> DocumentList:
            return DocumentList([])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SampleDocument

        with patch.object(sys, "argv", ["test", str(tmp_path)]):
            run_cli(
                flows=[dummy_flow],
                flow_configs=[DummyConfig],
                options_cls=FlowOptions,
                initializer=initializer,
            )

        # Check initial document was saved
        assert (tmp_path / "sample" / "init.txt").exists()

    def test_project_name_cli_override(self, tmp_path: Path):
        """Test that --project-name CLI flag overrides directory name."""
        project_name_used = None

        async def capture_project_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            nonlocal project_name_used
            project_name_used = project_name
            return DocumentList([])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SampleDocument

        with patch.object(sys, "argv", ["test", str(tmp_path), "--project-name", "custom-project"]):
            run_cli(
                flows=[capture_project_flow],
                flow_configs=[DummyConfig],
                options_cls=FlowOptions,
            )

        assert project_name_used == "custom-project"

    def test_run_cli_without_initializer(self, tmp_path: Path):
        """Test run_cli works without an initializer."""

        async def dummy_flow(*args, **kwargs) -> DocumentList:
            return DocumentList([SampleDocument(name="output.txt", content=b"output")])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SampleDocument

        with patch.object(sys, "argv", ["test", str(tmp_path)]):
            # Should work without initializer
            run_cli(
                flows=[dummy_flow],
                flow_configs=[DummyConfig],
                options_cls=FlowOptions,
                # No initializer provided
            )

        # Check output was created
        assert (tmp_path / "sample" / "output.txt").exists()

    async def test_run_pipeline_custom_flow_name(self, tmp_path: Path):
        """Test run_pipeline with custom flow name."""

        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([OutputDocument(name="result.txt", content=b"result")])

        # Prepare input documents since EdgeCaseFlowConfig requires them
        input_doc = SampleDocument(name="input.txt", content=b"input")
        save_documents_to_directory(tmp_path, DocumentList([input_doc]))

        # Test with custom flow name
        result = await run_pipeline(
            flow_func=test_flow,
            config=EdgeCaseFlowConfig,
            project_name="test",
            output_dir=tmp_path,
            flow_options=FlowOptions(),
            flow_name="CustomFlowName",
        )

        assert len(result) == 1
        assert result[0].name == "result.txt"

    async def test_run_pipeline_flow_with_name_attribute(self, tmp_path: Path):
        """Test run_pipeline gets name from flow function's name attribute."""

        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            return DocumentList([OutputDocument(name="result.txt", content=b"result")])

        # Prepare input documents since EdgeCaseFlowConfig requires them
        input_doc = SampleDocument(name="input.txt", content=b"input")
        save_documents_to_directory(tmp_path, DocumentList([input_doc]))

        # Add name attribute to function
        test_flow.name = "FlowWithNameAttribute"  # type: ignore

        result = await run_pipeline(
            flow_func=test_flow,
            config=EdgeCaseFlowConfig,
            project_name="test",
            output_dir=tmp_path,
            flow_options=FlowOptions(),
            # No flow_name provided, should use attribute
        )

        assert len(result) == 1

    async def test_run_pipelines_with_none_end_step(self, tmp_path: Path):
        """Test run_pipelines with end_step=None runs all flows."""
        flow_executions = []

        async def flow1(*args, **kwargs) -> DocumentList:
            flow_executions.append(1)
            return DocumentList([SampleDocument(name="f1.txt", content=b"1")])

        async def flow2(*args, **kwargs) -> DocumentList:
            flow_executions.append(2)
            return DocumentList([SampleDocument(name="f2.txt", content=b"2")])

        async def flow3(*args, **kwargs) -> DocumentList:
            flow_executions.append(3)
            return DocumentList([SampleDocument(name="f3.txt", content=b"3")])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SampleDocument

        await run_pipelines(
            project_name="test",
            output_dir=tmp_path,
            flows=[flow1, flow2, flow3],
            flow_configs=[DummyConfig, DummyConfig, DummyConfig],
            flow_options=FlowOptions(),
            start_step=1,
            end_step=None,  # Should run all flows
        )

        assert flow_executions == [1, 2, 3]

    def test_load_documents_with_test_data(self):
        """Test loading documents from test_data directory."""
        test_data_dir = Path("tests/test_data/sample_project")
        test_data_dir.mkdir(parents=True, exist_ok=True)

        # Create sample documents
        doc_dir = test_data_dir / "sample"
        doc_dir.mkdir(exist_ok=True)

        (doc_dir / "file1.txt").write_bytes(b"content1")
        (doc_dir / "file2.txt").write_bytes(b"content2")
        (doc_dir / "file2.txt.description.md").write_text("A description for file2")

        try:
            # Load documents
            documents = load_documents_from_directory(test_data_dir, [SampleDocument])

            assert len(documents) == 2
            file1 = next((d for d in documents if d.name == "file1.txt"), None)
            file2 = next((d for d in documents if d.name == "file2.txt"), None)

            assert file1 is not None
            assert file1.content == b"content1"
            assert file1.description is None

            assert file2 is not None
            assert file2.content == b"content2"
            assert file2.description == "A description for file2"

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(test_data_dir, ignore_errors=True)

    def test_cli_with_start_greater_than_1_skips_initial_docs(self, tmp_path: Path):
        """Test that initial documents are not saved when start > 1."""

        def initializer(opts: FlowOptions) -> tuple[str, DocumentList]:
            return "ignored", DocumentList([SampleDocument(name="init.txt", content=b"initial")])

        async def dummy_flow(*args, **kwargs) -> DocumentList:
            return DocumentList([])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SampleDocument

        with patch.object(sys, "argv", ["test", str(tmp_path), "--start", "2", "--end", "2"]):

            async def mock_pipelines(*args, **kwargs):
                pass

            with patch(
                "ai_pipeline_core.simple_runner.simple_runner.run_pipelines", new=mock_pipelines
            ):
                run_cli(
                    flows=[dummy_flow, dummy_flow],
                    flow_configs=[DummyConfig, DummyConfig],
                    options_cls=FlowOptions,
                    initializer=initializer,
                )

        # Initial document should NOT be saved when start > 1
        assert not (tmp_path / "sample" / "init.txt").exists()

    def test_custom_flow_options_in_cli(self, tmp_path: Path):
        """Test CLI with custom FlowOptions containing various field types."""

        class ComplexFlowOptions(FlowOptions):
            """Complex options with various field types."""

            string_opt: str = Field("default", description="String option")
            int_opt: int = Field(42, description="Integer option")
            float_opt: float = Field(3.14, description="Float option")
            bool_opt: bool = Field(True, description="Boolean option")
            optional_opt: str | None = Field(None, description="Optional option")

        options_received = None

        async def capture_options_flow(
            project_name: str, documents: DocumentList, flow_options: ComplexFlowOptions
        ) -> DocumentList:
            nonlocal options_received
            options_received = flow_options
            return DocumentList([])

        class DummyConfig(FlowConfig):
            INPUT_DOCUMENT_TYPES = []
            OUTPUT_DOCUMENT_TYPE = SampleDocument

        with patch.object(
            sys,
            "argv",
            [
                "test",
                str(tmp_path),
                "--string-opt",
                "custom",
                "--int-opt",
                "100",
                "--float-opt",
                "2.71",
                "--bool-opt",
                "false",
                "--optional-opt",
                "provided",
            ],
        ):
            run_cli(
                flows=[capture_options_flow],
                flow_configs=[DummyConfig],
                options_cls=ComplexFlowOptions,
            )

        assert options_received is not None
        assert options_received.string_opt == "custom"  # type: ignore
        assert options_received.int_opt == 100  # type: ignore
        assert options_received.float_opt == 2.71  # type: ignore
        assert options_received.bool_opt is False  # type: ignore
        assert options_received.optional_opt == "provided"  # type: ignore
