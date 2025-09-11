"""Tests for pipeline_flow storage functionality."""

import tempfile
from pathlib import Path

import pytest

from ai_pipeline_core import pipeline_flow
from ai_pipeline_core.documents import DocumentList, FlowDocument
from ai_pipeline_core.flow.config import FlowConfig
from ai_pipeline_core.flow.options import FlowOptions


class StorageInputDoc(FlowDocument):
    """Storage test input document."""

    pass


class StorageOutputDoc(FlowDocument):
    """Storage test output document."""

    pass


class StorageFlowConfig(FlowConfig):
    """Storage test flow configuration."""

    INPUT_DOCUMENT_TYPES = [StorageInputDoc]
    OUTPUT_DOCUMENT_TYPE = StorageOutputDoc


class StorageFlowOptions(FlowOptions):
    """Storage test flow options."""

    pass


@pytest.mark.asyncio
async def test_pipeline_flow_loads_documents_from_string(prefect_test_fixture):
    """Test that pipeline_flow loads documents when given a string path."""

    @pipeline_flow(config=StorageFlowConfig)
    async def test_flow(
        project_name: str, documents: DocumentList, flow_options: FlowOptions
    ) -> DocumentList:
        # Check we received the loaded documents
        assert len(documents) == 1
        assert isinstance(documents[0], StorageInputDoc)
        assert documents[0].name == "input.txt"
        assert documents[0].content == b"test input"

        # Return output
        return DocumentList([StorageOutputDoc(name="output.txt", content=b"test output")])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare input documents
        input_dir = Path(tmpdir) / "storageinputdoc"
        input_dir.mkdir()
        (input_dir / "input.txt").write_bytes(b"test input")

        # Create flow options with config_class
        options = StorageFlowOptions()

        # Run flow with string path
        result = await test_flow("test_project", tmpdir, options)

        # Check output was saved
        output_dir = Path(tmpdir) / "storageoutputdoc"
        assert output_dir.exists()
        assert (output_dir / "output.txt").exists()
        assert (output_dir / "output.txt").read_bytes() == b"test output"

        assert len(result) == 1
        assert isinstance(result[0], StorageOutputDoc)


@pytest.mark.asyncio
async def test_pipeline_flow_with_documentlist_no_save(prefect_test_fixture):
    """Test that pipeline_flow doesn't save when given DocumentList directly."""

    @pipeline_flow(config=StorageFlowConfig)
    async def test_flow(
        project_name: str, documents: DocumentList, flow_options: FlowOptions
    ) -> DocumentList:
        return DocumentList([StorageOutputDoc(name="output.txt", content=b"test output")])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input documents directly
        input_docs = DocumentList([StorageInputDoc(name="input.txt", content=b"test input")])

        # Create flow options with config_class
        options = StorageFlowOptions()

        # Run flow with DocumentList
        result = await test_flow("test_project", input_docs, options)

        # Check output was NOT saved (no path provided)
        output_dir = Path(tmpdir) / "storageoutputdoc"
        assert not output_dir.exists()

        assert len(result) == 1
        assert isinstance(result[0], StorageOutputDoc)


@pytest.mark.asyncio
async def test_pipeline_flow_without_config(prefect_test_fixture):
    """Test pipeline_flow with string path but no config."""

    @pipeline_flow()  # No config provided
    async def test_flow(
        project_name: str, documents: DocumentList, flow_options: FlowOptions
    ) -> DocumentList:
        # Should receive empty DocumentList when no config
        assert len(documents) == 0
        return DocumentList([StorageOutputDoc(name="output.txt", content=b"test output")])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create flow options
        options = FlowOptions()

        # Run flow with string path
        result = await test_flow("test_project", tmpdir, options)

        # Check output was NOT saved (no config)
        output_dir = Path(tmpdir) / "storageoutputdoc"
        assert not output_dir.exists()

        assert len(result) == 1
        assert isinstance(result[0], StorageOutputDoc)


@pytest.mark.asyncio
async def test_pipeline_flow_with_description_and_sources(prefect_test_fixture):
    """Test that pipeline_flow saves and loads document metadata."""

    @pipeline_flow(config=StorageFlowConfig)
    async def test_flow(
        project_name: str, documents: DocumentList, flow_options: FlowOptions
    ) -> DocumentList:
        # Check loaded metadata
        assert len(documents) == 1
        doc = documents[0]
        assert doc.description == "Input description"
        assert doc.sources == ["source1", "source2"]

        # Return output with metadata
        return DocumentList([
            StorageOutputDoc(
                name="output.txt",
                content=b"test output",
                description="Output description",
                sources=["source3", "source4"],
            )
        ])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare input with metadata
        input_dir = Path(tmpdir) / "storageinputdoc"
        input_dir.mkdir()
        (input_dir / "input.txt").write_bytes(b"test input")
        (input_dir / "input.txt.description.md").write_text("Input description")

        import json

        (input_dir / "input.txt.sources.json").write_text(json.dumps(["source1", "source2"]))

        # Create flow options with config_class
        options = StorageFlowOptions()

        # Run flow
        await test_flow("test_project", tmpdir, options)

        # Check output metadata was saved
        output_dir = Path(tmpdir) / "storageoutputdoc"
        assert (output_dir / "output.txt.description.md").exists()
        assert (output_dir / "output.txt.description.md").read_text() == "Output description"

        assert (output_dir / "output.txt.sources.json").exists()
        sources = json.loads((output_dir / "output.txt.sources.json").read_text())
        assert sources == ["source3", "source4"]
