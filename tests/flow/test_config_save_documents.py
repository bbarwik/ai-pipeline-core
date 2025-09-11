"""Tests for FlowConfig save_documents and load_documents methods."""

import json
import tempfile
from pathlib import Path

import pytest

from ai_pipeline_core.documents import DocumentList, FlowDocument, TaskDocument, TemporaryDocument
from ai_pipeline_core.exceptions import DocumentValidationError
from ai_pipeline_core.flow import FlowConfig


class InputDoc(FlowDocument):
    """Input document for testing."""

    pass


class SecondInputDoc(FlowDocument):
    """Second input document type for testing."""

    pass


class OutputDoc(FlowDocument):
    """Output document for testing."""

    pass


class UnrelatedDoc(FlowDocument):
    """Unrelated document type for testing."""

    pass


class SampleFlowConfig(FlowConfig):
    """Sample flow configuration for testing."""

    INPUT_DOCUMENT_TYPES = [InputDoc, SecondInputDoc]
    OUTPUT_DOCUMENT_TYPE = OutputDoc


class SingleInputFlowConfig(FlowConfig):
    """Flow config with single input type."""

    INPUT_DOCUMENT_TYPES = [InputDoc]
    OUTPUT_DOCUMENT_TYPE = OutputDoc


class OutputDocFlowConfig(FlowConfig):
    """Flow config for loading OutputDoc."""

    INPUT_DOCUMENT_TYPES = [OutputDoc]
    OUTPUT_DOCUMENT_TYPE = SecondInputDoc


class SampleTaskDoc(TaskDocument):
    """Sample task document for testing."""

    pass


# ============================================================================
# save_documents tests
# ============================================================================


@pytest.mark.asyncio
async def test_save_documents_basic():
    """Test basic document saving functionality."""
    # Create test documents
    doc1 = InputDoc(
        name="input.json",
        content=b'{"data": "test"}',
        description="Input document",
        sources=["source1", "source2"],
    )

    doc2 = OutputDoc(name="output.txt", content=b"output data")

    docs = DocumentList([doc1, doc2])

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = Path(tmpdir) / "output"

        # Save documents
        await SampleFlowConfig.save_documents(str(uri), docs, validate_output_type=False)

        # Verify files were created
        inputdoc_dir = uri / "inputdoc"
        assert inputdoc_dir.exists()
        assert (inputdoc_dir / "input.json").exists()
        assert (inputdoc_dir / "input.json.description.md").exists()
        assert (inputdoc_dir / "input.json.sources.json").exists()

        # Check description content
        desc_content = (inputdoc_dir / "input.json.description.md").read_text()
        assert desc_content == "Input document"

        # Check sources content
        sources_content = (inputdoc_dir / "input.json.sources.json").read_text()
        sources_data = json.loads(sources_content)
        assert sources_data == ["source1", "source2"]

        # Check OutputDoc was saved
        outputdoc_dir = uri / "outputdoc"
        assert outputdoc_dir.exists()
        assert (outputdoc_dir / "output.txt").exists()
        assert not (outputdoc_dir / "output.txt.description.md").exists()
        assert not (outputdoc_dir / "output.txt.sources.json").exists()


@pytest.mark.asyncio
async def test_save_documents_skip_non_flow():
    """Test that non-FlowDocument instances are skipped."""
    flow_doc = OutputDoc(name="flow.txt", content=b"flow data")

    task_doc = SampleTaskDoc(name="task.txt", content=b"task data")

    temp_doc = TemporaryDocument(name="temp.txt", content=b"temp data")

    docs = DocumentList([flow_doc, task_doc, temp_doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = Path(tmpdir) / "output"

        # Save documents (only flow_doc should be saved)
        await SampleFlowConfig.save_documents(str(uri), docs, validate_output_type=False)

        # Verify only FlowDocument was saved
        assert (uri / "outputdoc").exists()
        assert (uri / "outputdoc" / "flow.txt").exists()

        # Verify non-FlowDocument types were not saved
        assert not (uri / "sampletaskdoc").exists()
        assert not (uri / "temporarydocument").exists()


@pytest.mark.asyncio
async def test_save_documents_with_validation():
    """Test output type validation."""
    # Correct output type
    output_doc = OutputDoc(name="output.json", content=b'{"result": "success"}')

    docs_correct = DocumentList([output_doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = Path(tmpdir) / "output"

        # This should succeed
        await SampleFlowConfig.save_documents(str(uri), docs_correct, validate_output_type=True)
        assert (uri / "outputdoc" / "output.json").exists()


@pytest.mark.asyncio
async def test_save_documents_validation_error():
    """Test that validation error is raised for wrong document type."""
    # Wrong document type
    wrong_doc = InputDoc(name="wrong.json", content=b'{"wrong": "type"}')

    docs_wrong = DocumentList([wrong_doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = Path(tmpdir) / "output"

        # This should raise DocumentValidationError
        with pytest.raises(DocumentValidationError) as exc_info:
            await SampleFlowConfig.save_documents(str(uri), docs_wrong, validate_output_type=True)

        assert "incorrect type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_save_documents_no_metadata():
    """Test saving document without description or sources."""
    doc = OutputDoc(
        name="simple.txt",
        content=b"simple content",
        # No description or sources
    )

    docs = DocumentList([doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = Path(tmpdir) / "output"

        await SampleFlowConfig.save_documents(str(uri), docs, validate_output_type=True)

        outputdoc_dir = uri / "outputdoc"
        assert (outputdoc_dir / "simple.txt").exists()

        # Metadata files should not exist
        assert not (outputdoc_dir / "simple.txt.description.md").exists()
        assert not (outputdoc_dir / "simple.txt.sources.json").exists()


@pytest.mark.asyncio
async def test_save_documents_empty_sources():
    """Test saving document with empty sources list."""
    doc = OutputDoc(
        name="empty_sources.txt",
        content=b"content",
        sources=[],  # Empty sources list
    )

    docs = DocumentList([doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        uri = Path(tmpdir) / "output"

        await SampleFlowConfig.save_documents(str(uri), docs, validate_output_type=False)

        outputdoc_dir = uri / "outputdoc"
        assert (outputdoc_dir / "empty_sources.txt").exists()

        # Sources file should not be created for empty list
        assert not (outputdoc_dir / "empty_sources.txt.sources.json").exists()


# ============================================================================
# load_documents tests
# ============================================================================


@pytest.mark.asyncio
async def test_load_documents_basic():
    """Test basic document loading functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create document structure
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()

        # Write document content
        (input_dir / "test.json").write_bytes(b'{"test": "data"}')

        # Write metadata
        (input_dir / "test.json.description.md").write_text("Test description")
        (input_dir / "test.json.sources.json").write_text(json.dumps(["source1", "source2"]))

        # Load documents
        docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, InputDoc)
        assert doc.name == "test.json"
        assert doc.content == b'{"test": "data"}'
        assert doc.description == "Test description"
        assert doc.sources == ["source1", "source2"]


@pytest.mark.asyncio
async def test_load_documents_multiple_types():
    """Test loading multiple document types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create InputDoc
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()
        (input_dir / "input.txt").write_bytes(b"input data")

        # Create SecondInputDoc
        second_dir = Path(tmpdir) / "secondinputdoc"
        second_dir.mkdir()
        (second_dir / "second.txt").write_bytes(b"second data")

        # Load documents
        docs = await SampleFlowConfig.load_documents(tmpdir)

        assert len(docs) == 2

        # Check we have one of each type
        input_docs = [d for d in docs if isinstance(d, InputDoc)]
        second_docs = [d for d in docs if isinstance(d, SecondInputDoc)]

        assert len(input_docs) == 1
        assert len(second_docs) == 1

        assert input_docs[0].content == b"input data"
        assert second_docs[0].content == b"second data"


@pytest.mark.asyncio
async def test_load_documents_specific_types():
    """Test loading specific document types only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create InputDoc
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()
        (input_dir / "input.txt").write_bytes(b"input data")

        # Create SecondInputDoc
        second_dir = Path(tmpdir) / "secondinputdoc"
        second_dir.mkdir()
        (second_dir / "second.txt").write_bytes(b"second data")

        # Load using SingleInputFlowConfig which has only InputDoc
        docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(docs) == 1
        assert isinstance(docs[0], InputDoc)
        assert docs[0].content == b"input data"


@pytest.mark.asyncio
async def test_load_documents_no_metadata():
    """Test loading documents without metadata files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create document without metadata
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()
        (input_dir / "plain.txt").write_bytes(b"plain content")

        # Load documents
        docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.name == "plain.txt"
        assert doc.content == b"plain content"
        assert doc.description is None
        assert doc.sources == []


@pytest.mark.asyncio
async def test_load_documents_corrupted_metadata():
    """Test loading documents with corrupted metadata files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create document
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()
        (input_dir / "doc.txt").write_bytes(b"content")

        # Write corrupted sources JSON
        (input_dir / "doc.txt.sources.json").write_text("not valid json!")

        # Should still load document but skip corrupted metadata
        docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.name == "doc.txt"
        assert doc.content == b"content"
        assert doc.sources == []  # Corrupted sources ignored


@pytest.mark.asyncio
async def test_load_documents_missing_directory():
    """Test loading from non-existent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to load when inputdoc directory doesn't exist
        docs = await SingleInputFlowConfig.load_documents(tmpdir)

        # Should return empty list, not error
        assert len(docs) == 0


@pytest.mark.asyncio
async def test_load_documents_skip_metadata_files():
    """Test that metadata files are not loaded as documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create document with metadata
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()
        (input_dir / "doc.txt").write_bytes(b"content")
        (input_dir / "doc.txt.description.md").write_text("Description")
        (input_dir / "doc.txt.sources.json").write_text('["source"]')

        # Also add standalone metadata files (should be ignored)
        (input_dir / "standalone.description.md").write_text("Standalone desc")
        (input_dir / "standalone.sources.json").write_text('["standalone"]')

        # Load documents
        docs = await SingleInputFlowConfig.load_documents(tmpdir)

        # Should only load the actual document, not metadata files
        assert len(docs) == 1
        assert docs[0].name == "doc.txt"


@pytest.mark.asyncio
async def test_load_documents_no_input_types():
    """Test error when loading without document types on class without INPUT_DOCUMENT_TYPES."""
    # Create a mock class that bypasses validation but has no INPUT_DOCUMENT_TYPES
    # We need to directly create a class that inherits from FlowConfig
    # but doesn't trigger validation

    class NoInputTypesConfig(FlowConfig):
        """Config without INPUT_DOCUMENT_TYPES for testing."""

        INPUT_DOCUMENT_TYPES = []  # Empty list to pass validation
        OUTPUT_DOCUMENT_TYPE = OutputDoc

    # Remove the INPUT_DOCUMENT_TYPES after class creation to simulate missing attribute
    delattr(NoInputTypesConfig, "INPUT_DOCUMENT_TYPES")

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AttributeError) as exc_info:
            await NoInputTypesConfig.load_documents(tmpdir)

        assert "INPUT_DOCUMENT_TYPES" in str(exc_info.value)


# ============================================================================
# Round-trip tests (save then load)
# ============================================================================


@pytest.mark.asyncio
async def test_round_trip_basic():
    """Test saving and loading documents preserves all data."""
    # Create documents with all metadata
    doc1 = InputDoc(
        name="input.json",
        content=b'{"key": "value"}',
        description="Input document description",
        sources=["sha256_hash", "http://example.com"],
    )

    doc2 = SecondInputDoc(
        name="config.yaml",
        content=b"setting: true\nvalue: 42",
        description="Configuration file",
        sources=["config_source"],
    )

    docs = DocumentList([doc1, doc2])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save documents
        await SampleFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Load documents back
        loaded_docs = await SampleFlowConfig.load_documents(tmpdir)

        # Should have same number of documents
        assert len(loaded_docs) == 2

        # Find corresponding documents
        loaded_input = next(d for d in loaded_docs if isinstance(d, InputDoc))
        loaded_second = next(d for d in loaded_docs if isinstance(d, SecondInputDoc))

        # Verify InputDoc
        assert loaded_input.name == "input.json"
        assert loaded_input.content == b'{"key": "value"}'
        assert loaded_input.description == "Input document description"
        assert loaded_input.sources == ["sha256_hash", "http://example.com"]

        # Verify SecondInputDoc
        assert loaded_second.name == "config.yaml"
        assert loaded_second.content == b"setting: true\nvalue: 42"
        assert loaded_second.description == "Configuration file"
        assert loaded_second.sources == ["config_source"]


@pytest.mark.asyncio
async def test_round_trip_no_metadata():
    """Test round-trip for documents without metadata."""
    doc = InputDoc(name="simple.txt", content=b"simple content")

    docs = DocumentList([doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        await SingleInputFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Load
        loaded_docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(loaded_docs) == 1
        loaded = loaded_docs[0]

        assert loaded.name == "simple.txt"
        assert loaded.content == b"simple content"
        assert loaded.description is None
        assert loaded.sources == []


@pytest.mark.asyncio
async def test_round_trip_mixed_documents():
    """Test round-trip with mixed document types (only FlowDocuments persist)."""
    flow_doc = OutputDoc(name="flow.txt", content=b"flow data", description="Flow document")

    task_doc = SampleTaskDoc(name="task.txt", content=b"task data")

    temp_doc = TemporaryDocument(name="temp.txt", content=b"temp data")

    docs = DocumentList([flow_doc, task_doc, temp_doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save (only flow_doc should be saved)
        await SampleFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Load with OutputDoc type using OutputDocFlowConfig
        loaded_docs = await OutputDocFlowConfig.load_documents(tmpdir)

        # Should only have the FlowDocument
        assert len(loaded_docs) == 1
        loaded = loaded_docs[0]

        assert isinstance(loaded, OutputDoc)
        assert loaded.name == "flow.txt"
        assert loaded.content == b"flow data"
        assert loaded.description == "Flow document"


@pytest.mark.asyncio
async def test_round_trip_multiple_files_same_type():
    """Test round-trip with multiple files of the same document type."""
    doc1 = InputDoc(name="file1.txt", content=b"content 1", description="First file")

    doc2 = InputDoc(name="file2.txt", content=b"content 2", description="Second file")

    doc3 = InputDoc(name="file3.txt", content=b"content 3")

    docs = DocumentList([doc1, doc2, doc3])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        await SingleInputFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Load
        loaded_docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(loaded_docs) == 3

        # Verify all files loaded correctly
        names_content = {doc.name: doc.content for doc in loaded_docs}
        assert names_content["file1.txt"] == b"content 1"
        assert names_content["file2.txt"] == b"content 2"
        assert names_content["file3.txt"] == b"content 3"

        # Check descriptions
        descriptions = {doc.name: doc.description for doc in loaded_docs}
        assert descriptions["file1.txt"] == "First file"
        assert descriptions["file2.txt"] == "Second file"
        assert descriptions["file3.txt"] is None


@pytest.mark.asyncio
async def test_round_trip_special_characters():
    """Test round-trip with special characters in content and metadata."""
    doc = InputDoc(
        name="special.json",
        content=json.dumps({"text": "Special chars: 🎉 ñ é 中文"}).encode(),
        description="Description with émojis 🚀 and ñoñ-ASCII",
        sources=["source/with/slashes", "source:with:colons"],
    )

    docs = DocumentList([doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        await SingleInputFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Load
        loaded_docs = await SingleInputFlowConfig.load_documents(tmpdir)

        assert len(loaded_docs) == 1
        loaded = loaded_docs[0]

        # Verify content preserved exactly
        assert loaded.content == doc.content
        assert loaded.description == doc.description
        assert loaded.sources == doc.sources

        # Verify JSON is still valid
        data = json.loads(loaded.content)
        assert data["text"] == "Special chars: 🎉 ñ é 中文"


# ============================================================================
# Edge cases and error handling
# ============================================================================


@pytest.mark.asyncio
async def test_save_load_empty_document_list():
    """Test saving and loading empty document lists."""
    docs = DocumentList()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save empty list
        await SampleFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Load from directory (should be empty)
        loaded_docs = await SampleFlowConfig.load_documents(tmpdir)

        assert len(loaded_docs) == 0


@pytest.mark.asyncio
async def test_load_documents_with_unrelated_types():
    """Test loading documents that don't match requested types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create UnrelatedDoc (not in INPUT_DOCUMENT_TYPES)
        unrelated_dir = Path(tmpdir) / "unrelateddoc"
        unrelated_dir.mkdir()
        (unrelated_dir / "unrelated.txt").write_bytes(b"unrelated data")

        # Try to load with SampleFlowConfig (which doesn't include UnrelatedDoc)
        docs = await SampleFlowConfig.load_documents(tmpdir)

        # Should not load UnrelatedDoc
        assert len(docs) == 0


@pytest.mark.asyncio
async def test_canonical_name_directory_structure():
    """Test that documents are saved in canonical name directories."""
    doc1 = InputDoc(name="file.txt", content=b"data")
    doc2 = SecondInputDoc(name="file2.txt", content=b"data2")
    doc3 = OutputDoc(name="file3.txt", content=b"data3")

    docs = DocumentList([doc1, doc2, doc3])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Save documents
        await SampleFlowConfig.save_documents(tmpdir, docs, validate_output_type=False)

        # Verify directory structure
        assert (path / "inputdoc").is_dir()
        assert (path / "secondinputdoc").is_dir()
        assert (path / "outputdoc").is_dir()

        assert (path / "inputdoc" / "file.txt").exists()
        assert (path / "secondinputdoc" / "file2.txt").exists()
        assert (path / "outputdoc" / "file3.txt").exists()


@pytest.mark.asyncio
async def test_load_documents_file_read_error():
    """Test handling of file read errors during loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory where a file should be
        input_dir = Path(tmpdir) / "inputdoc"
        input_dir.mkdir()

        # Create a directory with the name of what should be a file
        # This will cause an error when trying to read it as a file
        (input_dir / "bad_file.txt").mkdir()

        # Should handle the error gracefully and return empty list
        docs = await SingleInputFlowConfig.load_documents(tmpdir)
        assert len(docs) == 0


@pytest.mark.asyncio
async def test_validation_with_mixed_output_types():
    """Test validation correctly identifies wrong output types."""
    correct_doc = OutputDoc(name="correct.txt", content=b"correct")
    wrong_doc = InputDoc(name="wrong.txt", content=b"wrong")

    mixed_docs = DocumentList([correct_doc, wrong_doc])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should fail validation
        with pytest.raises(DocumentValidationError) as exc_info:
            await SampleFlowConfig.save_documents(tmpdir, mixed_docs, validate_output_type=True)

        assert "wrong.txt" in str(exc_info.value)
        assert "InputDoc" in str(exc_info.value)
