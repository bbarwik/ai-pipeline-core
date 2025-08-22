"""Tests for FlowConfig."""

import pytest

from ai_pipeline_core.documents import DocumentList, FlowDocument
from ai_pipeline_core.flow import FlowConfig


class InputDoc1(FlowDocument):
    """First input document type."""

    def get_type(self) -> str:
        return "input1"


class InputDoc2(FlowDocument):
    """Second input document type."""

    def get_type(self) -> str:
        return "input2"


class OutputDoc(FlowDocument):
    """Output document type."""

    def get_type(self) -> str:
        return "output"


class WrongOutputDoc(FlowDocument):
    """Wrong output document type."""

    def get_type(self) -> str:
        return "wrong_output"


class TestFlowConfig(FlowConfig):
    """Test flow configuration."""

    INPUT_DOCUMENT_TYPES = [InputDoc1, InputDoc2]
    OUTPUT_DOCUMENT_TYPE = OutputDoc


class TestFlowConfigMethods:
    """Test FlowConfig methods."""

    def test_get_input_document_types(self):
        """Test getting input document types."""
        types = TestFlowConfig.get_input_document_types()
        assert types == [InputDoc1, InputDoc2]

    def test_get_output_document_type(self):
        """Test getting output document type."""
        output_type = TestFlowConfig.get_output_document_type()
        assert output_type == OutputDoc

    def test_has_input_documents_all_present(self):
        """Test has_input_documents when all required docs are present."""
        doc1 = InputDoc1(name="doc1.txt", content=b"content1")
        doc2 = InputDoc2(name="doc2.txt", content=b"content2")
        documents = DocumentList([doc1, doc2])

        assert TestFlowConfig.has_input_documents(documents) is True

    def test_has_input_documents_missing_one(self):
        """Test has_input_documents when one required doc is missing."""
        doc1 = InputDoc1(name="doc1.txt", content=b"content1")
        # Missing InputDoc2
        documents = DocumentList([doc1])

        assert TestFlowConfig.has_input_documents(documents) is False

    def test_has_input_documents_empty(self):
        """Test has_input_documents with empty document list."""
        documents = DocumentList()
        assert TestFlowConfig.has_input_documents(documents) is False

    def test_has_input_documents_with_extra(self):
        """Test has_input_documents with extra documents."""
        doc1 = InputDoc1(name="doc1.txt", content=b"content1")
        doc2 = InputDoc2(name="doc2.txt", content=b"content2")
        extra = OutputDoc(name="extra.txt", content=b"extra")
        documents = DocumentList([doc1, doc2, extra])

        # Should still return True as required docs are present
        assert TestFlowConfig.has_input_documents(documents) is True

    def test_get_input_documents_success(self):
        """Test getting input documents when all are present."""
        doc1 = InputDoc1(name="doc1.txt", content=b"content1")
        doc2 = InputDoc2(name="doc2.txt", content=b"content2")
        extra = OutputDoc(name="extra.txt", content=b"extra")
        documents = DocumentList([doc1, extra, doc2])

        input_docs = TestFlowConfig.get_input_documents(documents)

        assert isinstance(input_docs, DocumentList)
        assert len(input_docs) == 2
        assert doc1 in input_docs
        assert doc2 in input_docs
        assert extra not in input_docs

    def test_get_input_documents_missing_type(self):
        """Test getting input documents when a type is missing."""
        doc1 = InputDoc1(name="doc1.txt", content=b"content1")
        # Missing InputDoc2
        documents = DocumentList([doc1])

        with pytest.raises(ValueError) as exc_info:
            TestFlowConfig.get_input_documents(documents)

        assert "No input document found for class InputDoc2" in str(exc_info.value)

    def test_get_input_documents_multiple_of_same_type(self):
        """Test getting input documents with multiple docs of same type."""
        doc1a = InputDoc1(name="doc1a.txt", content=b"content1a")
        doc1b = InputDoc1(name="doc1b.txt", content=b"content1b")
        doc2 = InputDoc2(name="doc2.txt", content=b"content2")
        documents = DocumentList([doc1a, doc1b, doc2])

        input_docs = TestFlowConfig.get_input_documents(documents)

        # Should include all documents of the required types
        assert len(input_docs) == 3
        assert doc1a in input_docs
        assert doc1b in input_docs
        assert doc2 in input_docs

    def test_validate_output_documents_valid(self):
        """Test validating output documents with correct type."""
        out1 = OutputDoc(name="out1.txt", content=b"output1")
        out2 = OutputDoc(name="out2.txt", content=b"output2")
        documents = DocumentList([out1, out2])

        # Should not raise
        TestFlowConfig.validate_output_documents(documents)

    def test_validate_output_documents_invalid_type(self):
        """Test validating output documents with wrong type."""
        out1 = OutputDoc(name="out1.txt", content=b"output1")
        wrong = WrongOutputDoc(name="wrong.txt", content=b"wrong")
        documents = DocumentList([out1, wrong])

        with pytest.raises(AssertionError) as exc_info:
            TestFlowConfig.validate_output_documents(documents)

        error_msg = str(exc_info.value)
        assert "Documents must be of the correct type" in error_msg
        assert "Expected: OutputDoc" in error_msg
        assert "WrongOutputDoc" in error_msg

    def test_validate_output_documents_not_document_list(self):
        """Test that validate requires a DocumentList."""
        out1 = OutputDoc(name="out1.txt", content=b"output1")
        regular_list = [out1]  # Not a DocumentList

        with pytest.raises(AssertionError) as exc_info:
            TestFlowConfig.validate_output_documents(regular_list)  # type: ignore

        assert "Documents must be a DocumentList" in str(exc_info.value)

    def test_validate_output_documents_empty(self):
        """Test validating empty output documents."""
        documents = DocumentList()

        # Empty list should pass validation (no invalid types)
        TestFlowConfig.validate_output_documents(documents)

    def test_validate_output_documents_mixed_invalid(self):
        """Test validation error message includes all invalid types."""
        out1 = OutputDoc(name="out1.txt", content=b"output1")
        wrong1 = WrongOutputDoc(name="wrong1.txt", content=b"wrong1")
        wrong2 = InputDoc1(name="wrong2.txt", content=b"wrong2")
        documents = DocumentList([out1, wrong1, wrong2])

        with pytest.raises(AssertionError) as exc_info:
            TestFlowConfig.validate_output_documents(documents)

        error_msg = str(exc_info.value)
        assert "WrongOutputDoc" in error_msg
        assert "InputDoc1" in error_msg


class SingleInputFlowConfig(FlowConfig):
    """Flow config with single input type."""

    INPUT_DOCUMENT_TYPES = [InputDoc1]
    OUTPUT_DOCUMENT_TYPE = OutputDoc


class TestSingleInputFlow:
    """Test flow with single input document type."""

    def test_single_input_type(self):
        """Test flow with single input document type."""
        doc1 = InputDoc1(name="doc1.txt", content=b"content1")
        documents = DocumentList([doc1])

        assert SingleInputFlowConfig.has_input_documents(documents) is True

        input_docs = SingleInputFlowConfig.get_input_documents(documents)
        assert len(input_docs) == 1
        assert doc1 in input_docs


class TestFlowConfigValidation:
    """Test FlowConfig validation rules."""

    def test_output_type_not_in_input_types_raises_error(self):
        """Test that OUTPUT_DOCUMENT_TYPE cannot be in INPUT_DOCUMENT_TYPES."""

        with pytest.raises(TypeError) as exc_info:

            class InvalidFlowConfig(FlowConfig):  # pyright: ignore[reportUnusedClass]
                """Flow config with output type in input types."""

                INPUT_DOCUMENT_TYPES = [InputDoc1, OutputDoc]  # OutputDoc is also the output
                OUTPUT_DOCUMENT_TYPE = OutputDoc

        assert "OUTPUT_DOCUMENT_TYPE" in str(exc_info.value)
        assert "cannot be in INPUT_DOCUMENT_TYPES" in str(exc_info.value)
        assert "OutputDoc" in str(exc_info.value)

    def test_valid_config_does_not_raise(self):
        """Test that valid config with different input and output types works."""

        # This should not raise
        class ValidFlowConfig(FlowConfig):
            """Valid flow config."""

            INPUT_DOCUMENT_TYPES = [InputDoc1, InputDoc2]
            OUTPUT_DOCUMENT_TYPE = OutputDoc

        # Should be able to use the class normally
        assert ValidFlowConfig.get_output_document_type() == OutputDoc
        assert ValidFlowConfig.get_input_document_types() == [InputDoc1, InputDoc2]
