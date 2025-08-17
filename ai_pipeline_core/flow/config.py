"""Flow configuration base class."""

from abc import ABC
from typing import ClassVar

from ai_pipeline_core.documents import DocumentList, FlowDocument


class FlowConfig(ABC):
    """
    Configuration for a flow. It makes flow easier to implement and test.
    """

    INPUT_DOCUMENT_TYPES: ClassVar[list[type[FlowDocument]]]
    OUTPUT_DOCUMENT_TYPE: ClassVar[type[FlowDocument]]

    @classmethod
    def get_input_document_types(cls) -> list[type[FlowDocument]]:
        """
        Get the input document types for the flow.
        """
        return cls.INPUT_DOCUMENT_TYPES

    @classmethod
    def get_output_document_type(cls) -> type[FlowDocument]:
        """
        Get the output document type for the flow.
        """
        return cls.OUTPUT_DOCUMENT_TYPE

    @classmethod
    def has_input_documents(cls, documents: DocumentList) -> bool:
        """
        Check if the flow has all required input documents.
        """
        for doc_cls in cls.INPUT_DOCUMENT_TYPES:
            if not any(isinstance(doc, doc_cls) for doc in documents):
                return False
        return True

    @classmethod
    def get_input_documents(cls, documents: DocumentList) -> DocumentList:
        """
        Get the input documents for the flow.
        """
        input_documents = DocumentList()
        for doc_cls in cls.INPUT_DOCUMENT_TYPES:
            filtered_documents = [doc for doc in documents if isinstance(doc, doc_cls)]
            if not filtered_documents:
                raise ValueError(f"No input document found for class {doc_cls.__name__}")
            input_documents.extend(filtered_documents)
        return input_documents

    @classmethod
    def validate_output_documents(cls, documents: DocumentList) -> None:
        """
        Validate the output documents of the flow.
        """
        assert isinstance(documents, DocumentList), "Documents must be a DocumentList"
        output_document_class = cls.get_output_document_type()

        invalid = [type(d).__name__ for d in documents if not isinstance(d, output_document_class)]
        assert not invalid, (
            "Documents must be of the correct type. "
            f"Expected: {output_document_class.__name__}, Got invalid: {invalid}"
        )
