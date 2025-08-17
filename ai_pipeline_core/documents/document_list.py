from typing import Any, Iterable, SupportsIndex, Union, overload

from typing_extensions import Self

from .document import Document


class DocumentList(list[Document]):
    """
    A specialized list for Document objects with built-in validation.

    Features:
    - Optionally ensures no duplicate filenames within the list
    - Optionally validates that all documents have the same type (for flow outputs)
    - Provides convenience methods for document operations
    - Works with both FlowDocument and TaskDocument classes
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        validate_same_type: bool = False,
        validate_duplicates: bool = False,
    ) -> None:
        """
        Initialize DocumentList with optional initial documents.

        Args:
                documents: Initial list of documents
                validate_same_type: If True, validates that all documents have the same type.
                                                 Should be True for flow outputs, False for inputs.
        """
        super().__init__()
        self._validate_same_type = validate_same_type
        self._validate_duplicates = validate_duplicates
        if documents:
            self.extend(documents)

    def _validate_no_duplicates(self) -> None:
        """Validate that there are no duplicate filenames."""
        if not self._validate_duplicates:
            return

        filenames = [doc.name for doc in self]
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in filenames:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            unique_duplicates = list(set(duplicates))
            raise ValueError(f"Duplicate document names found: {unique_duplicates}")

    def _validate_no_description_files(self) -> None:
        """Validate that no documents have DESCRIPTION_EXTENSION suffix."""
        description_files = [
            doc.name for doc in self if doc.name.endswith(Document.DESCRIPTION_EXTENSION)
        ]
        if description_files:
            raise ValueError(
                f"Documents with {Document.DESCRIPTION_EXTENSION} suffix are not allowed: "
                f"{description_files}"
            )

    def _validate_types(self) -> None:
        """Validate that all documents have the same class type if required."""
        if not self._validate_same_type or not self:
            return

        first_class = type(self[0])
        different_types = [doc for doc in self if type(doc) is not first_class]
        if different_types:
            types = list({type(doc).__name__ for doc in self})
            raise ValueError(f"All documents must have the same type. Found types: {types}")

    def _validate(self) -> None:
        """Run all validations."""
        self._validate_no_duplicates()
        self._validate_no_description_files()
        self._validate_types()

    def append(self, document: Document) -> None:
        """Add a document to the list with validation."""
        super().append(document)
        self._validate()

    def extend(self, documents: Iterable[Document]) -> None:
        """Extend the list with multiple documents with validation."""
        super().extend(documents)
        self._validate()

    def insert(self, index: SupportsIndex, document: Document) -> None:
        """Insert a document at the specified index with validation."""
        super().insert(index, document)
        self._validate()

    @overload
    def __setitem__(self, index: SupportsIndex, value: Document) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Document]) -> None: ...

    def __setitem__(self, index: Union[SupportsIndex, slice], value: Any) -> None:
        """Set item with validation."""
        super().__setitem__(index, value)
        self._validate()

    def __iadd__(self, other: Any) -> "Self":
        """In-place addition with validation."""
        result = super().__iadd__(other)
        self._validate()
        return result

    def filter_by_type(self, document_type: type[Document]) -> "DocumentList":
        """Return a new DocumentList containing only instances of the specified document class."""
        return DocumentList([doc for doc in self if type(doc) is document_type])

    def filter_by_types(self, document_types: list[type[Document]]) -> "DocumentList":
        """Return a new DocumentList containing only instances of the specified document classes."""
        documents = DocumentList()
        for document_type in document_types:
            documents.extend(self.filter_by_type(document_type))
        return documents

    def get_by_name(self, name: str) -> Document | None:
        """Get a document by its name."""
        for doc in self:
            if doc.name == name:
                return doc
        return None
