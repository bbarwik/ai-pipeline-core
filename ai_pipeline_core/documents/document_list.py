"""Type-safe list container for Document objects.

@public
"""

from typing import Any, Iterable, SupportsIndex, Union, overload

from typing_extensions import Self

from .document import Document


class DocumentList(list[Document]):
    """Type-safe container for Document objects.

    @public

    Specialized list with validation and filtering for documents.

    Example:
        >>> docs = DocumentList(validate_same_type=True)
        >>> docs.append(MyDocument(name="file.txt", content=b"data"))
        >>> doc = docs.get_by_name("file.txt")
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        validate_same_type: bool = False,
        validate_duplicates: bool = False,
    ) -> None:
        """Initialize DocumentList.

        @public

        Args:
            documents: Initial list of documents.
            validate_same_type: Enforce same document type.
            validate_duplicates: Prevent duplicate filenames.
        """
        super().__init__()
        self._validate_same_type = validate_same_type
        self._validate_duplicates = validate_duplicates
        if documents:
            self.extend(documents)

    def _validate_no_duplicates(self) -> None:
        """Check for duplicate document names.

        Raises:
            ValueError: If duplicate document names are found.
        """
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
        """Ensure no documents use reserved description file extension.

        Raises:
            ValueError: If any document uses the reserved description file extension.
        """
        description_files = [
            doc.name for doc in self if doc.name.endswith(Document.DESCRIPTION_EXTENSION)
        ]
        if description_files:
            raise ValueError(
                f"Documents with {Document.DESCRIPTION_EXTENSION} suffix are not allowed: "
                f"{description_files}"
            )

    def _validate_types(self) -> None:
        """Ensure all documents are of the same class type.

        Raises:
            ValueError: If documents have different class types.
        """
        if not self._validate_same_type or not self:
            return

        first_class = type(self[0])
        different_types = [doc for doc in self if type(doc) is not first_class]
        if different_types:
            types = list({type(doc).__name__ for doc in self})
            raise ValueError(f"All documents must have the same type. Found types: {types}")

    def _validate(self) -> None:
        """Run all configured validation checks."""
        self._validate_no_duplicates()
        self._validate_no_description_files()
        self._validate_types()

    def append(self, document: Document) -> None:
        """Add a document to the end of the list."""
        super().append(document)
        self._validate()

    def extend(self, documents: Iterable[Document]) -> None:
        """Add multiple documents to the list."""
        super().extend(documents)
        self._validate()

    def insert(self, index: SupportsIndex, document: Document) -> None:
        """Insert a document at the specified position."""
        super().insert(index, document)
        self._validate()

    @overload
    def __setitem__(self, index: SupportsIndex, value: Document) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Document]) -> None: ...

    def __setitem__(self, index: Union[SupportsIndex, slice], value: Any) -> None:
        """Set item or slice with validation."""
        super().__setitem__(index, value)
        self._validate()

    def __iadd__(self, other: Any) -> "Self":
        """In-place addition (+=) with validation.

        Returns:
            Self: This DocumentList after modification.
        """
        result = super().__iadd__(other)
        self._validate()
        return result

    def filter_by_type(self, document_type: type[Document]) -> "DocumentList":
        """Filter documents by class type (including subclasses).

        @public

        Args:
            document_type: The document class to filter for.

        Returns:
            New DocumentList with filtered documents.
        """
        return DocumentList([doc for doc in self if isinstance(doc, document_type)])

    def filter_by_types(self, document_types: list[type[Document]]) -> "DocumentList":
        """Filter documents by multiple class types.

        @public

        Args:
            document_types: List of document classes to filter for.

        Returns:
            New DocumentList with filtered documents.
        """
        documents = DocumentList()
        for document_type in document_types:
            documents.extend(self.filter_by_type(document_type))
        return documents

    def get_by_name(self, name: str) -> Document | None:
        """Find a document by filename.

        @public

        Args:
            name: The document filename to search for.

        Returns:
            The first matching document, or None if not found.
        """
        for doc in self:
            if doc.name == name:
                return doc
        return None
