"""Type-safe list container for Document objects with validation.

@public

This module provides the DocumentList class that ensures all items are valid Document
instances while maintaining list-like interface.
"""

from typing import Any, Iterable, SupportsIndex, Union, overload

from typing_extensions import Self

from .document import Document


class DocumentList(list[Document]):
    """Type-safe container for Document objects with validation.
    
    @public

    A specialized list that ensures document integrity and provides
    convenient filtering and search operations. Used throughout the
    pipeline system to pass documents between flows and tasks.

    Key features:
    - Optional duplicate filename validation
    - Optional same-type validation (for flow outputs)
    - Automatic validation on all modifications
    - Filtering by document type
    - Lookup by document name

    Attributes:
        _validate_same_type: Whether to enforce all documents have same type
        _validate_duplicates: Whether to prevent duplicate filenames

    Example:
        >>> docs = DocumentList(validate_same_type=True)
        >>> docs.append(MyDocument(name="file1.txt", content=b"data"))
        >>> docs.append(MyDocument(name="file2.txt", content=b"more"))
        >>> doc = docs.get_by_name("file1.txt")

    Note:
        Validation is performed automatically after every modification.
        For flow outputs, use validate_same_type=True to ensure consistency.
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        validate_same_type: bool = False,
        validate_duplicates: bool = False,
    ) -> None:
        """Initialize DocumentList with optional documents and validation settings.

        Args:
            documents: Initial list of documents to add.
            validate_same_type: If True, enforces all documents have the same
                              class type. Should be True for flow outputs to
                              ensure type consistency.
            validate_duplicates: If True, prevents documents with duplicate
                               filenames from being added.

        Note:
            Validation errors from initial documents will raise ValueError.

        Example:
            >>> # For flow outputs - ensure same type
            >>> output_docs = DocumentList(validate_same_type=True)
            >>> # For general use - allow mixed types
            >>> mixed_docs = DocumentList([doc1, doc2, doc3])
        """
        super().__init__()
        self._validate_same_type = validate_same_type
        self._validate_duplicates = validate_duplicates
        if documents:
            self.extend(documents)

    def _validate_no_duplicates(self) -> None:
        """Check for duplicate document names in the list.

        Internal validation method that ensures no two documents
        have the same filename when validate_duplicates is True.

        Raises:
            ValueError: If duplicate filenames are found.
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
        """Ensure no documents use the reserved description file extension.

        Description files (.description.md) are handled separately
        and should not be included in the document list.

        Raises:
            ValueError: If any document name ends with .description.md.
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

        When validate_same_type is True, this ensures type consistency
        which is important for flow outputs.

        Raises:
            ValueError: If documents have different types and
                       validate_same_type is True.
        """
        if not self._validate_same_type or not self:
            return

        first_class = type(self[0])
        different_types = [doc for doc in self if type(doc) is not first_class]
        if different_types:
            types = list({type(doc).__name__ for doc in self})
            raise ValueError(f"All documents must have the same type. Found types: {types}")

    def _validate(self) -> None:
        """Run all configured validation checks.

        Called automatically after any list modification to ensure
        the document list remains in a valid state.

        Note:
            Calls validation methods that may raise ValueError.
        """
        self._validate_no_duplicates()
        self._validate_no_description_files()
        self._validate_types()

    def append(self, document: Document) -> None:
        """Add a document to the end of the list.

        Args:
            document: The document to add.

        Note:
            Validation errors will raise ValueError.
        """
        super().append(document)
        self._validate()

    def extend(self, documents: Iterable[Document]) -> None:
        """Add multiple documents to the list.

        Args:
            documents: Iterable of documents to add.

        Note:
            Validation errors will raise ValueError.
            All documents are added before validation,
            so on error the list may be partially modified.
        """
        super().extend(documents)
        self._validate()

    def insert(self, index: SupportsIndex, document: Document) -> None:
        """Insert a document at the specified position.

        Args:
            index: Position to insert at.
            document: Document to insert.

        Note:
            Validation errors will raise ValueError.
        """
        super().insert(index, document)
        self._validate()

    @overload
    def __setitem__(self, index: SupportsIndex, value: Document) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Document]) -> None: ...

    def __setitem__(self, index: Union[SupportsIndex, slice], value: Any) -> None:
        """Set item or slice with validation.

        Args:
            index: Index or slice to set.
            value: Document or iterable of documents.

        Note:
            Validation errors will raise ValueError.
        """
        super().__setitem__(index, value)
        self._validate()

    def __iadd__(self, other: Any) -> "Self":
        """In-place addition (+=) with validation.

        Args:
            other: Iterable of documents to add.

        Returns:
            Self for chaining.

        Note:
            Validation errors will raise ValueError.
        """
        result = super().__iadd__(other)
        self._validate()
        return result

    def filter_by_type(self, document_type: type[Document]) -> "DocumentList":
        """Filter documents by class type (including subclasses).
        
        @public

        Creates a new DocumentList containing documents that are
        instances of the specified type (includes subclasses).

        Args:
            document_type: The document class to filter for.

        Returns:
            New DocumentList with filtered documents.

        Example:
            >>> all_docs = DocumentList([flow_doc1, task_doc1, flow_doc2])
            >>> flow_docs = all_docs.filter_by_type(FlowDocument)
            >>> len(flow_docs)  # 2 (includes all FlowDocument subclasses)
        """
        return DocumentList([doc for doc in self if isinstance(doc, document_type)])

    def filter_by_types(self, document_types: list[type[Document]]) -> "DocumentList":
        """Filter documents by multiple class types.
        
        @public

        Creates a new DocumentList containing documents matching any
        of the specified types.

        Args:
            document_types: List of document classes to filter for.

        Returns:
            New DocumentList with filtered documents.

        Example:
            >>> docs = all_docs.filter_by_types([InputDoc, ConfigDoc])
        """
        documents = DocumentList()
        for document_type in document_types:
            documents.extend(self.filter_by_type(document_type))
        return documents

    def get_by_name(self, name: str) -> Document | None:
        """Find a document by filename.
        
        @public

        Searches for the first document with the specified name.

        Args:
            name: The document filename to search for.

        Returns:
            The first matching document, or None if not found.

        Example:
            >>> doc = docs.get_by_name("config.json")
            >>> if doc:
            ...     data = doc.as_json()
        """
        for doc in self:
            if doc.name == name:
                return doc
        return None
