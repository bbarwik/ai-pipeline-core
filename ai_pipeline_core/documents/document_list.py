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

    Best Practice: Use default constructor in 90% of cases. Only enable
    validate_same_type or validate_duplicates when you explicitly need them.

    Example:
        >>> # RECOMMENDED - default constructor for most cases
        >>> docs = DocumentList([doc1, doc2])
        >>> # Or empty initialization
        >>> docs = DocumentList()
        >>> docs.append(MyDocument(name="file.txt", content=b"data"))
        >>>
        >>> # Only use validation flags when specifically needed:
        >>> docs = DocumentList(validate_same_type=True)  # Rare use case
        >>> doc = docs.get_by("file.txt")  # Get by name
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

    @overload
    def filter_by(self, arg: str) -> "DocumentList": ...

    @overload
    def filter_by(self, arg: type[Document]) -> "DocumentList": ...

    @overload
    def filter_by(self, arg: Iterable[type[Document]]) -> "DocumentList": ...

    @overload
    def filter_by(self, arg: Iterable[str]) -> "DocumentList": ...

    def filter_by(
        self, arg: str | type[Document] | Iterable[type[Document]] | Iterable[str]
    ) -> "DocumentList":
        """Filter documents by name(s) or type(s).

        @public

        Args:
            arg: Can be one of:
                - str: Single document name to filter by
                - type[Document]: Single document type to filter by (includes subclasses)
                - Iterable[type[Document]]: Multiple document types to filter by
                  (list, tuple, set, generator, or any iterable)
                - Iterable[str]: Multiple document names to filter by
                  (list, tuple, set, generator, or any iterable)

        Returns:
            New DocumentList with filtered documents.

        Raises:
            TypeError: If arg is not a valid type (not str, type, or iterable),
                or if iterable contains mixed types (strings and types together).
            AttributeError: If arg is expected to be iterable but doesn't support iteration.

        Example:
            >>> docs.filter_by("file.txt")  # Filter by single name
            >>> docs.filter_by(MyDocument)  # Filter by single type
            >>> docs.filter_by([Doc1, Doc2])  # Filter by multiple types (list)
            >>> docs.filter_by({"file1.txt", "file2.txt"})  # Filter by multiple names (set)
            >>> docs.filter_by((SubDoc, AnotherDoc))  # Filter by multiple types (tuple)
            >>> docs.filter_by(name for name in ["a.txt", "b.txt"])  # Generator expression
        """
        if isinstance(arg, str):
            # Filter by single name
            return DocumentList([doc for doc in self if doc.name == arg])
        elif isinstance(arg, type):
            # Filter by single type (including subclasses)
            # The type system ensures arg is type[Document] due to overloads
            return DocumentList([doc for doc in self if isinstance(doc, arg)])
        else:
            # Try to consume as iterable
            try:
                # Convert to list to check the first element and allow reuse
                items = list(arg)  # type: ignore[arg-type]
                if not items:
                    return DocumentList()

                first_item = items[0]
                if isinstance(first_item, str):
                    # Iterable of names - validate all items are strings
                    for item in items:
                        if not isinstance(item, str):
                            raise TypeError(
                                "Iterable must contain only strings or only Document types, "
                                "not mixed types"
                            )
                    names_set = set(items)
                    return DocumentList([doc for doc in self if doc.name in names_set])
                elif isinstance(first_item, type):  # type: ignore[reportUnnecessaryIsInstance]
                    # Iterable of document types - validate all items are types
                    for item in items:
                        if not isinstance(item, type):
                            raise TypeError(
                                "Iterable must contain only strings or only Document types, "
                                "not mixed types"
                            )
                    # Convert to set for efficient lookup
                    types_set = set(items)
                    # Filter documents that match any of the requested types
                    matching = [
                        doc
                        for doc in self
                        if any(isinstance(doc, doc_type) for doc_type in types_set)  # type: ignore[arg-type]
                    ]
                    return DocumentList(matching)
                else:
                    raise TypeError(
                        f"Iterable must contain strings or Document types, "
                        f"got {type(first_item).__name__}"
                    )
            except (TypeError, AttributeError) as e:
                # If the error message already mentions Iterable, re-raise it
                if "Iterable" in str(e) or "strings or Document types" in str(e):
                    raise
                # Otherwise, provide a generic error message
                raise TypeError(f"Invalid argument type for filter_by: {type(arg).__name__}") from e

    @overload
    def get_by(self, arg: str) -> Document: ...

    @overload
    def get_by(self, arg: type[Document]) -> Document: ...

    @overload
    def get_by(self, arg: str, required: bool = True) -> Document | None: ...

    @overload
    def get_by(self, arg: type[Document], required: bool = True) -> Document | None: ...

    def get_by(self, arg: str | type[Document], required: bool = True) -> Document | None:
        """Get a single document by name or type.

        @public

        Args:
            arg: Document name (str) or document type.
            required: If True, raises ValueError when not found. If False, returns None.

        Returns:
            The first matching document, or None if not found and required=False.

        Raises:
            ValueError: If required=True and document not found.
            TypeError: If arg is not a string or Document type.

        Example:
            >>> doc = docs.get_by("file.txt")  # Get by name, raises if not found
            >>> doc = docs.get_by(MyDocument, required=False)  # Returns None if not found
        """
        if isinstance(arg, str):
            # Get by name
            for doc in self:
                if doc.name == arg:
                    return doc
            if required:
                raise ValueError(f"Document with name '{arg}' not found")
            return None
        elif isinstance(arg, type):  # type: ignore[reportUnnecessaryIsInstance]
            # Get by type (including subclasses)
            for doc in self:
                if isinstance(doc, arg):
                    return doc
            if required:
                raise ValueError(f"Document of type '{arg.__name__}' not found")
            return None
        else:
            raise TypeError(f"Invalid argument type for get_by: {type(arg)}")
