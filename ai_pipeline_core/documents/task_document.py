"""Task-specific document base class for temporary pipeline data.

@public

This module provides the TaskDocument abstract base class for documents
that exist only during Prefect task execution and are not persisted.
"""

from typing import Literal, final

from .document import Document


class TaskDocument(Document):
    """Abstract base class for temporary documents within task execution.

    @public

    TaskDocument is used for intermediate data that exists only during
    the execution of a Prefect task and is not persisted to disk. These
    documents are ideal for temporary processing results, transformations,
    and data that doesn't need to survive beyond the current task.

    Key characteristics:
    - Not persisted to file system
    - Exists only during task execution
    - Garbage collected after task completes
    - Used for intermediate processing results
    - Reduces persistent I/O for temporary data

    Creating TaskDocuments:
        Same as Document - use `create()` for automatic conversion, `__init__` for bytes.
        See Document.create() for detailed usage examples.

    Use Cases:
        - Intermediate transformation results
        - Temporary buffers during processing
        - Task-local cache data
        - Processing status documents

    Note:
        - Cannot instantiate TaskDocument directly - must subclass
        - Not saved by simple_runner utilities
        - Reduces I/O overhead for temporary data
        - No additional abstract methods to implement

    See Also:
        FlowDocument: For documents that persist across flow runs
        TemporaryDocument: Alternative for non-persistent documents
    """

    def __init__(
        self,
        *,
        name: str,
        content: bytes,
        description: str | None = None,
        sources: list[str] = [],
    ) -> None:
        """Initialize a TaskDocument with raw bytes content.

        See Document.__init__() for parameter details and usage notes.

        Prevents direct instantiation of the abstract TaskDocument class.
        TaskDocument must be subclassed for specific temporary document types.

        Args:
            name: Document filename (required, keyword-only)
            content: Document content as raw bytes (required, keyword-only)
            description: Optional human-readable description (keyword-only)
            sources: Optional list of strings for provenance tracking

        Raises:
            TypeError: If attempting to instantiate TaskDocument directly
                      instead of using a concrete subclass.

        Example:
            >>> from enum import StrEnum
            >>>
            >>> # Simple subclass:
            >>> class MyTaskDoc(TaskDocument):
            ...     pass
            >>>
            >>> # With FILES restriction:
            >>> class TempProcessDoc(TaskDocument):
            ...     class FILES(StrEnum):
            ...         BUFFER = "buffer.bin"
            ...         STATUS = "status.json"
            >>>
            >>> # Direct constructor - only for bytes:
            >>> doc = MyTaskDoc(name="temp.bin", content=b"raw data")
            >>>
            >>> # RECOMMENDED - use create for automatic conversion:
            >>> doc = TempProcessDoc.create(name="status.json", content={"percent": 50})
            >>> # This would raise DocumentNameError:
            >>> # doc = TempProcessDoc.create(name="other.json", content={})
        """
        if type(self) is TaskDocument:
            raise TypeError("Cannot instantiate abstract TaskDocument class directly")
        super().__init__(name=name, content=content, description=description, sources=sources)

    @final
    def get_base_type(self) -> Literal["task"]:
        """Return the base type identifier for task documents.

        This method is final and cannot be overridden by subclasses.
        It identifies this document as a task-scoped temporary document.

        Returns:
            "task" - Indicates this document is temporary within task execution.

        Note:
            This determines that the document will not be persisted and
            exists only during task execution.
        """
        return "task"
