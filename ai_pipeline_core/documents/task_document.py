"""Task-specific document base class for temporary pipeline data.

@public

This module provides the TaskDocument abstract base class for documents
that exist only during Prefect task execution and are not persisted.
"""

from typing import Any, Literal, final

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
    - More memory-efficient for temporary data

    Usage:
        Always subclass TaskDocument for temporary document types:

        >>> class TempProcessingDoc(TaskDocument):
        ...     def get_type(self) -> str:
        ...         return "temp_processing"
        >>> doc = TempProcessingDoc(name="temp.json", content=b'{}')

    Note:
        - Cannot instantiate TaskDocument directly - must subclass
        - Not saved by simple_runner utilities
        - Useful for data transformations within tasks
        - Reduces I/O overhead for temporary data

    See Also:
        FlowDocument: For documents that persist across flow runs
        TemporaryDocument: Alternative for non-persistent documents
    """

    def __init__(self, **data: Any) -> None:
        """Initialize a TaskDocument instance.

        Prevents direct instantiation of the abstract TaskDocument class.
        TaskDocument must be subclassed for specific temporary document types.

        Args:
            **data: Keyword arguments for name, description, and content fields.

        Raises:
            TypeError: If attempting to instantiate TaskDocument directly
                      instead of using a concrete subclass.

        Example:
            >>> # This will raise TypeError:
            >>> doc = TaskDocument(name="temp.txt", content=b"data")
            >>> # This is correct:
            >>> class MyTaskDoc(TaskDocument): ...
            >>> doc = MyTaskDoc(name="temp.txt", content=b"data")
        """
        if type(self) is TaskDocument:
            raise TypeError("Cannot instantiate abstract TaskDocument class directly")
        super().__init__(**data)

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
