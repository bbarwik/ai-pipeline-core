"""Task-specific document base class."""

from typing import Literal, final

from .document import Document


class TaskDocument(Document):
    """
    Abstract base class for task-specific documents.

    Task documents represent inputs, outputs, and intermediate results
    within a Prefect task execution context.

    Compared to FlowDocument, TaskDocument are not persisted across Prefect task runs.
    They are used for intermediate results that are not needed after the task completes.
    """

    @final
    def get_base_type(self) -> Literal["task"]:
        """Get the document type."""
        return "task"
