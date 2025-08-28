"""Flow-specific document base class for persistent pipeline data.

@public

This module provides the FlowDocument abstract base class for documents
that need to persist across Prefect flow runs and between pipeline steps.
"""

from typing import Any, Literal, final

from .document import Document


class FlowDocument(Document):
    """Abstract base class for documents that persist across flow runs.
    
    @public

    FlowDocument is used for data that needs to be saved between pipeline
    steps and across multiple flow executions. These documents are typically
    written to the file system using the simple_runner utilities.

    Key characteristics:
    - Persisted to file system between pipeline steps
    - Survives across multiple flow runs
    - Used for flow inputs and outputs
    - Saved in directories named after the document's canonical name

    Usage:
        Always subclass FlowDocument for your specific document types:

        >>> class InputDataDocument(FlowDocument):
        ...     def get_type(self) -> str:
        ...         return "input_data"
        >>> doc = InputDataDocument(name="data.json", content=b'{}')

    Note:
        - Cannot instantiate FlowDocument directly - must subclass
        - Documents are saved to {output_dir}/{canonical_name}/{filename}
        - Used with FlowConfig to define flow input/output types

    See Also:
        TaskDocument: For temporary documents within task execution
        TemporaryDocument: For documents that are never persisted
    """

    def __init__(self, **data: Any) -> None:
        """Initialize a FlowDocument instance.

        Prevents direct instantiation of the abstract FlowDocument class.
        FlowDocument must be subclassed for specific document types.

        Args:
            **data: Keyword arguments for name, description, and content fields.

        Raises:
            TypeError: If attempting to instantiate FlowDocument directly
                      instead of using a concrete subclass.

        Example:
            >>> # This will raise TypeError:
            >>> doc = FlowDocument(name="test.txt", content=b"data")
            >>> # This is correct:
            >>> class MyFlowDoc(FlowDocument): ...
            >>> doc = MyFlowDoc(name="test.txt", content=b"data")
        """
        if type(self) is FlowDocument:
            raise TypeError("Cannot instantiate abstract FlowDocument class directly")
        super().__init__(**data)

    @final
    def get_base_type(self) -> Literal["flow"]:
        """Return the base type identifier for flow documents.

        This method is final and cannot be overridden by subclasses.
        It identifies this document as a flow-persistent document.

        Returns:
            "flow" - Indicates this document persists across flow runs.

        Note:
            This determines the document's lifecycle and persistence behavior
            in the pipeline system.
        """
        return "flow"
