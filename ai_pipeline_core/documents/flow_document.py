"""Flow-specific document base class."""

from typing import Any, Literal, final

from .document import Document


class FlowDocument(Document):
    """
    Abstract base class for flow-specific documents.

    Flow documents represent inputs, outputs, and intermediate results
    within a Prefect flow execution context.

    Compared to TaskDocument, FlowDocument are persistent across Prefect flow runs.
    """

    def __init__(self, **data: Any) -> None:
        """Prevent direct instantiation of abstract FlowDocument class."""
        if type(self) is FlowDocument:
            raise TypeError("Cannot instantiate abstract FlowDocument class directly")
        super().__init__(**data)

    @final
    def get_base_type(self) -> Literal["flow"]:
        """Get the document type."""
        return "flow"
