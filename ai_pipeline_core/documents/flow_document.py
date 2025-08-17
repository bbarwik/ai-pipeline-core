"""Flow-specific document base class."""

from typing import Literal, final

from .document import Document


class FlowDocument(Document):
    """
    Abstract base class for flow-specific documents.

    Flow documents represent inputs, outputs, and intermediate results
    within a Prefect flow execution context.

    Compared to TaskDocument, FlowDocument are persistent across Prefect flow runs.
    """

    @final
    def get_base_type(self) -> Literal["flow"]:
        """Get the document type."""
        return "flow"
