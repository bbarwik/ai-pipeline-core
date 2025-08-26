"""Task-specific document base class."""

from typing import Literal, final

from .document import Document


@final
class TemporaryDocument(Document):
    """
    Temporary document is a document that is not persisted in any case.
    """

    def get_base_type(self) -> Literal["temporary"]:
        """Get the document type."""
        return "temporary"
