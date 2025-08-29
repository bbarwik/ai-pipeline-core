"""Temporary document implementation for non-persistent data.

@public

This module provides the TemporaryDocument class for documents that
are never persisted, regardless of context.
"""

from typing import Literal, final

from .document import Document


@final
class TemporaryDocument(Document):
    r"""Concrete document class for data that is never persisted.

    @public

    TemporaryDocument is a final (non-subclassable) document type for
    data that should never be saved to disk, regardless of whether it's
    used in a flow or task context. Unlike FlowDocument and TaskDocument
    which are abstract, TemporaryDocument can be instantiated directly.

    Key characteristics:
    - Never persisted to file system
    - Can be instantiated directly (not abstract)
    - Cannot be subclassed (annotated with Python's @final decorator in code)
    - Useful for transient data like API responses or intermediate calculations
    - Ignored by simple_runner save operations

    Creating TemporaryDocuments:
        **Use the `create` classmethod** for most use cases. It handles automatic
        conversion of various content types. Only use __init__ when you have bytes.

        >>> # RECOMMENDED - automatic conversion:
        >>> doc = TemporaryDocument.create(
        ...     name="api_response.json",
        ...     content={"status": "ok", "data": [1, 2, 3]}
        ... )
        >>> doc = TemporaryDocument.create(
        ...     name="credentials.txt",
        ...     content="secret_token_xyz"
        ... )
        >>>
        >>> # Direct constructor - only for bytes:
        >>> doc = TemporaryDocument(
        ...     name="binary.dat",
        ...     content=b"\x00\x01\x02"
        ... )
        >>>
        >>> doc.is_temporary  # Always True

    Use Cases:
        - API responses that shouldn't be cached
        - Sensitive credentials or tokens
        - Intermediate calculations
        - Temporary transformations
        - Data explicitly marked as non-persistent

    Note:
        - This is a final class and cannot be subclassed
        - Use when you explicitly want to prevent persistence
        - Useful for sensitive data that shouldn't be written to disk

    See Also:
        FlowDocument: For documents that persist across flow runs
        TaskDocument: For documents temporary within task execution
    """

    def get_base_type(self) -> Literal["temporary"]:
        """Return the base type identifier for temporary documents.

        Identifies this document as temporary, ensuring it will
        never be persisted by the pipeline system.

        Returns:
            "temporary" - Indicates this document is never persisted.

        Note:
            Documents with this type are explicitly excluded from
            all persistence operations in the pipeline system.
        """
        return "temporary"
