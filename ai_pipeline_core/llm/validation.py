"""Validation for LLM inputs.

Validates documents and attachments before sending to LLM to catch
empty, corrupted, or invalid content early. Filters invalid content
and logs warnings instead of failing the entire request.
"""

from io import BytesIO

from PIL import Image
from pypdf import PdfReader

from ai_pipeline_core.documents import Document
from ai_pipeline_core.documents.attachment import Attachment
from ai_pipeline_core.logging import get_pipeline_logger

from .ai_messages import AIMessages, AIMessageType

logger = get_pipeline_logger(__name__)


def _validate_image_content(content: bytes, name: str) -> str | None:
    """Validate image content. Returns error message or None if valid."""
    if not content:
        return f"empty image content in '{name}'"
    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()
        return None
    except Exception as e:
        return f"invalid image in '{name}': {e}"


def _validate_pdf_content(content: bytes, name: str) -> str | None:
    """Validate PDF content. Returns error message or None if valid."""
    if not content:
        return f"empty PDF content in '{name}'"

    # Check PDF header signature
    if not content.lstrip().startswith(b"%PDF-"):
        return f"invalid PDF header in '{name}' (missing %PDF- signature)"

    # Check page count - catches 0-page and corrupted PDFs
    try:
        reader = PdfReader(BytesIO(content))
        if len(reader.pages) == 0:
            return f"PDF has no pages in '{name}'"
    except Exception as e:
        return f"corrupted PDF in '{name}': {e}"

    return None


def _validate_text_content(content: bytes, name: str) -> str | None:
    """Validate text content. Returns error message or None if valid."""
    if not content:
        return f"empty text content in '{name}'"

    # Check for null bytes (indicates binary content)
    if b"\x00" in content:
        return f"binary content (null bytes) in text '{name}'"

    # Check UTF-8 encoding
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as e:
        return f"invalid UTF-8 encoding in '{name}': {e}"

    return None


def _validate_attachment(att: Attachment, parent_name: str) -> str | None:
    """Validate a single attachment. Returns error message or None if valid."""
    att_name = f"attachment '{att.name}' of '{parent_name}'"

    if att.is_image:
        return _validate_image_content(att.content, att_name)
    if att.is_pdf:
        return _validate_pdf_content(att.content, att_name)
    if att.is_text:
        return _validate_text_content(att.content, att_name)

    # Unknown type - let it through, document_to_prompt will handle/skip it
    return None


def _validate_document(doc: Document) -> tuple[Document | None, list[str]]:
    """Validate a document and its attachments.

    Returns:
        Tuple of (validated_document_or_None, list_of_error_messages).
        Returns None for document if main content is invalid.
        Filters out invalid attachments but keeps the document.
    """
    errors: list[str] = []

    # Validate main content based on type
    err: str | None = None
    if doc.is_image:
        err = _validate_image_content(doc.content, doc.name)
    elif doc.is_pdf:
        err = _validate_pdf_content(doc.content, doc.name)
    elif doc.is_text:
        err = _validate_text_content(doc.content, doc.name)
    # else: unknown type - let document_to_prompt handle it

    if err:
        errors.append(err)
        return None, errors

    # Validate attachments
    if not doc.attachments:
        return doc, errors

    valid_attachments: list[Attachment] = []
    attachments_changed = False

    for att in doc.attachments:
        if err := _validate_attachment(att, doc.name):
            errors.append(err)
            attachments_changed = True
        else:
            valid_attachments.append(att)

    if attachments_changed:
        # Return document with filtered attachments
        return doc.model_copy(update={"attachments": tuple(valid_attachments)}), errors

    return doc, errors


def validate_messages(messages: AIMessages) -> tuple[AIMessages, list[str]]:
    """Validate all documents in messages and filter out invalid content.

    Validates documents and their attachments. Invalid documents are removed
    entirely, invalid attachments are filtered from their parent documents.
    All validation errors are logged as warnings.

    Args:
        messages: AIMessages to validate.

    Returns:
        Tuple of (validated_messages, list_of_warning_messages).
        The validated_messages has invalid documents removed and invalid
        attachments filtered from remaining documents.
    """
    if not messages:
        return messages, []

    # Quick check: if no documents, nothing to validate
    has_documents = any(isinstance(m, Document) for m in messages)
    if not has_documents:
        return messages, []

    valid_msgs: list[AIMessageType] = []
    warnings: list[str] = []

    for msg in messages:
        if isinstance(msg, Document):
            valid_doc, doc_errors = _validate_document(msg)

            for err in doc_errors:
                warning_msg = f"LLM input validation: filtering {err}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)

            if valid_doc is not None:
                valid_msgs.append(valid_doc)
        else:
            valid_msgs.append(msg)

    # Return original if nothing changed (preserve identity for caching)
    if len(valid_msgs) == len(messages) and not warnings:
        return messages, []

    return AIMessages(valid_msgs), warnings
