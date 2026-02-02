"""Lightweight attachment model for multi-part documents."""

import base64
from functools import cached_property

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from ai_pipeline_core.exceptions import DocumentNameError

from .mime_type import (
    detect_mime_type,
    is_image_mime_type,
    is_pdf_mime_type,
    is_text_mime_type,
)


class Attachment(BaseModel):
    """Immutable binary attachment for multi-part documents.

    Carries binary content (screenshots, PDFs, supplementary files) without full Document machinery.
    ``mime_type`` is a cached_property — not included in ``model_dump()`` output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    content: bytes
    description: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Reject path traversal, reserved suffixes, whitespace issues."""
        if v.endswith(".description.md"):
            raise DocumentNameError(f"Attachment names cannot end with .description.md: {v}")
        if v.endswith(".sources.json"):
            raise DocumentNameError(f"Attachment names cannot end with .sources.json: {v}")
        if v.endswith(".attachments.json"):
            raise DocumentNameError(f"Attachment names cannot end with .attachments.json: {v}")
        if ".." in v or "\\" in v or "/" in v:
            raise DocumentNameError(f"Invalid attachment name - contains path traversal characters: {v}")
        if not v or v.startswith(" ") or v.endswith(" "):
            raise DocumentNameError(f"Invalid attachment name format: {v}")
        return v

    @field_serializer("content")
    def serialize_content(self, v: bytes) -> str:  # noqa: PLR6301
        """UTF-8 decode for text, base64 for binary."""
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(v).decode("ascii")

    @cached_property
    def mime_type(self) -> str:
        """Detected MIME type from content and filename. Cached."""
        return detect_mime_type(self.content, self.name)

    @property
    def is_image(self) -> bool:
        """True if MIME type starts with image/."""
        return is_image_mime_type(self.mime_type)

    @property
    def is_pdf(self) -> bool:
        """True if MIME type is application/pdf."""
        return is_pdf_mime_type(self.mime_type)

    @property
    def is_text(self) -> bool:
        """True if MIME type indicates text content."""
        return is_text_mime_type(self.mime_type)

    @property
    def size(self) -> int:
        """Content size in bytes."""
        return len(self.content)

    @property
    def text(self) -> str:
        """Content decoded as UTF-8. Raises ValueError if not text."""
        if not self.is_text:
            raise ValueError(f"Attachment is not text: {self.name}")
        return self.content.decode("utf-8")
