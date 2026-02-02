"""Vulture whitelist — methods called by frameworks, not direct code."""

# Pydantic validators/serializers — called by Pydantic, not our code
from ai_pipeline_core.documents.document import Document

Document.validate_content
Document.validate_name
Document.validate_total_size
Document.serialize_content

from ai_pipeline_core.documents.attachment import Attachment

Attachment.validate_name
Attachment.serialize_content

# __init_subclass__ — called by Python
Document.__init_subclass__

# Prefect/deployment hooks
# Add more as vulture reports false positives
