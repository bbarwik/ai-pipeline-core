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

# Tool base class — execute(input) is an interface method, overridden by subclasses
from ai_pipeline_core.llm.tools import Tool

Tool.execute
input  # parameter of Tool.execute base method — used by subclass implementations

# OTel SpanProcessor overrides — parameter signature required by base class
from ai_pipeline_core.observability._tracking._processor import PipelineSpanProcessor

PipelineSpanProcessor.force_flush

# Prefect/deployment hooks
# Add more as vulture reports false positives
