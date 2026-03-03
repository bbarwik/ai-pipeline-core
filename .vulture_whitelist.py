"""Vulture whitelist — methods called by frameworks, not direct code."""

# Pydantic validators/serializers — called by Pydantic, not our code
from ai_pipeline_core.documents.document import Document

Document._validate_content
Document._validate_name
Document._validate_total_size
Document._serialize_content

from ai_pipeline_core.documents.attachment import Attachment

Attachment._validate_name
Attachment._serialize_content

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
