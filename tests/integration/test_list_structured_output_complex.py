"""Complex integration tests for list[T] structured output.

Real LLM calls only — no mocking. Covers degeneration false positives,
large outputs, special characters, follow-up conversations, wrapper key
collisions, and multi-model validation.

Requires API keys configured in .env file.
"""

import json
from enum import StrEnum

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core._llm_core._degeneration import detect_output_degeneration
from ai_pipeline_core.documents import Document
from ai_pipeline_core.llm import Conversation
from ai_pipeline_core.llm._list_wrappers import _derive_list_field_name, get_or_create_wrapper
from ai_pipeline_core.prompt_compiler.components import Role
from ai_pipeline_core.prompt_compiler.spec import PromptSpec
from ai_pipeline_core.settings import settings
from tests.integration.model_categories import CORE_MODELS

pytestmark = pytest.mark.integration

HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)


# ── Enums ──────────────────────────────────────────────────────────────


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingCategory(StrEnum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    USABILITY = "usability"


# ── Complex nested models ──────────────────────────────────────────────


class AffectedComponent(BaseModel):
    name: str = Field(description="Component or module name")
    version: str = Field(description="Version string or 'unknown'")


class Recommendation(BaseModel):
    action: str = Field(description="What to do")
    effort_hours: int = Field(description="Estimated hours to fix")
    priority: int = Field(description="1=highest, 5=lowest")


class AuditFinding(BaseModel):
    """Deliberately complex schema to stress-test structured output."""

    title: str = Field(description="Short finding title")
    description: str = Field(description="Detailed description of the finding")
    severity: Severity = Field(description="Severity level")
    category: FindingCategory = Field(description="Finding category")
    affected_components: list[AffectedComponent] = Field(description="Components affected")
    recommendations: list[Recommendation] = Field(description="Recommended actions")
    cvss_score: float = Field(description="CVSS score 0.0-10.0")
    is_exploitable: bool = Field(description="Whether the vulnerability is currently exploitable")


class InventoryItem(BaseModel):
    """Item with optional fields to test partial population."""

    name: str = Field(description="Item name")
    sku: str = Field(description="Stock keeping unit")
    price_usd: float = Field(description="Price in USD")
    description: str | None = Field(default=None, description="Optional description")
    tags: list[str] = Field(default_factory=list, description="Optional tags")
    in_stock: bool = Field(default=True, description="Whether in stock")


class SimpleTag(BaseModel):
    label: str = Field(description="Tag label")
    confidence: float = Field(description="Confidence score 0.0-1.0")


# ── Models for wrapper key collision test ──────────────────────────────


class TextSnippet(BaseModel):
    source: str = Field(description="Source identifier")
    text: str = Field(description="The text content")


class LabelItem(BaseModel):
    label: str = Field(description="A text label")


class ColorItem(BaseModel):
    name: str = Field(description="Color name")
    hex_code: str = Field(description="Hex color code like #FF0000")


class ShapeItem(BaseModel):
    name: str = Field(description="Shape name")
    sides: int = Field(description="Number of sides, 0 for circle")


class TaskGroupModel(BaseModel):
    group_id: str
    objective: str


class TaskGroupSchema(BaseModel):
    group_id: str
    description: str


# ── PromptSpec definitions ─────────────────────────────────────────────


class AuditorRole(Role):
    """Security auditor role."""

    text = "senior security auditor with expertise in vulnerability assessment"


class DataEntryRole(Role):
    """Data entry role."""

    text = "meticulous data entry clerk"


class AuditFindingsSpec(PromptSpec[list[AuditFinding]]):
    """Extract security audit findings from a report."""

    input_documents = ()
    role = AuditorRole
    task = (
        "Analyze the provided text and extract all security findings. "
        "For each finding, fill in every field completely including nested components and recommendations. "
        "Use realistic CVSS scores and effort estimates."
    )


class InventorySpec(PromptSpec[list[InventoryItem]]):
    """Extract inventory items."""

    input_documents = ()
    role = DataEntryRole
    task = "Extract all inventory items from the provided text. Fill in all fields including optional ones when information is available."


class ReportDocument(Document):
    """A report document for audit findings."""


# ── Test 1: Degeneration false positive on complex nested output ──────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestDegenerationFalsePositive:
    """Verify that large list structured output does NOT trigger the degeneration detector.

    Production bug: pretty-printed JSON from deeply nested schemas causes repeated
    whitespace patterns which the degeneration detector flags as output degeneration.
    The wrapper model adds an extra nesting level, making indentation deeper.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_complex_list_output_not_flagged_as_degeneration(self, model: str) -> None:
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Generate exactly 5 security audit findings for a web application. "
            "Finding 1: SQL injection in login (CRITICAL, security, CVSS 9.8, exploitable). "
            "Finding 2: Missing rate limiting on API (HIGH, security, CVSS 7.5, exploitable). "
            "Finding 3: Slow database queries on search page (MEDIUM, performance, CVSS 4.0, not exploitable). "
            "Finding 4: No CSRF tokens on forms (HIGH, security, CVSS 8.1, exploitable). "
            "Finding 5: Missing alt text on images (LOW, usability, CVSS 0.0, not exploitable). "
            "Each finding must have at least 1 affected_component and 1 recommendation.",
            response_format=list[AuditFinding],
        )

        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 5
        assert all(isinstance(f, AuditFinding) for f in conv.parsed)

        parsed_content = json.loads(conv.content)
        assert isinstance(parsed_content, list)

        # CRITICAL: raw content must NOT trigger degeneration detection
        degeneration = detect_output_degeneration(conv.content)
        assert degeneration is None, (
            f"Degeneration detector false positive on list structured output: {degeneration}. Content length: {len(conv.content)} chars."
        )


# ── Test 2: Large list output stress test ─────────────────────────────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestLargeListOutput:
    """Stress tests for large list outputs near token limits."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_ten_item_rich_list(self, model: str) -> None:
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Generate exactly 10 inventory items for a hardware store. "
            "Item names: Hammer, Screwdriver, Wrench, Pliers, Drill, Saw, Level, "
            "Tape Measure, Sandpaper, Paint Brush. "
            "Each item must have: a realistic SKU (like 'HW-HAM-001'), "
            "a price between $5 and $200, a description (1-2 sentences), "
            "at least 2 tags, and in_stock set realistically.",
            response_format=list[InventoryItem],
        )

        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 10, f"Expected 10 items, got {len(conv.parsed)}"
        assert all(isinstance(item, InventoryItem) for item in conv.parsed)

        for i, item in enumerate(conv.parsed):
            assert item.name, f"Item {i} has empty name"
            assert item.sku, f"Item {i} has empty SKU"
            assert item.price_usd > 0, f"Item {i} has zero/negative price"

        parsed_content = json.loads(conv.content)
        assert len(parsed_content) == 10


# ── Test 3: Special characters in fields ──────────────────────────────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestSpecialCharacters:
    """Test list output with special characters in string fields."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_unicode_in_fields(self, model: str) -> None:
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Return exactly 3 text snippets with non-ASCII characters:\n"
            "1. source='french', text should contain accented characters like é, ü, or ñ\n"
            "2. source='symbols', text should contain currency symbols and arrows like €, $, →\n"
            "3. source='cjk', text should contain Chinese or Japanese characters",
            response_format=list[TextSnippet],
        )

        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 3
        assert all(isinstance(s, TextSnippet) for s in conv.parsed)

        # Verify JSON round-trip preserved non-ASCII content
        parsed_content = json.loads(conv.content)
        assert isinstance(parsed_content, list)
        assert len(parsed_content) == 3
        # At least one snippet should contain non-ASCII characters
        all_text = " ".join(s.text for s in conv.parsed)
        has_non_ascii = any(ord(c) > 127 for c in all_text)
        assert has_non_ascii, f"Expected non-ASCII characters in output, got: {all_text[:200]}"

    @pytest.mark.asyncio
    async def test_duplicate_items_not_deduped(self) -> None:
        conv = Conversation(model="gemini-3-flash")
        conv = await conv.send_structured(
            "Return exactly 4 label items. The first two must BOTH have label='duplicate'. The third has label='unique_a'. The fourth has label='unique_b'.",
            response_format=list[LabelItem],
        )

        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 4
        labels = [item.label for item in conv.parsed]
        assert labels.count("duplicate") == 2, (
            f"Expected 2 items with label='duplicate', got {labels.count('duplicate')}. Labels: {labels}. Items may have been silently deduplicated."
        )


# ── Test 4: Follow-up conversation after list output ──────────────────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestFollowUpConversation:
    """Test multi-turn conversations involving list structured output."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_follow_up_after_list_output(self, model: str) -> None:
        conv = Conversation(model=model)

        conv = await conv.send_structured(
            "Return exactly 3 tags: label='alpha' confidence=0.9, label='beta' confidence=0.7, label='gamma' confidence=0.5.",
            response_format=list[SimpleTag],
        )
        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 3

        conv = await conv.send("How many items did you just return? Reply with only the number.")
        assert "3" in conv.content, f"Follow-up should reference prior list output. Got: {conv.content[:200]}"

    @pytest.mark.asyncio
    async def test_two_consecutive_list_outputs(self) -> None:
        conv = Conversation(model="gemini-3-flash")

        conv = await conv.send_structured(
            "Return exactly 2 colors: Red (#FF0000) and Blue (#0000FF).",
            response_format=list[ColorItem],
        )
        assert len(conv.parsed) == 2
        assert all(isinstance(c, ColorItem) for c in conv.parsed)

        conv = await conv.send_structured(
            "Now return exactly 2 shapes: Triangle (3 sides) and Circle (0 sides).",
            response_format=list[ShapeItem],
        )
        assert len(conv.parsed) == 2
        assert all(isinstance(s, ShapeItem) for s in conv.parsed)
        assert any(s.sides == 3 for s in conv.parsed)


# ── Test 5: send_spec with list output + documents in context ─────────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestSendSpecWithDocuments:
    """Test PromptSpec[list[T]] through the full send_spec path with documents."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_send_spec_list_with_document_context(self, model: str) -> None:
        report_text = (
            "Security Audit Report - 2026-Q1\n\n"
            "Finding 1: SQL Injection in /api/login endpoint.\n"
            "Severity: CRITICAL. CVSS: 9.8. Exploitable: Yes.\n"
            "Affected: auth-service v2.1.0\n"
            "Recommendation: Use parameterized queries. Effort: 8 hours.\n\n"
            "Finding 2: Missing rate limiting on /api/search.\n"
            "Severity: HIGH. CVSS: 7.5. Exploitable: Yes.\n"
            "Affected: api-gateway v1.4.2\n"
            "Recommendation: Add rate limiting middleware. Effort: 4 hours."
        )
        report_doc = ReportDocument.create_root(
            name="audit-report.md",
            content=report_text,
            reason="integration test",
        )

        conv = Conversation(model=model)
        spec = AuditFindingsSpec()
        conv = await conv.send_spec(spec, documents=[report_doc])

        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) >= 2, f"Expected at least 2 findings, got {len(conv.parsed)}"
        assert all(isinstance(f, AuditFinding) for f in conv.parsed)

        for finding in conv.parsed:
            assert finding.title
            assert finding.severity in list(Severity)
            assert finding.category in list(FindingCategory)
            assert len(finding.affected_components) >= 1
            assert len(finding.recommendations) >= 1


# ── Test 6: Wrapper key collision ─────────────────────────────────────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestWrapperKeyCollision:
    """Test that models with similar names producing the same wrapper field name work independently."""

    @pytest.mark.asyncio
    async def test_same_derived_key_different_types(self) -> None:
        assert _derive_list_field_name(TaskGroupModel) == _derive_list_field_name(TaskGroupSchema)

        wrapper_a = get_or_create_wrapper(TaskGroupModel)
        wrapper_b = get_or_create_wrapper(TaskGroupSchema)
        assert wrapper_a is not wrapper_b

        conv = Conversation(model="gemini-3-flash")

        conv_a = await conv.send_structured(
            "Return 2 task groups: group_id='A' objective='Research', group_id='B' objective='Analysis'.",
            response_format=list[TaskGroupModel],
        )
        assert isinstance(conv_a.parsed, list)
        assert len(conv_a.parsed) == 2
        assert all(isinstance(g, TaskGroupModel) for g in conv_a.parsed)

        conv_b = await conv.send_structured(
            "Return 2 task groups: group_id='X' description='Design phase', group_id='Y' description='Build phase'.",
            response_format=list[TaskGroupSchema],
        )
        assert isinstance(conv_b.parsed, list)
        assert len(conv_b.parsed) == 2
        assert all(isinstance(g, TaskGroupSchema) for g in conv_b.parsed)

        assert hasattr(conv_a.parsed[0], "objective")
        assert hasattr(conv_b.parsed[0], "description")


# ── Test 7: Enum and boolean consistency ──────────────────────────────


@pytest.mark.skipif(not HAS_API_KEYS, reason="API keys not configured")
class TestEnumAndBooleanFields:
    """Test that enum and boolean fields in list items are correctly typed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", CORE_MODELS)
    async def test_enum_fields_in_list_items(self, model: str) -> None:
        conv = Conversation(model=model)
        conv = await conv.send_structured(
            "Return exactly 3 audit findings:\n"
            "1. title='XSS in search', severity=high, category=security, CVSS=8.0, exploitable=true, "
            "component: search-ui v3.0, recommendation: sanitize input (4h, priority 1)\n"
            "2. title='Slow queries', severity=medium, category=performance, CVSS=4.0, exploitable=false, "
            "component: db-service v2.1, recommendation: add indexes (8h, priority 2)\n"
            "3. title='Poor contrast', severity=low, category=usability, CVSS=0.0, exploitable=false, "
            "component: frontend v1.5, recommendation: update CSS (2h, priority 3)",
            response_format=list[AuditFinding],
        )

        assert isinstance(conv.parsed, list)
        assert len(conv.parsed) == 3

        for finding in conv.parsed:
            assert isinstance(finding.severity, Severity), f"severity: got {type(finding.severity).__name__}: {finding.severity!r}"
            assert isinstance(finding.category, FindingCategory), f"category: got {type(finding.category).__name__}: {finding.category!r}"
            assert isinstance(finding.is_exploitable, bool), f"is_exploitable: got {type(finding.is_exploitable).__name__}"
            assert 0.0 <= finding.cvss_score <= 10.0, f"CVSS {finding.cvss_score} out of range"


# ── Test 8: Degeneration detector regression (no API keys needed) ─────


class TestDegenerationDetectorWithStructuredJSON:
    """Direct tests for the degeneration detector against structured JSON patterns.

    No API keys needed — tests the detector against synthetic but realistic JSON.
    """

    def test_deeply_nested_wrapper_json_not_flagged(self) -> None:
        """Deeply indented wrapper JSON must not trigger degeneration."""
        items = [
            {
                "title": f"Finding {i}: {'x' * 40}",
                "description": f"Description for finding {i} with enough text to be realistic " * 2,
                "severity": "high",
                "category": "security",
                "affected_components": [
                    {"name": f"component-{i}-a", "version": "1.0.0"},
                    {"name": f"component-{i}-b", "version": "2.3.1"},
                ],
                "recommendations": [
                    {"action": f"Fix issue {i}", "effort_hours": i + 1, "priority": 1},
                ],
                "cvss_score": 7.5,
                "is_exploitable": True,
            }
            for i in range(10)
        ]

        wrapper_json = json.dumps({"audit_findings": items}, indent=2)
        result = detect_output_degeneration(wrapper_json)
        assert result is None, f"Degeneration false positive on wrapper JSON ({len(wrapper_json)} chars): {result}"

    def test_array_json_not_flagged(self) -> None:
        """Unwrapped array JSON must not trigger degeneration."""
        items = [
            {
                "name": f"Item {i}",
                "sku": f"SKU-{i:04d}",
                "price_usd": 10.0 + i,
                "description": f"A {'very ' * 10}long description for item {i}",
                "tags": ["tag-a", "tag-b", "tag-c"],
                "in_stock": True,
            }
            for i in range(10)
        ]
        array_json = json.dumps(items, indent=2)
        result = detect_output_degeneration(array_json)
        assert result is None, f"Degeneration false positive on array JSON ({len(array_json)} chars): {result}"

    def test_actual_degeneration_still_detected(self) -> None:
        """Genuine degeneration (token loop) must still be caught."""
        degenerate = '{"findings": [' + '{"x": "a"}, ' * 500 + '{"x": "a"}]}'
        result = detect_output_degeneration(degenerate)
        assert result is not None, "Genuine degeneration should still be detected"
