"""Prove structured JSON output can trigger false degeneration detection.

detect_output_degeneration does not distinguish between degenerate repetition
and JSON structural whitespace. The call site in client.py does not skip
detection for structured output responses.
"""

import inspect
import json


from ai_pipeline_core._llm_core._degeneration import (
    detect_output_degeneration,
)


# ── Proving tests: PASS on current code, demonstrate the bug ─────────


class TestWhitespaceTriggersDegeneration:
    """Prove that whitespace patterns trigger false positive detection."""

    def test_consecutive_spaces_detected_as_degeneration(self) -> None:
        """300+ consecutive spaces trigger degeneration detection."""
        content = "Some prefix text\n" + " " * 350 + "\nSome suffix text"
        result = detect_output_degeneration(content)
        assert result is not None
        assert "repeated" in result

    def test_consecutive_newlines_detected(self) -> None:
        """300+ consecutive newlines trigger degeneration detection."""
        content = "prefix" + "\n" * 350 + "suffix"
        result = detect_output_degeneration(content)
        assert result is not None

    def test_json_with_trailing_whitespace_padding(self) -> None:
        """JSON with trailing whitespace (Gemini artifact) triggers false positive."""
        fields = {f"field_{i:03d}": f"value_{i}" for i in range(80)}
        json_text = json.dumps(fields, indent=4)
        content = json_text + "\n" + " " * 400
        result = detect_output_degeneration(content)
        assert result is not None

    def test_indented_json_with_repeated_structure(self) -> None:
        """Deeply indented JSON with repeated patterns triggers false positive."""
        # 4 spaces per indent, many levels of nesting produce long whitespace runs
        items = [{"id": i, "value": ""} for i in range(50)]
        content = json.dumps({"data": {"nested": {"items": items}}}, indent=8)
        # Add trailing padding that Gemini sometimes produces
        content += " " * 500
        result = detect_output_degeneration(content)
        assert result is not None


class TestStructuredOutputSkipsDegeneration:
    """Verify the call site skips degeneration for structured output."""

    def test_call_site_has_response_format_guard(self) -> None:
        """_generate_impl skips degeneration check when response_format is set."""
        from ai_pipeline_core._llm_core.client import _generate_impl

        source = inspect.getsource(_generate_impl)

        assert "detect_output_degeneration" in source
        assert "has_tool_calls" in source

        # Find the degeneration check line and verify response_format guard is present
        lines = source.split("\n")
        degen_line_indices = [i for i, line in enumerate(lines) if "detect_output_degeneration" in line]
        assert len(degen_line_indices) >= 1

        degen_idx = degen_line_indices[0]
        degen_line = lines[degen_idx]
        assert "response_format" in degen_line


# ── Regression guards: real degeneration must still be detected ──────


class TestRealDegenerationStillDetected:
    """Ensure actual degeneration is caught (regression guard)."""

    def test_word_repetition_loop(self) -> None:
        """Actual token loop is correctly detected."""
        content = "The answer is " + "bau " * 120 + "end."
        result = detect_output_degeneration(content)
        assert result is not None

    def test_normal_text_not_flagged(self) -> None:
        """Normal varied text is not flagged."""
        paragraphs = [f"Paragraph {i} discusses topic {chr(65 + i % 26)} with unique observations." for i in range(30)]
        content = "\n\n".join(paragraphs)
        result = detect_output_degeneration(content)
        assert result is None
