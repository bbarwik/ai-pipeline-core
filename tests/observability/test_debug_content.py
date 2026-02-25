"""Tests for ContentWriter."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml

from ai_pipeline_core.observability._debug import ContentWriter, TraceDebugConfig
from ai_pipeline_core.observability._trimming import _CONTENT_TRIM_THRESHOLD


@pytest.fixture
def config(tmp_path: Path) -> TraceDebugConfig:
    """Create test configuration."""
    return TraceDebugConfig(path=tmp_path)


@pytest.fixture
def writer(config: TraceDebugConfig) -> ContentWriter:
    """Create test ContentWriter."""
    return ContentWriter(config)


class TestContentWriter:
    """Tests for ContentWriter."""

    def test_write_none_returns_none_ref(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test writing None returns none type reference."""
        ref = writer.write(None, tmp_path, "test")
        assert ref["type"] == "none"

    def test_write_small_text_to_file(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test small text content is written to file (for persistence)."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        ref = writer.write("Hello", tmp_path, "test")
        assert ref["type"] == "file"
        assert ref["path"] == "test.yaml"
        assert (tmp_path / "test.yaml").exists()
        content = (tmp_path / "test.yaml").read_text()
        assert "Hello" in content

    def test_write_large_text_to_file(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test large text content is written to file."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        large_text = "x" * 200  # Larger than max_element_bytes (100)
        ref = writer.write(large_text, tmp_path, "test")

        assert ref["type"] == "file"
        assert ref["path"] == "test.yaml"
        assert (tmp_path / "test.yaml").exists()

    def test_write_dict_serialized_to_yaml(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test dict is serialized to YAML file."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"key": "value", "number": 42}
        ref = writer.write(data, tmp_path, "test")

        # All content is written to files for persistence
        assert ref["type"] == "file"
        content = (tmp_path / "test.yaml").read_text()
        assert "key" in content
        assert "value" in content

    def test_write_large_dict_to_file(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test large dict is written to file."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"key": "x" * 200}
        ref = writer.write(data, tmp_path, "test")

        assert ref["type"] == "file"
        content = (tmp_path / "test.yaml").read_text()
        assert "key" in content

    def test_redaction_openai_key(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test OpenAI API key is redacted."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"api_key": "sk-1234567890abcdefghijklmnop"}
        writer.write(data, tmp_path, "test")

        content = (tmp_path / "test.yaml").read_text()
        assert "sk-1234567890" not in content
        assert "[REDACTED]" in content

    def test_redaction_password(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test password is redacted."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"config": "password: mysecretpassword"}
        writer.write(data, tmp_path, "test")

        content = (tmp_path / "test.yaml").read_text()
        assert "mysecretpassword" not in content
        assert "[REDACTED]" in content

    def test_convert_uuid(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test UUID is converted to string."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        test_uuid = UUID("12345678-1234-5678-1234-567812345678")
        data = {"id": test_uuid}
        writer.write(data, tmp_path, "test")

        content = (tmp_path / "test.yaml").read_text()
        assert "12345678-1234-5678-1234-567812345678" in content

    def test_convert_datetime(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test datetime is converted to ISO format."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        dt = datetime(2026, 1, 28, 13, 45, 0)
        data = {"timestamp": dt}
        writer.write(data, tmp_path, "test")

        content = (tmp_path / "test.yaml").read_text()
        assert "2026-01-28" in content

    def test_convert_path(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test Path is converted to string."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"file": Path("/tmp/test.txt")}
        writer.write(data, tmp_path, "test")

        content = (tmp_path / "test.yaml").read_text()
        assert "/tmp/test.txt" in content

    def test_convert_set(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test set is converted to sorted list."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"tags": {"c", "a", "b"}}
        writer.write(data, tmp_path, "test")

        content = (tmp_path / "test.yaml").read_text()
        # Should be sorted
        assert "- a" in content
        assert "- b" in content
        assert "- c" in content

    def test_circular_reference_handled(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test circular references don't cause infinite loop."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"self": None}
        data["self"] = data  # Circular reference

        ref = writer.write(data, tmp_path, "test")

        # Should complete without hanging
        assert ref["type"] == "file"
        content = (tmp_path / "test.yaml").read_text()
        assert "[circular reference]" in content


class TestLLMMessageWriting:
    """Tests for LLM message content extraction."""

    def test_write_simple_messages(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test writing simple LLM messages."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        ref = writer.write(messages, tmp_path, "input")

        assert ref["type"] == "file"
        assert (tmp_path / "input.yaml").exists()

        # Check content
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        assert content["type"] == "llm_messages"
        assert content["message_count"] == 2
        assert content["messages"][0]["role"] == "system"
        assert "helpful assistant" in content["messages"][0]["parts"][0]["content"]

    def test_write_multimodal_messages_with_image(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test extracting base64 images from multimodal messages."""
        # Create a small test image (1x1 red PNG)
        png_data = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\n\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_data}"},
                    },
                ],
            }
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        ref = writer.write(messages, tmp_path, "input")

        assert ref["type"] == "file"
        assert (tmp_path / "input.yaml").exists()

        # Check content structure
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        assert content["type"] == "llm_messages"
        assert content["message_count"] == 1
        assert len(content["messages"][0]["parts"]) == 2

        # Check text part
        text_part = content["messages"][0]["parts"][0]
        assert text_part["type"] == "text"
        assert "What's in this image?" in text_part["content"]

        # Check image part (should be inline for small image since no artifact store)
        image_part = content["messages"][0]["parts"][1]
        assert image_part["type"] == "image"
        assert image_part["format"] == "png"


class TestDocumentListWriting:
    """Tests for writing lists of documents."""

    def test_write_document_list(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Test writing a list of documents."""
        docs = [
            {
                "class_name": "SampleFlowDocument",
                "name": "report.txt",
                "content": "This is the report content.",
                "content_encoding": "utf-8",
            },
            {
                "class_name": "SampleTaskDocument",
                "name": "data.json",
                "content": '{"key": "value"}',
                "content_encoding": "utf-8",
            },
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        ref = writer.write(docs, tmp_path, "docs")

        assert ref["type"] == "file"
        assert (tmp_path / "docs.yaml").exists()

        # Check content structure
        content = yaml.safe_load((tmp_path / "docs.yaml").read_text())
        assert content["type"] == "document_list"
        assert content["document_count"] == 2
        assert len(content["documents"]) == 2

        # Check document entries
        assert content["documents"][0]["name"] == "report.txt"
        assert "This is the report content." in content["documents"][0]["content"]


class TestContentTruncation:
    """Tests for content truncation."""

    def test_truncate_very_large_content(self, tmp_path: Path) -> None:
        """Test very large content is truncated at file level."""
        config = TraceDebugConfig(
            path=tmp_path,
            max_file_bytes=500,  # Small file limit for testing
        )
        writer = ContentWriter(config)

        tmp_path.mkdir(parents=True, exist_ok=True)
        # Create large content that will exceed max_file_bytes when serialized to YAML
        large_content = {"data": "x" * 10000}
        ref = writer.write(large_content, tmp_path, "test")

        assert ref["type"] == "file"
        file_content = (tmp_path / "test.yaml").read_text()
        # Should be truncated at file level
        file_size = len(file_content.encode("utf-8"))
        assert file_size <= 600  # Allow margin for truncation message
        assert "[TRUNCATED:" in file_content or file_size < 10000


class TestDocumentXmlTrimming:
    """Bug: trimming cuts through XML document tags instead of only trimming <content>."""

    def test_llm_message_trims_only_document_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        """When an LLM message contains <document> XML, only the <content> inner text should be trimmed.

        The XML metadata (<id>, <name>, <description>) must be preserved intact.
        """
        document_xml = (
            "<document>\n"
            "<id>ABC123</id>\n"
            "<name>research_task.md</name>\n"
            "<description>Research objective</description>\n"
            "<content>\n" + "x" * (_CONTENT_TRIM_THRESHOLD * 2) + "\n</content>\n"
            "</document>"
        )
        messages = [{"role": "user", "content": document_xml}]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        text = content["messages"][0]["parts"][0]["content"]

        # XML metadata must be fully preserved
        assert "<id>ABC123</id>" in text
        assert "<name>research_task.md</name>" in text
        assert "<description>Research objective</description>" in text
        assert "</document>" in text
        # Only the inner content of <content> should be trimmed
        assert "trimmed" in text

    def test_llm_message_trims_only_content_in_multi_document(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Multiple <document> blocks: each should have only its <content> trimmed."""
        long_content = "y" * (_CONTENT_TRIM_THRESHOLD * 2)
        text_with_docs = (
            "<document>\n<id>DOC1</id>\n<name>first.md</name>\n"
            f"<content>\n{long_content}\n</content>\n</document>\n\n"
            "<document>\n<id>DOC2</id>\n<name>second.md</name>\n"
            f"<content>\n{long_content}\n</content>\n</document>"
        )
        messages = [{"role": "user", "content": text_with_docs}]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        text = content["messages"][0]["parts"][0]["content"]

        assert "<id>DOC1</id>" in text
        assert "<id>DOC2</id>" in text
        assert "<name>first.md</name>" in text
        assert "<name>second.md</name>" in text

    def test_llm_message_short_document_not_trimmed(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Documents with short <content> should not be trimmed."""
        document_xml = "<document>\n<id>SHORT</id>\n<name>small.md</name>\n<content>\nShort text\n</content>\n</document>"
        messages = [{"role": "user", "content": document_xml}]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        text = content["messages"][0]["parts"][0]["content"]

        assert "Short text" in text
        assert "trimmed" not in text


class TestPlainTextNotTrimmed:
    """Bug: plain text LLM messages (task instructions, prompts) are trimmed like document content."""

    def test_plain_text_message_not_trimmed(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Plain text user messages must NOT be trimmed — only document <content> should be.

        Task instructions like '# Task\nAnalyze the...' are critical for debugging
        and should be preserved in full in the trace.
        """
        task_instruction = (
            "# Task\n\n"
            "Now write the Analysis Context for the research definition.\n\n"
            "The Analysis Context is the ONLY background information the research agent receives. "
            "It must contain everything needed to understand the subject, the landscape, and the "
            "known challenges: what is hard to find, commonly misunderstood, restricted, or "
            "unreliable in this area.\n\n"
            "Include specific details about the entities, relationships, and terminology that "
            "the agent needs to know before starting research."
        )
        assert len(task_instruction) > _CONTENT_TRIM_THRESHOLD  # Confirm it exceeds threshold

        messages = [{"role": "user", "content": task_instruction}]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]

        # Full text must be preserved — no trimming
        assert part["content"] == task_instruction
        assert "truncated" not in part or part.get("truncated") is not True
        assert "trimmed" not in part["content"]

    def test_plain_text_multipart_message_not_trimmed(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Plain text parts in multipart messages must NOT be trimmed."""
        long_text = "Analyze the following data carefully. " * 20  # ~740 chars
        assert len(long_text) > _CONTENT_TRIM_THRESHOLD

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": long_text},
                ],
            }
        ]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]

        assert part["content"] == long_text
        assert "trimmed" not in part["content"]

    def test_document_xml_still_trimmed(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Document XML content should still be trimmed (only <content> inner text)."""
        document_xml = "<document>\n<id>ABC123</id>\n<name>report.md</name>\n<content>\n" + "x" * (_CONTENT_TRIM_THRESHOLD * 3) + "\n</content>\n</document>"
        messages = [{"role": "user", "content": document_xml}]

        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "input")

        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]

        # Document content should be trimmed
        assert "trimmed" in part["content"]
        # But XML metadata preserved
        assert "<id>ABC123</id>" in part["content"]


class TestYamlMultilineFormatting:
    """Bug: strings with trailing spaces use escaped \\n instead of YAML block scalar."""

    def test_generic_multiline_string_uses_block_style(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Multiline generic content must use YAML block scalar (|), not escaped \\n."""
        multiline = "Line 1\nLine 2\nLine 3"
        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(multiline, tmp_path, "test")

        raw_yaml = (tmp_path / "test.yaml").read_text()
        # Must NOT contain literal \n escape sequences
        assert "\\n" not in raw_yaml
        # Must use block scalar indicator
        assert "|-" in raw_yaml or "| " in raw_yaml or "|\n" in raw_yaml

    def test_multiline_string_with_trailing_spaces_uses_block_style(self, writer: ContentWriter, tmp_path: Path) -> None:
        """Strings with trailing spaces on lines must still use block scalar, not escaped \\n."""
        text_with_trailing = "Line with trailing spaces  \nAnother line  \nFinal line"
        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(text_with_trailing, tmp_path, "test")

        raw_yaml = (tmp_path / "test.yaml").read_text()
        # Must NOT contain literal \n escape sequences
        assert "\\n" not in raw_yaml

    def test_llm_message_with_trailing_spaces_uses_block_style(self, writer: ContentWriter, tmp_path: Path) -> None:
        """LLM message text with trailing spaces must use block scalar formatting."""
        messages = [{"role": "assistant", "content": "Result:  \n- Item 1  \n- Item 2"}]
        tmp_path.mkdir(parents=True, exist_ok=True)
        writer.write(messages, tmp_path, "output")

        raw_yaml = (tmp_path / "output.yaml").read_text()
        assert "\\n" not in raw_yaml


# ---------------------------------------------------------------------------
# Coverage gap tests: _structure_message_part branches, _convert_types branches,
# _is_llm_messages/_is_document_list edge cases, _structure_image branches
# ---------------------------------------------------------------------------


class TestIsLlmMessagesEdgeCases:
    def test_empty_list(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        ref = writer.write([], tmp_path, "test")
        assert ref["type"] == "file"
        content = yaml.safe_load((tmp_path / "test.yaml").read_text())
        assert content["type"] == "generic"

    def test_not_all_dicts(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = [{"role": "user", "content": "hi"}, "not-a-dict"]
        writer.write(data, tmp_path, "test")
        content = yaml.safe_load((tmp_path / "test.yaml").read_text())
        assert content["type"] == "generic"


class TestMessagePartBranches:
    def test_tool_use_part(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"query": "test"}},
                ],
                "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "search"}}],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "tool_use"
        assert part["name"] == "search"
        assert "tool_calls" in content["messages"][0]

    def test_tool_result_string_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "is_error": False, "content": "result text"},
                ],
                "tool_call_id": "t1",
                "name": "search",
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "tool_result"
        assert part["tool_use_id"] == "t1"
        assert "tool_call_id" in content["messages"][0]
        assert "name" in content["messages"][0]

    def test_tool_result_list_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": [{"type": "text", "text": "inner"}],
                    },
                ],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "tool_result"
        assert isinstance(part["content"], list)

    def test_tool_result_other_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t3", "content": 42},
                ],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "tool_result"

    def test_unknown_part_type(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [{"type": "custom_thing", "data": "abc"}],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "unknown"
        assert part["original_type"] == "custom_thing"

    def test_content_none(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "assistant", "content": None}]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        assert content["messages"][0]["parts"] == []

    def test_content_unknown_type(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [{"role": "user", "content": 12345}]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "unknown"

    def test_function_call_in_msg(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "assistant",
                "content": "call func",
                "function_call": {"name": "get_weather", "arguments": "{}"},
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        assert "function_call" in content["messages"][0]


class TestStructureImageBranches:
    def test_image_url_not_data_uri(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "low"}},
                ],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "image_url"
        assert part["url"] == "https://example.com/img.png"

    def test_image_url_bad_data_uri(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/broken"}},
                ],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "image_parse_error"

    def test_anthropic_image_type(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": "AQID"},
                    },
                ],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "image"
        assert part["format"] == "jpeg"

    def test_anthropic_image_non_base64(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "media_type": "image/png"},
                    },
                ],
            }
        ]
        writer.write(messages, tmp_path, "input")
        content = yaml.safe_load((tmp_path / "input.yaml").read_text())
        part = content["messages"][0]["parts"][0]
        assert part["type"] == "image"
        assert part["source_type"] == "url"


class TestStructureDocumentsBranches:
    def test_binary_document_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        docs = [{"class_name": "BinaryDoc", "name": "img.png", "content": "data:image/png;base64,AAAA"}]
        writer.write(docs, tmp_path, "docs")
        content = yaml.safe_load((tmp_path / "docs.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["content"] == "[binary content removed]"

    def test_non_string_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        docs = [{"class_name": "NumberDoc", "name": "val.txt", "content": 42}]
        writer.write(docs, tmp_path, "docs")
        content = yaml.safe_load((tmp_path / "docs.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["content"] == "42"

    def test_document_with_attachments(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        docs = [
            {
                "class_name": "AttDoc",
                "name": "main.txt",
                "content": "main content",
                "attachments": [
                    {"name": "att.txt", "content": "att content", "description": "an attachment"},
                    {"name": "bin.png", "content": "data:image/png;base64,AAAA"},
                ],
            }
        ]
        writer.write(docs, tmp_path, "docs")
        content = yaml.safe_load((tmp_path / "docs.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 2
        assert doc_entry["attachments"][0]["name"] == "att.txt"
        assert doc_entry["attachments"][0]["description"] == "an attachment"
        assert doc_entry["attachments"][1]["content"] == "[binary content removed]"

    def test_non_dict_attachments_skipped(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        docs = [
            {
                "class_name": "Doc",
                "name": "main.txt",
                "content": "content",
                "attachments": ["not-a-dict", {"name": "real.txt", "content": "ok"}],
            }
        ]
        writer.write(docs, tmp_path, "docs")
        content = yaml.safe_load((tmp_path / "docs.yaml").read_text())
        doc_entry = content["documents"][0]
        assert doc_entry["attachment_count"] == 1


class TestConvertTypesBranches:
    def test_secret_str(self, tmp_path: Path) -> None:
        from pydantic import SecretStr

        no_redact_config = TraceDebugConfig(path=tmp_path, redact_patterns=())
        w = ContentWriter(no_redact_config)
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"secret": SecretStr("value123")}
        w.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "[REDACTED:SecretStr]" in content

    def test_bytes_short(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"data": b"\x01\x02\x03"}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "bytes: 3 bytes" in content
        assert "preview" in content

    def test_bytes_long(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"data": b"\x00" * 200}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "bytes: 200 bytes" in content
        assert "preview" not in content

    def test_enum_value(self, writer: ContentWriter, tmp_path: Path) -> None:
        from enum import Enum

        class Color(Enum):
            RED = "red"

        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"color": Color.RED}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "red" in content

    def test_pydantic_model(self, writer: ContentWriter, tmp_path: Path) -> None:
        from pydantic import BaseModel

        class Item(BaseModel):
            val: int = 10

        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"item": Item()}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "10" in content

    def test_frozenset(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"fs": frozenset({"b", "a"})}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "- a" in content
        assert "- b" in content

    def test_tuple_content(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        data = {"items": (1, 2, 3)}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "1" in content
        assert "3" in content

    def test_unknown_type_str_fallback(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)

        class Custom:
            def __str__(self) -> str:
                return "custom_repr"

        data = {"obj": Custom()}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "custom_repr" in content

    def test_unknown_type_str_fails(self, writer: ContentWriter, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)

        class BadStr:
            def __str__(self) -> str:
                raise RuntimeError("no str")

        data = {"obj": BadStr()}
        writer.write(data, tmp_path, "test")
        content = (tmp_path / "test.yaml").read_text()
        assert "<BadStr>" in content
