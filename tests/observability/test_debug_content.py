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
