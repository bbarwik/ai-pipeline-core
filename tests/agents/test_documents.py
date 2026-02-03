"""Tests for ai_pipeline_core.agents.documents module."""

import pytest

from ai_pipeline_core.agents import AgentResult
from ai_pipeline_core.agents.documents import AgentOutputDocument, _sanitize_name

# Valid BASE32-encoded SHA256 hashes for testing (52 chars, A-Z2-7, high entropy)
VALID_SHA256_1 = "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ"
VALID_SHA256_2 = "Q4BFNB3QTZJMLGZWCVBMKMNKJZWAK3RREK4T6WUNE3Y7TPPEG3RQ"


class TestFromResultSuccess:
    """Tests for AgentOutputDocument.from_result() with successful results."""

    def test_extracts_primary_artifact_content(self, sample_result: AgentResult):
        """Primary artifact should become document content."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            artifact_name="output.md",
            origins=(VALID_SHA256_1,),
        )
        assert doc.text == "# Report\n\nThis is the main output."

    def test_fallback_to_any_md_artifact(self):
        """Should fall back to any .md artifact if primary not found."""
        result = AgentResult(
            success=True,
            artifacts={
                "report.md": b"# Fallback content",
                "data.json": b"{}",
            },
        )
        doc = AgentOutputDocument.from_result(
            result,
            artifact_name="nonexistent.md",
            origins=(VALID_SHA256_1,),
        )
        assert "Fallback content" in doc.text

    def test_no_artifact_shows_helpful_message(self):
        """Should show helpful message when no artifacts available."""
        result = AgentResult(
            success=True,
            artifacts={"data.json": b"{}"},
        )
        doc = AgentOutputDocument.from_result(
            result,
            artifact_name="output.md",
            origins=(VALID_SHA256_1,),
        )
        assert "no output artifact found" in doc.text.lower()
        assert "data.json" in doc.text

    def test_empty_artifacts_shows_message(self):
        """Should handle empty artifacts dict."""
        result = AgentResult(success=True, artifacts={})
        doc = AgentOutputDocument.from_result(
            result,
            origins=(VALID_SHA256_1,),
        )
        assert "no output artifact found" in doc.text.lower()


class TestFromResultFailure:
    """Tests for AgentOutputDocument.from_result() with failed results."""

    def test_includes_error_message(self, failed_result: AgentResult):
        """Failed result should include error message."""
        doc = AgentOutputDocument.from_result(
            failed_result,
            origins=(VALID_SHA256_1,),
        )
        assert "Connection timeout" in doc.text

    def test_includes_traceback(self, failed_result: AgentResult):
        """Failed result should include traceback if available."""
        doc = AgentOutputDocument.from_result(
            failed_result,
            origins=(VALID_SHA256_1,),
        )
        assert "Traceback" in doc.text

    def test_includes_stderr(self, failed_result: AgentResult):
        """Failed result should include stderr if available."""
        doc = AgentOutputDocument.from_result(
            failed_result,
            origins=(VALID_SHA256_1,),
        )
        assert "ERROR: Failed to connect" in doc.text

    def test_includes_exit_code(self, failed_result: AgentResult):
        """Failed result should include exit code if available."""
        doc = AgentOutputDocument.from_result(
            failed_result,
            origins=(VALID_SHA256_1,),
        )
        assert "Exit code: 1" in doc.text

    def test_handles_minimal_failure(self):
        """Should handle failure with only error message."""
        result = AgentResult(
            success=False,
            error="Something broke",
        )
        doc = AgentOutputDocument.from_result(
            result,
            origins=(VALID_SHA256_1,),
        )
        assert "Something broke" in doc.text

    def test_handles_failure_without_error(self):
        """Should handle failure without error message."""
        result = AgentResult(success=False)
        doc = AgentOutputDocument.from_result(
            result,
            origins=(VALID_SHA256_1,),
        )
        assert "Unknown error" in doc.text


class TestProvenance:
    """Tests for provenance (origins) handling."""

    def test_requires_origins(self, sample_result: AgentResult):
        """Should raise ValueError if origins is empty."""
        with pytest.raises(ValueError, match="origins is required"):
            AgentOutputDocument.from_result(sample_result, origins=())

    def test_error_message_is_helpful(self, sample_result: AgentResult):
        """Error message should guide user on how to fix."""
        try:
            AgentOutputDocument.from_result(sample_result, origins=())
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            msg = str(e)
            assert "SHA256" in msg
            assert "example" in msg.lower() or "input_doc" in msg

    def test_sets_origins_on_document(self, sample_result: AgentResult):
        """Origins should be set on the created document."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            origins=(VALID_SHA256_1, VALID_SHA256_2),
        )
        assert doc.origins == (VALID_SHA256_1, VALID_SHA256_2)

    def test_single_origin_works(self, sample_result: AgentResult):
        """Single origin should work."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            origins=(VALID_SHA256_1,),
        )
        assert len(doc.origins) == 1


class TestDocumentMetadata:
    """Tests for document name and description."""

    def test_default_name(self, sample_result: AgentResult):
        """Default document name should be 'agent_output.md'."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            origins=(VALID_SHA256_1,),
        )
        assert doc.name == "agent_output.md"

    def test_custom_name(self, sample_result: AgentResult):
        """Custom name should be respected."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            name="research_report.md",
            origins=(VALID_SHA256_1,),
        )
        assert doc.name == "research_report.md"

    def test_auto_description(self, sample_result: AgentResult):
        """Description should mention agent name."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            origins=(VALID_SHA256_1,),
        )
        assert doc.description is not None and "test_agent" in doc.description

    def test_custom_description(self, sample_result: AgentResult):
        """Custom description should be respected."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            description="My custom description",
            origins=(VALID_SHA256_1,),
        )
        assert doc.description == "My custom description"


class TestAttachments:
    """Tests for attachment handling."""

    def test_attachments_disabled_by_default(self, sample_result: AgentResult):
        """Attachments should not be included by default."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            origins=(VALID_SHA256_1,),
        )
        assert doc.attachments == ()

    def test_attachments_included_when_enabled(self, sample_result: AgentResult):
        """Attachments should be included when flag is True."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            artifact_name="output.md",
            origins=(VALID_SHA256_1,),
            include_artifacts_as_attachments=True,
        )
        assert doc.attachments is not None
        names = [a.name for a in doc.attachments]
        assert "data.json" in names

    def test_primary_artifact_not_in_attachments(self, sample_result: AgentResult):
        """Primary artifact should not appear in attachments."""
        doc = AgentOutputDocument.from_result(
            sample_result,
            artifact_name="output.md",
            origins=(VALID_SHA256_1,),
            include_artifacts_as_attachments=True,
        )
        names = [a.name for a in doc.attachments]
        assert "output.md" not in names

    def test_large_artifacts_skipped(self, caplog):
        """Artifacts larger than max_attachment_size should be skipped."""
        result = AgentResult(
            success=True,
            artifacts={
                "output.md": b"content",
                "huge.bin": b"x" * 1000,
            },
        )
        doc = AgentOutputDocument.from_result(
            result,
            origins=(VALID_SHA256_1,),
            include_artifacts_as_attachments=True,
            max_attachment_size=100,
        )
        names = [a.name for a in doc.attachments]
        assert "huge.bin" not in names
        assert "Skipping large artifact" in caplog.text

    def test_path_separators_sanitized(self):
        """Path separators in artifact names should be replaced."""
        result = AgentResult(
            success=True,
            artifacts={
                "output.md": b"content",
                "subdir/file.txt": b"nested",
                "other\\file.txt": b"backslash",
            },
        )
        doc = AgentOutputDocument.from_result(
            result,
            origins=(VALID_SHA256_1,),
            include_artifacts_as_attachments=True,
        )
        names = [a.name for a in doc.attachments]
        assert "subdir_file.txt" in names
        assert "other_file.txt" in names
        assert "subdir/file.txt" not in names

    def test_invalid_names_skipped(self, caplog):
        """Invalid attachment names should be skipped with warning."""
        result = AgentResult(
            success=True,
            artifacts={
                "output.md": b"content",
                "invalid<name>.txt": b"bad",
            },
        )
        doc = AgentOutputDocument.from_result(
            result,
            origins=(VALID_SHA256_1,),
            include_artifacts_as_attachments=True,
        )
        names = [a.name for a in doc.attachments]
        assert not any("<" in n for n in names)
        assert "invalid name" in caplog.text.lower()

    def test_empty_attachments_returns_none(self):
        """When only primary artifact exists, attachments should be None."""
        result = AgentResult(
            success=True,
            artifacts={"output.md": b"content"},
        )
        doc = AgentOutputDocument.from_result(
            result,
            artifact_name="output.md",
            origins=(VALID_SHA256_1,),
            include_artifacts_as_attachments=True,
        )
        # Only artifact is the primary one, so no attachments
        assert doc.attachments == ()


class TestSanitizeName:
    """Tests for _sanitize_name() helper function."""

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("file.txt", "file.txt"),
            ("sub/file.txt", "sub_file.txt"),
            ("sub\\file.txt", "sub_file.txt"),
            ("a/b/c.md", "a_b_c.md"),
            ("file-name_v2.tar.gz", "file-name_v2.tar.gz"),
            ("report.md", "report.md"),
            ("data_2024.json", "data_2024.json"),
        ],
    )
    def test_valid_names(self, input_name: str, expected: str):
        """Valid names should be sanitized correctly."""
        assert _sanitize_name(input_name) == expected

    @pytest.mark.parametrize(
        "input_name",
        [
            "file<name>.txt",
            "file>name.txt",
            "file:name.txt",
            "file|name.txt",
            "file?name.txt",
            "file*name.txt",
            'file"name.txt',
            "file\x00name.txt",
        ],
    )
    def test_invalid_names_return_none(self, input_name: str):
        """Invalid names should return None."""
        assert _sanitize_name(input_name) is None
