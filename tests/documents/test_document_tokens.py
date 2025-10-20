"""Tests for Document approximate_tokens_count property."""

from tests.test_helpers import ConcreteFlowDocument


class TestDocumentApproximateTokensCount:
    """Test Document approximate_tokens_count property."""

    def test_text_document_tokens(self):
        """Test token count for text document."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"Hello world")
        count = doc.approximate_tokens_count
        assert count > 0
        assert isinstance(count, int)
        assert count < 10  # Should be very few tokens

    def test_long_text_document(self):
        """Test token count for long text document."""
        long_text = b"This is a much longer document " * 100
        doc = ConcreteFlowDocument(name="long.txt", content=long_text)
        count = doc.approximate_tokens_count
        assert count > 100  # Should have many tokens

    def test_empty_text_document(self):
        """Test token count for empty text document."""
        doc = ConcreteFlowDocument(name="empty.txt", content=b"")
        count = doc.approximate_tokens_count
        # Empty content is detected as text/plain, so returns 0 tokens
        assert count == 0

    def test_unicode_text_document(self):
        """Test token count with unicode content."""
        doc = ConcreteFlowDocument(name="unicode.txt", content="Hello 世界 🌍".encode())
        count = doc.approximate_tokens_count
        assert count > 0
        assert isinstance(count, int)

    def test_non_text_document_fixed_estimate(self):
        """Test that non-text documents return fixed 1024 estimate."""
        # Binary PNG header
        doc = ConcreteFlowDocument(name="image.png", content=b"\x89PNG\r\n\x1a\n")
        count = doc.approximate_tokens_count
        assert count == 1024  # Fixed estimate

    def test_binary_document_fixed_estimate(self):
        """Test that binary documents return fixed 1024 estimate."""
        doc = ConcreteFlowDocument(name="data.bin", content=b"\x00\x01\x02\x03\x04")
        count = doc.approximate_tokens_count
        assert count == 1024  # Fixed estimate

    def test_consistency(self):
        """Test that token count is consistent for same document."""
        doc = ConcreteFlowDocument(name="test.txt", content=b"Consistent content")
        count1 = doc.approximate_tokens_count
        count2 = doc.approximate_tokens_count
        count3 = doc.approximate_tokens_count
        assert count1 == count2 == count3

    def test_json_document_tokens(self):
        """Test token count for JSON document."""
        json_content = b'{"name": "test", "value": 123, "nested": {"key": "value"}}'
        doc = ConcreteFlowDocument(name="data.json", content=json_content)
        count = doc.approximate_tokens_count
        assert count > 0

    def test_code_document_tokens(self):
        """Test token count for code document."""
        code = b"""
def hello_world():
    print("Hello, World!")
    return True
"""
        doc = ConcreteFlowDocument(name="code.py", content=code)
        count = doc.approximate_tokens_count
        assert count > 0

    def test_multiline_document(self):
        """Test token count for multiline document."""
        multiline = b"""Line 1
Line 2
Line 3
Line 4
"""
        doc = ConcreteFlowDocument(name="lines.txt", content=multiline)
        count = doc.approximate_tokens_count
        assert count > 0

    def test_document_with_description_tokens(self):
        """Test token count only counts content, not description."""
        doc1 = ConcreteFlowDocument(
            name="test.txt",
            content=b"Content",
        )
        doc2 = ConcreteFlowDocument(
            name="test.txt",
            content=b"Content",
            description="This is a description",
        )
        # Both should have same token count (description not included)
        assert doc1.approximate_tokens_count == doc2.approximate_tokens_count

    def test_special_characters_tokens(self):
        """Test token count with special characters."""
        doc = ConcreteFlowDocument(name="special.txt", content=b"Hello! @#$% ^&*() <>?")
        count = doc.approximate_tokens_count
        assert count > 0

    def test_numbers_tokens(self):
        """Test token count with numbers."""
        doc = ConcreteFlowDocument(name="numbers.txt", content=b"The year 2024 has 365 days")
        count = doc.approximate_tokens_count
        assert count > 0

    def test_very_large_document(self):
        """Test token count for very large document."""
        large_content = b"Word " * 10000
        doc = ConcreteFlowDocument(name="large.txt", content=large_content)
        count = doc.approximate_tokens_count
        assert count > 5000  # Should have many tokens

    def test_markdown_document(self):
        """Test token count for markdown document."""
        markdown = b"""# Header

This is **bold** and this is *italic*.

- List item 1
- List item 2
"""
        doc = ConcreteFlowDocument(name="doc.md", content=markdown)
        count = doc.approximate_tokens_count
        assert count > 0
