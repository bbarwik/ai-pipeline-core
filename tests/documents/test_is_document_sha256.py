"""Tests for is_document_sha256 function."""

import hashlib
from base64 import b32encode

from ai_pipeline_core.documents import Document, is_document_sha256


class TestIsDocumentSha256:
    """Test the is_document_sha256 function."""

    def test_real_document_hash(self):
        """Test with a real document SHA256 hash."""

        # Create a real document and get its hash
        class SampleDoc(Document):
            pass

        doc = SampleDoc.create(name="test.txt", content="test content")
        assert is_document_sha256(doc.sha256)

    def test_various_real_hashes(self):
        """Test with various real SHA256 hashes."""
        # Generate real hashes from different content
        test_contents = [
            b"Hello, World!",
            b"The quick brown fox jumps over the lazy dog",
            b"",  # Empty content
            b"1234567890",
            b"\x00\x01\x02\x03\x04",  # Binary content
        ]

        for content in test_contents:
            sha256_hash = b32encode(hashlib.sha256(content).digest()).decode("ascii").upper().rstrip("=")
            assert is_document_sha256(sha256_hash), f"Failed for content: {content!r}"

    def test_insufficient_entropy(self):
        """Test that low-entropy strings are rejected."""
        # All same character - definitely not a real hash
        assert not is_document_sha256("A" * 52)
        assert not is_document_sha256("2" * 52)
        assert not is_document_sha256("7" * 52)

        # Only 2 unique characters
        assert not is_document_sha256("ABABABABABABABABABABABABABABABABABABABABABABABABABA")

        # Only 3 unique characters
        assert not is_document_sha256("ABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCABCA")

        # 7 unique characters (just below threshold)
        test_str = "ABCDEFG" * 7 + "ABC"  # 52 chars with 7 unique
        assert len(test_str) == 52
        assert len(set(test_str)) == 7
        assert not is_document_sha256(test_str)

    def test_sufficient_entropy(self):
        """Test that strings with sufficient entropy are accepted."""
        # Exactly 8 unique characters (minimum threshold)
        test_str = "ABCDEFGH" * 6 + "ABCD"  # 52 chars with 8 unique
        assert len(test_str) == 52
        assert len(set(test_str)) == 8
        assert is_document_sha256(test_str)

        # More than 8 unique characters
        test_str = "ABCDEFGHIJ23456" * 3 + "ABCDEFG"  # 52 chars with 15 unique
        assert len(test_str) == 52
        assert is_document_sha256(test_str)

    def test_wrong_length(self):
        """Test that strings with wrong length are rejected."""
        # Too short
        assert not is_document_sha256("ABC")
        assert not is_document_sha256("A" * 51)

        # Too long
        assert not is_document_sha256("A" * 53)
        # 55 chars
        assert not is_document_sha256("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567ABCDEFGHIJKLMNOPQRSTUV")

        # With padding (old format)
        assert not is_document_sha256("A" * 52 + "====")

    def test_invalid_characters(self):
        """Test that strings with invalid base32 characters are rejected."""
        # Lowercase letters (base32 is uppercase only)
        assert not is_document_sha256("a" * 52)
        assert not is_document_sha256("AbCdEfGhIjKlMnOpQrStUvWxYz234567AbCdEfGhIjKlMnOpQrSt")

        # Invalid digits (0, 1, 8, 9 are not in base32)
        assert not is_document_sha256("0" * 52)
        assert not is_document_sha256("1" * 52)
        assert not is_document_sha256("8" * 52)
        assert not is_document_sha256("9" * 52)

        # Mixed invalid characters
        test_str = "ABCDEFGH" + "0189" + "ABCDEFGH" * 4 + "ABCDEFGH"  # Contains 0,1,8,9
        assert len(test_str) == 52
        assert not is_document_sha256(test_str)

        # Special characters
        assert not is_document_sha256("A" * 25 + "=" + "A" * 26)
        assert not is_document_sha256("A" * 25 + "-" + "A" * 26)
        assert not is_document_sha256("A" * 25 + "_" + "A" * 26)

    def test_valid_base32_characters(self):
        """Test that all valid base32 characters are accepted."""
        # Use all valid base32 characters (A-Z, 2-7)
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
        test_str = (valid_chars * 2)[:52]  # Take first 52 chars
        assert len(test_str) == 52
        assert is_document_sha256(test_str)

    def test_edge_cases(self):
        """Test edge cases and special inputs."""
        # Non-string inputs
        assert not is_document_sha256(None)  # type: ignore[arg-type]
        assert not is_document_sha256(123)  # type: ignore[arg-type]
        assert not is_document_sha256([])  # type: ignore[arg-type]
        assert not is_document_sha256(b"A" * 52)  # type: ignore[arg-type]

        # Empty string
        assert not is_document_sha256("")

        # Whitespace
        assert not is_document_sha256(" " * 52)
        assert not is_document_sha256("A" * 51 + " ")

    def test_known_patterns(self):
        """Test with known hash patterns."""
        # These are actual SHA256 hashes from the codebase tests
        known_hashes = [
            "P3AEMA2PSYILKFYVBUALJLMIYWVZIS2QDI3S5VTMD2X7SOODF2YQ",  # From verify_sources.py
            "DSITTXMIGUJ5CHKJEVTW3IOQFYJ3LHOXZFWZBN7FH7AR3DGWTAXA",  # Example from earlier
        ]

        for hash_val in known_hashes:
            assert is_document_sha256(hash_val), f"Known hash rejected: {hash_val}"

    def test_integration_with_document_sources(self):
        """Test that it works correctly with document sources field."""

        # Create a document and use its hash
        class SampleDoc(Document):
            pass

        doc = SampleDoc.create(name="test.txt", content="integration test")

        # The hash should be valid
        assert is_document_sha256(doc.sha256)

        # Create another document with the first as source
        doc2 = SampleDoc.create(
            name="derived.txt",
            content="derived from first",
            sources=(doc.sha256, "https://example.com/manual-reference"),
        )

        # Check that the source is properly categorized
        doc_sources = doc2.source_documents
        ref_sources = doc2.source_references

        assert len(doc_sources) == 1
        assert doc_sources[0] == doc.sha256
        assert len(ref_sources) == 1
        assert ref_sources[0] == "https://example.com/manual-reference"

    def test_performance_characteristics(self):
        """Test that the function is efficient for typical use."""
        import time

        # Generate a real hash
        real_hash = b32encode(hashlib.sha256(b"performance test").digest()).decode("ascii").upper().rstrip("=")

        # Test should be very fast (sub-millisecond)
        start = time.perf_counter()
        for _ in range(1000):
            is_document_sha256(real_hash)
        elapsed = time.perf_counter() - start

        # Should process 1000 hashes in well under a second
        assert elapsed < 0.1, f"Performance issue: 1000 checks took {elapsed:.3f}s"
