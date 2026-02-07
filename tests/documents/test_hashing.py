"""Tests for document hashing utilities used by store implementations."""

from ai_pipeline_core.documents import Attachment, Document
from ai_pipeline_core.documents._hashing import compute_content_sha256, compute_document_sha256


class HashDoc(Document):
    pass


class TestComputeDocumentSha256:
    def test_deterministic(self):
        doc = HashDoc.create(name="a.txt", content="hello")
        h1 = compute_document_sha256(doc)
        h2 = compute_document_sha256(doc)
        assert h1 == h2

    def test_base32_format(self):
        doc = HashDoc.create(name="a.txt", content="hello")
        h = compute_document_sha256(doc)
        assert h.isascii()
        assert h == h.upper()
        assert len(h) == 52  # SHA256 in base32 without padding

    def test_different_name_different_hash(self):
        doc1 = HashDoc.create(name="a.txt", content="hello")
        doc2 = HashDoc.create(name="b.txt", content="hello")
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)

    def test_different_content_different_hash(self):
        doc1 = HashDoc.create(name="a.txt", content="hello")
        doc2 = HashDoc.create(name="a.txt", content="world")
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)

    def test_attachments_affect_hash(self):
        doc1 = HashDoc.create(name="a.txt", content="hello")
        att = Attachment(name="img.png", content=b"\x89PNG")
        doc2 = HashDoc.create(name="a.txt", content="hello", attachments=(att,))
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)

    def test_attachment_order_does_not_matter(self):
        """Attachments are sorted by name before hashing."""
        att_a = Attachment(name="a.txt", content=b"aaa")
        att_b = Attachment(name="b.txt", content=b"bbb")
        doc1 = HashDoc.create(name="doc.txt", content="content", attachments=(att_a, att_b))
        doc2 = HashDoc.create(name="doc.txt", content="content", attachments=(att_b, att_a))
        assert compute_document_sha256(doc1) == compute_document_sha256(doc2)

    def test_description_does_not_affect_hash(self):
        """Description is excluded from document_sha256."""
        doc1 = HashDoc.create(name="a.txt", content="hello", description="desc1")
        doc2 = HashDoc.create(name="a.txt", content="hello", description="desc2")
        assert compute_document_sha256(doc1) == compute_document_sha256(doc2)

    def test_sources_affect_hash(self):
        """Sources are included in document_sha256."""
        doc1 = HashDoc.create(name="a.txt", content="hello", sources=("https://example.com/src1",))
        doc2 = HashDoc.create(name="a.txt", content="hello", sources=("https://example.com/src2",))
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)

    def test_origins_affect_hash(self):
        """Origins are included in document_sha256."""
        origin_a = compute_document_sha256(HashDoc.create(name="origin_a.txt", content="origin a"))
        origin_b = compute_document_sha256(HashDoc.create(name="origin_b.txt", content="origin b"))
        doc1 = HashDoc.create(name="a.txt", content="hello", origins=(origin_a,))
        doc2 = HashDoc.create(name="a.txt", content="hello", origins=(origin_b,))
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)

    def test_source_order_does_not_matter(self):
        """Sources are sorted before hashing for order-independence."""
        doc1 = HashDoc.create(name="a.txt", content="hello", sources=("https://a.com", "https://b.com"))
        doc2 = HashDoc.create(name="a.txt", content="hello", sources=("https://b.com", "https://a.com"))
        assert compute_document_sha256(doc1) == compute_document_sha256(doc2)

    def test_origin_order_does_not_matter(self):
        """Origins are sorted before hashing for order-independence."""
        origin_a = compute_document_sha256(HashDoc.create(name="origin_a.txt", content="origin a"))
        origin_b = compute_document_sha256(HashDoc.create(name="origin_b.txt", content="origin b"))
        doc1 = HashDoc.create(name="a.txt", content="hello", origins=(origin_a, origin_b))
        doc2 = HashDoc.create(name="a.txt", content="hello", origins=(origin_b, origin_a))
        assert compute_document_sha256(doc1) == compute_document_sha256(doc2)

    def test_no_provenance_still_deterministic(self):
        """Documents without sources/origins produce consistent hashes."""
        doc1 = HashDoc.create(name="a.txt", content="hello")
        doc2 = HashDoc.create(name="a.txt", content="hello")
        assert compute_document_sha256(doc1) == compute_document_sha256(doc2)

    def test_group_count_disambiguation(self):
        """Moving items between sources and origins produces different hashes.

        Validates count-prefix boundaries: sources=['A'], origins=['B']
        must differ from sources=['A','B'], origins=[].
        """
        url_a = "https://example.com/a"
        url_b = "https://example.com/b"
        doc1 = HashDoc.create(name="a.txt", content="hello", sources=(url_a,), origins=())
        doc2 = HashDoc.create(name="a.txt", content="hello", sources=(), origins=())
        doc3 = HashDoc.create(name="a.txt", content="hello", sources=(url_a, url_b))
        doc4 = HashDoc.create(name="a.txt", content="hello", sources=(url_a,))
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)
        assert compute_document_sha256(doc3) != compute_document_sha256(doc4)

    def test_length_prefix_prevents_collision(self):
        """Length prefixing prevents 'ab' + 'cd' from colliding with 'abc' + 'd'."""
        # Different name/content splits should produce different hashes
        doc1 = HashDoc.create(name="ab", content="cd")
        doc2 = HashDoc.create(name="abc", content="d")
        assert compute_document_sha256(doc1) != compute_document_sha256(doc2)


class TestComputeContentSha256:
    def test_deterministic(self):
        h1 = compute_content_sha256(b"hello")
        h2 = compute_content_sha256(b"hello")
        assert h1 == h2

    def test_base32_format(self):
        h = compute_content_sha256(b"hello")
        assert h.isascii()
        assert h == h.upper()
        assert len(h) == 52

    def test_different_content(self):
        assert compute_content_sha256(b"hello") != compute_content_sha256(b"world")

    def test_empty_content(self):
        h = compute_content_sha256(b"")
        assert len(h) == 52
