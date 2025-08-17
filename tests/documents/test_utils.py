"""Tests for document utilities."""

from ai_pipeline_core.documents.utils import sanitize_url


class TestSanitizeUrl:
    """Test URL sanitization for filenames."""

    def test_url_with_protocol(self):
        """Test sanitization of URLs with protocol."""
        assert sanitize_url("https://example.com/path") == "example.com_path"
        assert sanitize_url("http://test.org/file.pdf") == "test.org_file.pdf"

    def test_invalid_filename_characters(self):
        """Test removal of invalid filename characters."""
        assert sanitize_url('file<>:"/\\|?*name') == "file_name"
        assert sanitize_url("path/to/file") == "path_to_file"
        assert sanitize_url("file:name") == "file_name"

    def test_multiple_underscores(self):
        """Test collapsing of multiple underscores."""
        assert sanitize_url("file___name") == "file_name"
        assert sanitize_url("path//to//file") == "path_to_file"

    def test_trim_edges(self):
        """Test trimming of leading/trailing underscores and dots."""
        assert sanitize_url("_file_") == "file"
        assert sanitize_url(".file.") == "file"
        assert sanitize_url("__file__") == "file"

    def test_length_limit(self):
        """Test that long names are truncated to 100 characters."""
        long_name = "a" * 150
        result = sanitize_url(long_name)
        assert len(result) == 100
        assert result == "a" * 100

    def test_empty_fallback(self):
        """Test fallback to 'unnamed' for empty results."""
        assert sanitize_url("") == "unnamed"
        assert sanitize_url("...") == "unnamed"
        assert sanitize_url("___") == "unnamed"
        assert sanitize_url("//:") == "unnamed"

    def test_complex_urls(self):
        """Test complex URL patterns."""
        # URLs with query strings - query part is ignored when parsing
        assert (
            sanitize_url("https://api.example.com/v1/resource?id=123&type=pdf")
            == "api.example.com_v1_resource"
        )
        # FTP URLs aren't handled by the http/https check, @ is preserved
        assert (
            sanitize_url("ftp://user:pass@host.com/file.txt") == "ftp_user_pass@host.com_file.txt"
        )

    def test_query_strings(self):
        """Test handling of query strings without protocol."""
        # Query strings without protocol preserve ? and &
        assert sanitize_url("search?q=test&page=1") == "search_q=test&page=1"
        assert sanitize_url("file.php?download=true") == "file.php_download=true"
