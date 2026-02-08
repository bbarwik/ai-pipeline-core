"""Tests for document utilities."""

from ai_pipeline_core.documents.utils import camel_to_snake, sanitize_url


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
        assert sanitize_url("https://api.example.com/v1/resource?id=123&type=pdf") == "api.example.com_v1_resource"
        # FTP URLs aren't handled by the http/https check, @ is preserved
        assert sanitize_url("ftp://user:pass@host.com/file.txt") == "ftp_user_pass@host.com_file.txt"

    def test_query_strings(self):
        """Test handling of query strings without protocol."""
        # Query strings without protocol preserve ? and &
        assert sanitize_url("search?q=test&page=1") == "search_q=test&page=1"
        assert sanitize_url("file.php?download=true") == "file.php_download=true"


class TestCamelToSnake:
    """Test camel case to snake case conversion."""

    def test_basic_camel_case(self):
        """Test basic CamelCase conversion."""
        assert camel_to_snake("CamelCase") == "camel_case"
        assert camel_to_snake("SimpleTest") == "simple_test"
        assert camel_to_snake("MyClassName") == "my_class_name"

    def test_acronyms(self):
        """Test conversion with acronyms."""
        assert camel_to_snake("HTTPResponse") == "http_response"
        assert camel_to_snake("XMLParser") == "xml_parser"
        assert camel_to_snake("JSONData") == "json_data"
        assert camel_to_snake("PDFDocument") == "pdf_document"

    def test_mixed_cases(self):
        """Test mixed case patterns."""
        assert camel_to_snake("HTTPSConnection") == "https_connection"
        assert camel_to_snake("getHTTPResponseCode") == "get_http_response_code"
        assert camel_to_snake("IOError") == "io_error"

    def test_single_word(self):
        """Test single word inputs."""
        assert camel_to_snake("Document") == "document"
        assert camel_to_snake("Test") == "test"
        assert camel_to_snake("A") == "a"

    def test_already_snake_case(self):
        """Test already snake_case inputs."""
        assert camel_to_snake("already_snake_case") == "already_snake_case"
        assert camel_to_snake("lower_case") == "lower_case"

    def test_edge_cases(self):
        """Test edge cases."""
        assert camel_to_snake("") == ""
        assert camel_to_snake("ABC") == "abc"
        assert camel_to_snake("aBC") == "a_bc"
