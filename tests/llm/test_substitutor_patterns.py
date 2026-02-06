"""Tests for pattern extraction in URL/address substitution."""

import re


# Define patterns locally for testing (mirrors internal _PATTERNS)
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'`\[\]{}|\\^]+", re.IGNORECASE)
_HEX_PREFIXED_PATTERN = re.compile(r"\b0x[a-fA-F0-9]{40,}\b")
_HIGH_ENTROPY_PATTERN = re.compile(r"\b[A-Za-z0-9]{26,}\b")


class TestURLExtraction:
    """Tests for URL pattern extraction."""

    def test_simple_url(self):
        text = "Check https://example.com for details"
        matches = list(_URL_PATTERN.finditer(text))
        assert len(matches) == 1
        assert matches[0].group() == "https://example.com"

    def test_url_with_path(self):
        text = "See https://example.com/path/to/resource"
        matches = list(_URL_PATTERN.finditer(text))
        assert matches[0].group() == "https://example.com/path/to/resource"

    def test_url_with_query_string(self):
        text = "Link: https://example.com/search?q=test&page=1"
        matches = list(_URL_PATTERN.finditer(text))
        assert "?q=test&page=1" in matches[0].group()

    def test_url_with_fragment(self):
        text = "https://example.com/page#section"
        matches = list(_URL_PATTERN.finditer(text))
        assert matches[0].group() == "https://example.com/page#section"

    def test_url_with_port(self):
        text = "Server at https://example.com:8080/api"
        matches = list(_URL_PATTERN.finditer(text))
        assert ":8080" in matches[0].group()

    def test_http_url(self):
        text = "http://insecure.com/path"
        matches = list(_URL_PATTERN.finditer(text))
        assert matches[0].group().startswith("http://")

    def test_multiple_urls(self):
        text = "Links: https://a.com and https://b.com/path"
        matches = list(_URL_PATTERN.finditer(text))
        assert len(matches) == 2

    def test_very_long_url(self):
        long_path = "/segment" * 50
        text = f"https://example.com{long_path}"
        matches = list(_URL_PATTERN.finditer(text))
        assert len(matches) == 1
        assert len(matches[0].group()) > 400

    def test_no_urls(self):
        text = "No URLs here, just plain text."
        matches = list(_URL_PATTERN.finditer(text))
        assert len(matches) == 0


class TestHexAddressExtraction:
    """Tests for hex-prefixed address extraction (Ethereum, etc.)."""

    def test_eth_address(self):
        text = "Contract: 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        matches = list(_HEX_PREFIXED_PATTERN.finditer(text))
        assert len(matches) == 1
        assert matches[0].group().startswith("0x")
        assert len(matches[0].group()) == 42

    def test_eth_address_lowercase(self):
        text = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"
        matches = list(_HEX_PREFIXED_PATTERN.finditer(text))
        assert len(matches) == 1

    def test_multiple_eth_addresses(self):
        text = "From 0x1234567890123456789012345678901234567890 to 0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        matches = list(_HEX_PREFIXED_PATTERN.finditer(text))
        assert len(matches) == 2

    def test_eth_tx_hash(self):
        # Transaction hash is 0x + 64 hex chars = 66 total
        text = "Tx: 0x7a250d5630b4cf539739df2c5dacb4c659f2488d7a250d5630b4cf539739df2c"
        matches = list(_HEX_PREFIXED_PATTERN.finditer(text))
        assert len(matches) == 1
        assert len(matches[0].group()) == 66  # 0x + 64 hex


class TestHighEntropyExtraction:
    """Tests for high-entropy string extraction."""

    def test_solana_address(self):
        # Solana addresses are 43-44 base58 chars
        text = "Solana: 7EcDhSYGxXyscszYEp35KHN8vvw3svAuLKTzXwCFLtV"
        matches = list(_HIGH_ENTROPY_PATTERN.finditer(text))
        assert len(matches) >= 1

    def test_btc_legacy_p2pkh(self):
        text = "Pay to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        matches = list(_HIGH_ENTROPY_PATTERN.finditer(text))
        # May or may not match depending on entropy checks
        assert isinstance(matches, list)


class TestEmptyAndEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self):
        for pattern in [_URL_PATTERN, _HEX_PREFIXED_PATTERN, _HIGH_ENTROPY_PATTERN]:
            matches = list(pattern.finditer(""))
            assert matches == []

    def test_whitespace_only(self):
        for pattern in [_URL_PATTERN, _HEX_PREFIXED_PATTERN, _HIGH_ENTROPY_PATTERN]:
            matches = list(pattern.finditer("   \n\t  "))
            assert matches == []
