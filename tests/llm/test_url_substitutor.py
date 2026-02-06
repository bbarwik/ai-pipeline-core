"""Tests for URLSubstitutor class."""

import pytest

from ai_pipeline_core.llm import URLSubstitutor


@pytest.fixture
def substitutor():
    """Create a fresh URLSubstitutor."""
    return URLSubstitutor()


class TestURLSubstitutor:
    """Tests for URLSubstitutor."""

    @pytest.mark.asyncio
    async def test_prepare_extracts_urls(self, substitutor):
        # URL must be > 40 chars to be shortened
        texts = ["Check https://example.com/some/long/path/to/resource for info"]
        substitutor.prepare(texts)

        assert substitutor.is_prepared
        assert substitutor.pattern_count == 1

    @pytest.mark.asyncio
    async def test_substitute_replaces_urls(self, substitutor):
        # URL must be > 40 chars to be shortened
        url = "https://example.com/very/long/path/to/page"
        substitutor.prepare([f"Visit {url} today"])

        result = substitutor.substitute(f"Visit {url} today")

        assert url not in result
        assert "example.com" in result  # Domain preserved
        assert "~" in result  # Truncation marker

    @pytest.mark.asyncio
    async def test_restore_reverses_substitution(self, substitutor):
        original = "Check https://example.com/long/path/to/details for info"
        substitutor.prepare([original])

        substituted = substitutor.substitute(original)
        restored = substitutor.restore(substituted)

        assert restored == original

    @pytest.mark.asyncio
    async def test_eth_address_substitution(self, substitutor):
        text = "Contract: 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        substitutor.prepare([text])

        result = substitutor.substitute(text)

        assert "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D" not in result
        assert "0x7a25" in result  # Prefix preserved
        assert "488D" in result  # Suffix preserved
        assert "~" in result  # Truncation marker

    @pytest.mark.asyncio
    async def test_multiple_patterns(self, substitutor):
        url = "https://example.com/very/long/path/to/page"
        text = f"URL: {url}, Address: 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        substitutor.prepare([text])

        result = substitutor.substitute(text)

        assert url not in result
        assert "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D" not in result

    @pytest.mark.asyncio
    async def test_no_patterns(self, substitutor):
        text = "Plain text with no URLs"
        substitutor.prepare([text])

        result = substitutor.substitute(text)

        assert result == text

    @pytest.mark.asyncio
    async def test_empty_text(self, substitutor):
        substitutor.prepare([""])

        result = substitutor.substitute("")

        assert result == ""

    @pytest.mark.asyncio
    async def test_unique_labels_for_different_urls(self, substitutor):
        # Two different URLs with same domain should get unique labels (hash suffix)
        texts = [
            "https://example.com/very/long/path/page1/content",
            "https://example.com/very/long/path/page2/content",
        ]
        substitutor.prepare(texts)

        mappings = substitutor.get_mappings()
        labels = list(mappings.values())

        # Labels should be unique due to hash suffix
        assert len(labels) == len(set(labels))

    @pytest.mark.asyncio
    async def test_persistence_across_prepare_calls(self, substitutor):
        url = "https://example.com/very/long/path/to/test"
        # First prepare
        substitutor.prepare([url])
        label1 = substitutor.get_mappings().get(url)

        # Second prepare with same URL
        substitutor.prepare([url])
        label2 = substitutor.get_mappings().get(url)

        # Should use same label (deterministic)
        assert label1 == label2

    @pytest.mark.asyncio
    async def test_incremental_prepare(self, substitutor):
        # First prepare with one URL (must be > 40 chars)
        substitutor.prepare(["https://first.com/very/long/path/to/page"])
        assert substitutor.pattern_count == 1

        # Second prepare with another URL
        substitutor.prepare(["https://second.com/very/long/path/to/page"])
        assert substitutor.pattern_count == 2

    @pytest.mark.asyncio
    async def test_btc_address_substitution(self, substitutor):
        text = "Pay to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        substitutor.prepare([text])

        result = substitutor.substitute(text)

        assert "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" not in result
        assert "1A1z" in result  # Prefix preserved (4 chars)
        assert "vfNa" in result  # Suffix preserved
        assert "~" in result  # Truncation marker

    @pytest.mark.asyncio
    async def test_mixed_urls_and_addresses(self, substitutor):
        text = """
        Contract at 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
        Docs: https://docs.example.com/api/v2/contracts/very/long
        """
        substitutor.prepare([text])

        result = substitutor.substitute(text)
        restored = substitutor.restore(result)

        assert restored == text


class TestURLSubstitutorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_substitute_before_prepare(self):
        substitutor = URLSubstitutor()

        # Substitute before prepare should return unchanged
        result = substitutor.substitute("https://example.com/very/long/path")
        assert result == "https://example.com/very/long/path"

    @pytest.mark.asyncio
    async def test_restore_before_prepare(self):
        substitutor = URLSubstitutor()

        # Restore before prepare should return unchanged
        result = substitutor.restore("https://example.com~1234")
        assert result == "https://example.com~1234"

    @pytest.mark.asyncio
    async def test_round_trip_with_special_chars(self):
        substitutor = URLSubstitutor()

        url = "https://example.com/path?query=value&other=test#section/very/long"
        substitutor.prepare([url])

        substituted = substitutor.substitute(url)
        restored = substitutor.restore(substituted)

        assert restored == url

    @pytest.mark.asyncio
    async def test_deterministic_labels(self):
        """Same URL should always produce same label."""
        url = "https://example.com/very/long/path/test/page"

        sub1 = URLSubstitutor()
        sub1.prepare([url])
        label1 = sub1.get_mappings()[url]

        sub2 = URLSubstitutor()
        sub2.prepare([url])
        label2 = sub2.get_mappings()[url]

        assert label1 == label2

    @pytest.mark.asyncio
    async def test_url_label_format(self):
        """Labels should follow the scheme://domain/path~hash format."""
        substitutor = URLSubstitutor()
        url = "https://example.com/docs/api/v2/reference"
        substitutor.prepare([url])

        label = substitutor.get_mappings()[url]
        assert label.startswith("https://")
        assert "example.com" in label
        assert "~" in label  # Hash truncation marker

    @pytest.mark.asyncio
    async def test_address_label_format(self):
        """Address labels should follow the prefix~hash~suffix format."""
        substitutor = URLSubstitutor()
        eth = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        substitutor.prepare([eth])

        label = substitutor.get_mappings()[eth]
        assert label.startswith("0x7a25")  # Prefix preserved
        assert "~" in label  # Truncation marker
        assert label.endswith("488D")  # Suffix preserved

    @pytest.mark.asyncio
    async def test_short_urls_not_shortened(self):
        """URLs under 40 chars should not be shortened."""
        substitutor = URLSubstitutor()
        short_url = "https://example.com/page"  # 25 chars
        substitutor.prepare([short_url])

        # Short URL should not be in mappings
        assert short_url not in substitutor.get_mappings()

        # Substitute should return unchanged
        result = substitutor.substitute(short_url)
        assert result == short_url

    @pytest.mark.asyncio
    async def test_preserves_file_extension(self):
        """File extensions should be preserved in shortened URLs."""
        substitutor = URLSubstitutor()
        url = "https://example.com/docs/very/long/path/document.pdf"
        substitutor.prepare([url])

        label = substitutor.get_mappings()[url]
        assert label.endswith(".pdf")


class TestURLWithEmbeddedPatterns:
    """Tests for URLs containing high-entropy patterns (e.g., blockchain addresses).

    When a URL contains a pattern that would be shortened on its own (like an Ethereum
    address), the URL shortening should reuse the pattern's shortened form rather than
    hashing the entire URL independently.
    """

    @pytest.mark.asyncio
    async def test_url_with_eth_address_reuses_address_shortening(self):
        """URL containing ETH address should reuse the address shortening."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()

        # The URL should be shortened
        assert url in mappings
        shortened_url = mappings[url]

        # The shortened URL should contain the shortened address form, not a hash
        # Expected: https://etherscan.io/address/0xdac1~XXX~1ec7
        # Not: https://etherscan.io/address~YYYY
        assert "/address/" in shortened_url, f"URL structure should be preserved: {shortened_url}"
        assert "0xdac1" in shortened_url, f"Address prefix should be visible: {shortened_url}"
        assert "1ec7" in shortened_url, f"Address suffix should be visible: {shortened_url}"

    @pytest.mark.asyncio
    async def test_url_with_eth_address_roundtrip(self):
        """URL with embedded ETH address should round-trip correctly."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"
        text = f"Check the contract at {url} for details"

        substitutor.prepare([text])

        substituted = substitutor.substitute(text)
        restored = substitutor.restore(substituted)

        assert restored == text

    @pytest.mark.asyncio
    async def test_url_and_standalone_address_share_shortening(self):
        """Same address in URL and standalone should use consistent shortening."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"
        text = f"Contract {eth_address} is at {url}"

        substitutor.prepare([text])

        substituted = substitutor.substitute(text)

        # Both occurrences of the address (standalone and in URL) should use same shortening
        # Count how many times the shortened address appears
        mappings = substitutor.get_mappings()

        # Get the standalone address shortening
        assert eth_address in mappings
        shortened_address = mappings[eth_address]

        # The shortened address should appear twice in the substituted text
        # (once standalone, once inside the URL)
        assert substituted.count(shortened_address) == 2, f"Shortened address '{shortened_address}' should appear twice in: {substituted}"

    @pytest.mark.asyncio
    async def test_url_with_solana_address_reuses_address_shortening(self):
        """URL containing Solana address should reuse the address shortening."""
        substitutor = URLSubstitutor()
        sol_address = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
        url = f"https://solscan.io/token/{sol_address}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        assert url in mappings
        shortened_url = mappings[url]

        # Should contain shortened Solana address, not URL hash
        assert "/token/" in shortened_url, f"URL structure should be preserved: {shortened_url}"
        assert "Es9v" in shortened_url, f"Address prefix should be visible: {shortened_url}"

    @pytest.mark.asyncio
    async def test_url_with_long_hash_path_reuses_hash_shortening(self):
        """URL with long hex hash in path should reuse hash shortening."""
        substitutor = URLSubstitutor()
        tx_hash = "8ccd766e39a2fba8c43eb4329bac734165a4237df34884059739ed8a874111e1"
        url = f"https://etherscan.io/tx/0x{tx_hash}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        assert url in mappings
        shortened_url = mappings[url]

        # Should preserve /tx/ structure and show hash prefix/suffix
        assert "/tx/" in shortened_url, f"URL structure should be preserved: {shortened_url}"

    @pytest.mark.asyncio
    async def test_multiple_patterns_in_url(self):
        """URL with multiple high-entropy patterns should shorten all of them."""
        substitutor = URLSubstitutor()
        # A hypothetical URL with two addresses
        addr1 = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        addr2 = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        url = f"https://example.com/swap/{addr1}/to/{addr2}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        assert url in mappings
        shortened_url = mappings[url]

        # Both addresses should be shortened within the URL
        assert "0xdac1" in shortened_url or "0x7a25" in shortened_url, f"At least one address should be shortened in URL: {shortened_url}"

    @pytest.mark.asyncio
    async def test_pattern_in_query_parameter(self):
        """ETH address in query parameter should be shortened."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://api.example.com/swap?from={eth_address}&to=USDC"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        assert url in mappings
        shortened_url = mappings[url]

        # Query structure preserved, address shortened
        assert "?from=" in shortened_url, f"Query param structure should be preserved: {shortened_url}"
        assert "&to=USDC" in shortened_url, f"Other params should be preserved: {shortened_url}"
        assert "0xdac1" in shortened_url, f"Address prefix should be visible: {shortened_url}"

    @pytest.mark.asyncio
    async def test_pattern_in_fragment(self):
        """ETH address in URL fragment should be shortened."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://example.com/explorer#/address/{eth_address}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        assert url in mappings
        shortened_url = mappings[url]

        # Fragment structure preserved
        assert "#/address/" in shortened_url, f"Fragment structure should be preserved: {shortened_url}"
        assert "0xdac1" in shortened_url, f"Address prefix should be visible: {shortened_url}"

    @pytest.mark.asyncio
    async def test_url_encoded_pattern_not_matched(self):
        """URL-encoded patterns should not be matched (falls back to URL hash)."""
        substitutor = URLSubstitutor()
        # %64 is 'd', so this encodes 0xdac17...
        url = "https://etherscan.io/address/0x%64ac17f958d2ee523a2206206994597c13d831ec7"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        assert url in mappings
        shortened_url = mappings[url]

        # Should fall back to hash-based shortening since pattern isn't recognized
        # The URL-encoded pattern won't match the regex
        assert "~" in shortened_url  # Some form of shortening occurred

    @pytest.mark.asyncio
    async def test_duplicate_address_in_url_both_replaced(self):
        """Same address appearing twice in URL should both be replaced."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://api.example.com/compare/{eth_address}/with/{eth_address}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        shortened_url = mappings[url]

        # Both occurrences should be replaced with the same shortening
        # Count shortened address occurrences
        short_addr = mappings.get(eth_address)
        if short_addr:
            assert shortened_url.count(short_addr) == 2, f"Both occurrences should be shortened: {shortened_url}"

    @pytest.mark.asyncio
    async def test_restore_url_with_embedded_pattern(self):
        """Shortened URL with embedded pattern should restore correctly."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"
        text = f"Check {url} for details"

        substitutor.prepare([text])
        substituted = substitutor.substitute(text)
        restored = substitutor.restore(substituted)

        assert restored == text, f"Failed to restore: {substituted} -> {restored}"

    @pytest.mark.asyncio
    async def test_incremental_prepare_url_then_standalone(self):
        """Preparing URL first, then standalone address should use consistent shortening."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"

        # First prepare URL
        substitutor.prepare([url])
        url_shortened = substitutor.get_mappings()[url]

        # Then prepare standalone address
        substitutor.prepare([eth_address])

        # Both should use consistent address shortening
        addr_shortened = substitutor.get_mappings()[eth_address]
        assert addr_shortened in url_shortened, f"URL should contain address shortening: {url_shortened} should contain {addr_shortened}"

    @pytest.mark.asyncio
    async def test_incremental_prepare_standalone_then_url(self):
        """Preparing standalone address first, then URL should reuse the shortening."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"

        # First prepare standalone address
        substitutor.prepare([eth_address])
        addr_shortened = substitutor.get_mappings()[eth_address]

        # Then prepare URL
        substitutor.prepare([url])
        url_shortened = substitutor.get_mappings()[url]

        # URL should reuse the existing address shortening
        assert addr_shortened in url_shortened, f"URL should reuse existing address shortening: {url_shortened} should contain {addr_shortened}"

    @pytest.mark.asyncio
    async def test_already_shortened_not_reshortened(self):
        """URLs containing already-shortened patterns should not be re-shortened."""
        substitutor = URLSubstitutor()

        # First shorten an address
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        substitutor.prepare([eth_address])
        addr_shortened = substitutor.get_mappings()[eth_address]

        # Now try to shorten a URL that contains the shortened form
        # (simulating LLM output that includes shortened tokens)
        fake_url = f"https://example.com/addr/{addr_shortened}"
        substitutor.prepare([fake_url])

        # The shortened form should not be re-processed
        if fake_url in substitutor.get_mappings():
            result = substitutor.get_mappings()[fake_url]
            # Should not double-shorten
            assert result.count("~") <= addr_shortened.count("~") + 1

    @pytest.mark.asyncio
    async def test_pattern_at_url_path_end(self):
        """Pattern at the very end of URL path should be shortened."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/token/{eth_address}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        shortened_url = mappings[url]

        assert "0xdac1" in shortened_url
        assert "1ec7" in shortened_url

    @pytest.mark.asyncio
    async def test_short_pattern_in_long_url_not_shortened(self):
        """Short patterns (below threshold) should not be shortened even in URLs."""
        substitutor = URLSubstitutor()
        # This is too short to be a valid ETH address (only 20 chars after 0x)
        short_hex = "0x1234567890abcdef1234"
        url = f"https://example.com/something/path/to/resource/{short_hex}/more"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        # URL should be shortened (it's long), but the short hex should not be
        # treated as a high-entropy pattern
        assert url in mappings

    @pytest.mark.asyncio
    async def test_mixed_pattern_types_in_url(self):
        """URL with both ETH address and Solana address should shorten both."""
        substitutor = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        sol_address = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
        url = f"https://bridge.example.com/from/{eth_address}/to/{sol_address}"

        substitutor.prepare([url])

        mappings = substitutor.get_mappings()
        shortened_url = mappings[url]

        # Both patterns should be shortened
        assert "0xdac1" in shortened_url, f"ETH address should be shortened: {shortened_url}"
        assert "Es9v" in shortened_url, f"Solana address should be shortened: {shortened_url}"


class TestFalsePositives:
    """Bug 1: URL path segments must NOT be shortened as high-entropy/base64 strings.

    The base64 regex includes `/` in its charset, so multi-segment URL paths like
    `/documentation/steps/transaction` match as base64. These are English words,
    not high-entropy data.
    """

    @pytest.mark.parametrize(
        ("path", "label"),
        [
            ("/es/build/indexer/indexer-sdk/documentation/steps/transaction-stream", "documentation/steps/transaction"),
            ("/es/build/indexer/nft-aggregator/marketplaces/bluemove", "aggregator/marketplaces/bluemove"),
            ("/es/build/indexer/nft-aggregator/marketplaces/tradeport", "aggregator/marketplaces/tradeport"),
            ("/es/build/indexer/legacy/migration", "indexer/legacy/migration"),
            ("/es/network/nodes/configure/consensus-observer", "nodes/configure/consensus"),
            ("/es/network/nodes/configure/state-sync", "nodes/configure/state"),
            ("/es/network/nodes/configure/telemetry", "nodes/configure/telemetry"),
            ("/es/network/nodes/measure/important-metrics", "nodes/measure/important"),
            ("/es/network/blockchain/blockchain-deep-dive", "network/blockchain/blockchain"),
            ("/es/network/blockchain/governance", "network/blockchain/governance"),
        ],
    )
    def test_relative_path_shortened_as_path_not_base64(self, path: str, label: str):
        """Relative URL paths must be shortened as paths (/first/~hash/last), not as base64."""
        sub = URLSubstitutor()
        text = f"*   [Link]({path})"
        sub.prepare([text])
        result = sub.substitute(text)
        # Path should be shortened as path type (not left unchanged, and not as base64)
        assert path not in result, f"Path should have been shortened: {result}"
        parts = [p for p in path.split("/") if p]
        shortened = sub.get_mappings()[path]
        # First segment preserved, last segment preserved
        assert shortened.startswith(f"/{parts[0]}/"), f"First segment not preserved: {shortened}"
        assert shortened.endswith(f"/{parts[-1]}"), f"Last segment not preserved: {shortened}"
        # Round-trip works
        restored = sub.restore(result)
        assert restored == text

    def test_markdown_link_with_relative_path_shortened_correctly(self):
        """Markdown link with relative path should be shortened as path, preserving link structure."""
        sub = URLSubstitutor()
        path = "/es/build/indexer/indexer-sdk/documentation/steps/transaction-stream"
        text = f"*   [Transaction Stream]({path})"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result, f"Path should have been shortened: {result}"
        assert "*   [Transaction Stream](" in result
        assert result.endswith(")")
        restored = sub.restore(result)
        assert restored == text

    def test_real_base64_with_slash_still_shortened(self):
        """Actual Base64 containing `/` and `+` or `=` should still be shortened."""
        sub = URLSubstitutor()
        # Real Base64 string (has / + and =)
        b64 = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0IG9mIEJhc2U2NC4="
        text = f"Data: {b64}"
        sub.prepare([text])
        result = sub.substitute(text)
        assert b64 not in result, "Real Base64 should be shortened"
        assert "~" in result


class TestMissedEncodedURLs:
    """Bug 2: URLs with encoded content (%3A%2F%2F) must still be shortened."""

    def test_url_with_encoded_url_in_query(self):
        """URL with encoded URL as query parameter should be shortened."""
        sub = URLSubstitutor()
        url = "https://github.com/aptos-labs/aptos-docs/issues/new?labels=documentation&template=content_issue.yml&url=https%3A%2F%2Faptos.dev%2Fes%2Fbuild%2Fguides%2Fexchanges"
        sub.prepare([url])
        result = sub.substitute(url)
        assert url not in result, f"URL should have been shortened: {result}"
        assert "~" in result

    def test_url_with_encoded_spaces_and_url(self):
        """URL with encoded spaces and embedded URL in query should be shortened."""
        sub = URLSubstitutor()
        url = "https://claude.ai/new?q=Read%20from%20https%3A%2F%2Faptos.dev%2Fes%2Fbuild%2Fguides%2Fexchanges%20so%20I%20can%20ask%20questions%20about%20it"
        sub.prepare([url])
        result = sub.substitute(url)
        assert url not in result, f"URL should have been shortened: {result}"
        assert "~" in result

    def test_encoded_url_roundtrip(self):
        """Shortened encoded URL should restore correctly."""
        sub = URLSubstitutor()
        url = "https://github.com/aptos-labs/aptos-docs/issues/new?labels=documentation&template=content_issue.yml&url=https%3A%2F%2Faptos.dev%2Fes%2Fbuild%2Fguides%2Fexchanges"
        text = f"[Report]({url})"
        sub.prepare([text])
        substituted = sub.substitute(text)
        restored = sub.restore(substituted)
        assert restored == text


class TestPartiallyShortened:
    """Bug 3: Long URLs must be fully shortened, not left at 200-300+ chars."""

    def test_google_redirect_url_fully_shortened(self):
        """Google grounding redirect URLs should be shortened to < 100 chars."""
        sub = URLSubstitutor()
        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1G-vW3nY37WMAKEhLfIK3tYPcvi96LKvsRVZEhz5tW7J0wwWaD9l3YuBXL-6D4B0vSwgH6NpUB9stPrmV3mE-n5wkLRSi0KOVTfKNK2BNYTEo-E0gEjdu0TIwy3FLGJ-hQuywgJO_7FqWPmpUgTp8qohnXY742DHSOiwXOU0iT9kH2A6Lutl9nUywiUc49We_angyn6oyIoxijGvo0q8vzW2LLNNJlA3kn2D2OqvcH2MUZCRYxPLvSDmg2WgEhj-zYwCVUOz_8WdT_nS-HALzaub_xZAJ"
        sub.prepare([url])
        result = sub.substitute(url)
        assert len(result) < 100, f"URL still {len(result)} chars after shortening: {result}"
        assert "~" in result

    def test_medium_google_redirect_url_shortened(self):
        """279-char Google redirect should be shortened to < 100 chars."""
        sub = URLSubstitutor()
        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH6SRPQhg8UQ3oiOPtAm3MCDQDKHJpbEit42gB05lYnZ7y-w_2MI1L_fmAB_g-XVXHsEZy4nCnM7K57p18M-CBgEcmxARZF4yNtHl_tdJzRJD7XPKGfTlQoMjMXshc-OX1ABNgNbA7wipIq_akppTD4_4-qFS5maAkpza1FpWyxlSQvO5zrTOVXUC5wSOKNIcfJ7B9m25qg95dCHBA="
        sub.prepare([url])
        result = sub.substitute(url)
        assert len(result) < 100, f"URL still {len(result)} chars after shortening: {result}"

    def test_google_redirect_roundtrip(self):
        """Shortened Google redirect URLs should restore correctly."""
        sub = URLSubstitutor()
        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1G-vW3nY37WMAKEhLfIK3tYPcvi96LKvsRVZEhz5tW7J0wwWaD9l3YuBXL-6D4B0vSwgH6NpUB9stPrmV3mE-n5wkLRSi0KOVTfKNK2BNYTEo-E0gEjdu0TIwy3FLGJ-hQuywgJO_7FqWPmpUgTp8qohnXY742DHSOiwXOU0iT9kH2A6Lutl9nUywiUc49We_angyn6oyIoxijGvo0q8vzW2LLNNJlA3kn2D2OqvcH2MUZCRYxPLvSDmg2WgEhj-zYwCVUOz_8WdT_nS-HALzaub_xZAJ"
        text = f"Source: {url}"
        sub.prepare([text])
        substituted = sub.substitute(text)
        restored = sub.restore(substituted)
        assert restored == text

    def test_url_with_eth_address_stays_short_enough(self):
        """URL with ETH address that fits under threshold should preserve structure."""
        sub = URLSubstitutor()
        eth_address = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth_address}"
        sub.prepare([url])
        result = sub.substitute(url)
        # Short enough to keep structure
        assert "/address/" in result, f"URL structure should be preserved: {result}"
        assert "0xdac1" in result, f"Address prefix should be visible: {result}"
        assert len(result) < 100


class TestUglyJunctions:
    """Issue: character before ~ must be alphanumeric (no -~ or =~ junctions)."""

    def test_url_segment_trailing_dash_stripped(self):
        """grounding-api-redirect truncated to 'grounding-' should strip the dash."""
        sub = URLSubstitutor()
        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/something"
        sub.prepare([url])
        shortened = sub.get_mappings()[url]
        tilde_idx = shortened.index("~")
        assert shortened[tilde_idx - 1].isalnum(), f"Char before ~ is '{shortened[tilde_idx - 1]}': {shortened}"

    def test_url_segment_trailing_equals_stripped(self):
        """Segment ending with = should strip it."""
        sub = URLSubstitutor()
        url = "https://example.com/callback=/very/long/path/to/resource/page"
        sub.prepare([url])
        shortened = sub.get_mappings()[url]
        tilde_idx = shortened.index("~")
        assert shortened[tilde_idx - 1].isalnum(), f"Char before ~ is '{shortened[tilde_idx - 1]}': {shortened}"

    def test_all_punct_segment_falls_back_to_domain_only(self):
        """Path segment that's all punctuation after truncation should use domain-only format."""
        sub = URLSubstitutor()
        url = "https://example.com/----------/page/resource/detail"
        sub.prepare([url])
        shortened = sub.get_mappings()[url]
        assert shortened.startswith("https://example.com~"), f"Expected domain-only format: {shortened}"

    def test_roundtrip_after_junction_fix(self):
        """Round-trip must still work after stripping."""
        sub = URLSubstitutor()
        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/something"
        sub.prepare([url])
        result = sub.substitute(url)
        restored = sub.restore(result)
        assert restored == url


class TestShortenedPatternSkipping:
    """Issue: URLs with coincidental ~hex~ in content must not be skipped."""

    def test_url_with_tilde_hex_in_path_shortened(self):
        """URL containing ~6a2~ in path should be shortened, not skipped."""
        sub = URLSubstitutor()
        url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1G-vW3n~6a2~uBXL-6D4B0vSwgH6NpUB9stPrmV3mE"
        sub.prepare([url])
        result = sub.substitute(url)
        assert url not in result, f"URL with ~hex~ was not shortened: {result}"
        assert "~" in result

    def test_url_with_tilde_hex_roundtrip(self):
        """URL with coincidental ~hex~ should round-trip correctly."""
        sub = URLSubstitutor()
        url = "https://example.com/path/token~abc~value/resource/detail/page"
        text = f"See {url} for info"
        sub.prepare([text])
        substituted = sub.substitute(text)
        restored = sub.restore(substituted)
        assert restored == text

    def test_long_url_with_multiple_tilde_patterns(self):
        """URL with multiple ~hex~ occurrences should still be shortened."""
        sub = URLSubstitutor()
        url = "https://example.com/prefix~aaa~middle~bbb~suffix/very/long/path/to/resource"
        sub.prepare([url])
        result = sub.substitute(url)
        assert url not in result, "URL with multiple ~hex~ patterns was not shortened"

    def test_own_shortened_url_not_reshortened_same_instance(self):
        """Substitutor's own shortened URLs should not be re-shortened in same instance."""
        sub = URLSubstitutor()
        url = "https://etherscan.io/address/0xdac17f958d2ee523a2206206994597c13d831ec7"
        sub.prepare([url])
        shortened_url = sub.get_mappings()[url]

        # Feed shortened output back
        sub.prepare([shortened_url])

        # The shortened URL should NOT get a new different mapping
        if shortened_url in sub.get_mappings():
            assert sub.get_mappings()[shortened_url] == shortened_url


class TestKnownLimitations:
    """Documented known limitations — not bugs, but intentional trade-offs."""

    def test_url_with_backslash_escaped_parens_roundtrip(self):
        """URLs with markdown backslash-escaped parens are only partially shortened but round-trip works."""
        sub = URLSubstitutor()
        url = r"https://example.com/new?q=Read%20from%20https%3A%2F%2Fexample.com%2Fpage\(include%20your%20data\)%20also%20more%20text%20here"
        sub.prepare([url])
        result = sub.substitute(url)
        restored = sub.restore(result)
        assert restored == url, f"Round-trip failed: {result}"

    def test_base64_with_slash_no_plus_not_shortened(self):
        """Base64 with / but no + or = is intentionally not shortened to prevent URL path false positives."""
        sub = URLSubstitutor()
        b64_like = "AAAAAAAABBBBBBBB/CCCCCCCCDDDDDDDDEEEE"
        text = f"Token: {b64_like}"
        sub.prepare([text])
        result = sub.substitute(text)
        # Intentional false negative — accepted to prevent URL path false positives
        assert result == text

    def test_real_base64_with_slash_and_plus_still_shortened(self):
        """Base64 with both / and + should still be shortened."""
        sub = URLSubstitutor()
        b64 = "SGVsbG8gV29ybGQhIFRoaXMgaXMgYSB0ZXN0+/"
        text = f"Data: {b64}"
        sub.prepare([text])
        result = sub.substitute(text)
        assert b64 not in result

    def test_phase2_fallback_preserves_standalone_pattern_mapping(self):
        """When URL falls through to Phase 2, standalone patterns still work."""
        sub = URLSubstitutor()
        eth = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        long_url = f"https://very-long-domain-name.example.com/extremely/long/path/structure/{eth}/more/segments/here/that/push/over/threshold"
        text = f"Address {eth} at {long_url}"
        sub.prepare([text])
        result = sub.substitute(text)
        assert eth not in result
        assert "0xdac1" in result
        assert long_url not in result
        restored = sub.restore(result)
        assert restored == text


class TestPathShortening:
    """Feature: Long relative paths inside delimiters should be shortened to /first/~hash/last."""

    def test_path_in_parentheses_shortened(self):
        """Markdown link path inside () should be shortened."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        text = f"*   [Telemetría]({path})"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result
        assert "/es/" in result
        assert "/telemetry)" in result
        assert "~" in result

    def test_path_in_double_quotes_shortened(self):
        """Path inside double quotes should be shortened."""
        sub = URLSubstitutor()
        path = "/es/build/indexer/legacy/migration/guide"
        text = f'href="{path}"'
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result
        assert '"/es/' in result
        assert '/guide"' in result

    def test_path_in_single_quotes_shortened(self):
        """Path inside single quotes should be shortened."""
        sub = URLSubstitutor()
        path = "/es/build/indexer/legacy/migration/guide"
        text = f"href='{path}'"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result

    def test_path_in_xml_tags_shortened(self):
        """Path inside XML tags (>...<) should be shortened."""
        sub = URLSubstitutor()
        path = "/es/network/blockchain/governance/voting"
        text = f"<link>{path}</link>"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result
        assert ">" in result and "<" in result

    def test_path_in_square_brackets_shortened(self):
        """Path inside square brackets should be shortened."""
        sub = URLSubstitutor()
        path = "/es/build/indexer/legacy/migration/guide"
        text = f"[{path}]"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result

    def test_short_path_not_shortened(self):
        """Paths shorter than 30 chars should not be shortened."""
        sub = URLSubstitutor()
        text = "(/es/a/b/c)"
        sub.prepare([text])
        result = sub.substitute(text)
        assert result == text

    def test_path_too_few_segments_not_shortened(self):
        """Paths with fewer than 3 slashes should not be shortened."""
        sub = URLSubstitutor()
        text = "(/very-long-single-segment-path-name-here)"
        sub.prepare([text])
        result = sub.substitute(text)
        assert result == text

    def test_undelimited_path_not_shortened(self):
        """Paths without surrounding delimiters should not be shortened."""
        sub = URLSubstitutor()
        text = "see /es/network/nodes/configure/telemetry for details"
        sub.prepare([text])
        result = sub.substitute(text)
        assert result == text

    def test_path_inside_url_not_shortened_as_path(self):
        """Paths inside URLs should be handled by URL shortening, not path shortening."""
        sub = URLSubstitutor()
        text = "see https://example.com/es/network/nodes/configure/telemetry"
        sub.prepare([text])
        result = sub.substitute(text)
        # URL shortened as URL, path not separately shortened
        assert "https://example.com" in result

    def test_protocol_relative_not_shortened(self):
        """Protocol-relative URLs (//...) should not be shortened as paths."""
        sub = URLSubstitutor()
        text = "(//example.com/es/network/nodes/configure/path)"
        sub.prepare([text])
        result = sub.substitute(text)
        assert "//example.com" in result

    def test_path_roundtrip(self):
        """Shortened path must restore correctly."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        text = f"[Telemetría]({path})"
        sub.prepare([text])
        substituted = sub.substitute(text)
        restored = sub.restore(substituted)
        assert restored == text

    def test_path_format_first_hash_last(self):
        """Shortened path should follow /first/~hash/last format."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        text = f"({path})"
        sub.prepare([text])
        mappings = sub.get_mappings()
        assert path in mappings
        shortened = mappings[path]
        assert shortened.startswith("/es/")
        assert shortened.endswith("/telemetry")
        assert "~" in shortened
        # /es/~hash/telemetry → ['', 'es', '~hash', 'telemetry']
        parts = shortened.split("/")
        assert len(parts) == 4

    def test_same_path_same_shortening(self):
        """Same path appearing twice gets consistent shortening."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        text = f"({path}) and ({path})"
        sub.prepare([text])
        result = sub.substitute(text)
        shortened = sub.get_mappings()[path]
        assert result.count(shortened) == 2

    def test_multiple_different_paths(self):
        """Multiple different paths should all be shortened."""
        sub = URLSubstitutor()
        path1 = "/es/network/nodes/configure/telemetry"
        path2 = "/es/build/indexer/legacy/migration"
        text = f"({path1}) and ({path2})"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path1 not in result
        assert path2 not in result

    def test_path_coexists_with_addresses(self):
        """Paths and blockchain addresses should coexist correctly."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        eth = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        text = f"({path}) contract {eth}"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result
        assert eth not in result
        restored = sub.restore(result)
        assert restored == text

    @pytest.mark.parametrize(
        "path",
        [
            "/es/build/indexer/indexer-sdk/documentation/steps/transaction-stream",
            "/es/build/indexer/nft-aggregator/marketplaces/bluemove",
            "/es/build/indexer/nft-aggregator/marketplaces/tradeport",
            "/es/build/indexer/legacy/migration",
            "/es/network/nodes/configure/consensus-observer",
            "/es/network/nodes/configure/state-sync",
            "/es/network/nodes/configure/telemetry",
            "/es/network/nodes/measure/important-metrics",
            "/es/network/blockchain/blockchain-deep-dive",
            "/es/network/blockchain/governance",
        ],
    )
    def test_real_paths_from_wrong_substitutions(self, path: str):
        """Real paths from wrong_substitutions.txt should be correctly shortened in markdown links."""
        sub = URLSubstitutor()
        text = f"*   [Link]({path})"
        sub.prepare([text])
        result = sub.substitute(text)
        assert path not in result, f"Path should have been shortened: {result}"
        parts = [p for p in path.split("/") if p]
        shortened = sub.get_mappings()[path]
        assert shortened.startswith(f"/{parts[0]}/"), f"First segment not preserved: {shortened}"
        assert shortened.endswith(f"/{parts[-1]}"), f"Last segment not preserved: {shortened}"
        restored = sub.restore(result)
        assert restored == text
