"""Tests for LLM corruption resilience in URLSubstitutor.restore().

LLMs may corrupt shortened forms in two ways:
1. Unicode ellipsis: `...` → `…` (U+2026)
2. Case change: `0xdAC17F95` → `0xdac17f95`

restore() handles both via Unicode normalization and case-insensitive lookup.
"""

from ai_pipeline_core.llm import URLSubstitutor


ETH_ADDRESS = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
ETH_ADDRESS_2 = "0xdac17f958d2ee523a2206206994597c13d831ec7"
SOL_ADDRESS = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
LONG_URL = "https://etherscan.io/address/0xdac17f958d2ee523a2206206994597c13d831ec7"
LONG_URL_2 = "https://polygonscan.com/token/0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"


def _prepare(sub: URLSubstitutor, *texts: str) -> dict[str, str]:
    """Prepare substitutor and return original → shortened mappings."""
    sub.prepare(list(texts))
    return sub.get_mappings()


class TestUnicodeEllipsisRestore:
    """Category 1: Unicode ellipsis normalization."""

    def test_unicode_ellipsis_restores_address(self):
        """ETH address with `…` (Unicode ellipsis) should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        # Simulate LLM converting ... to …
        corrupted = shortened.replace("...", "\u2026")
        text = f"Contract: {corrupted}"
        restored = sub.restore(text)
        assert restored == f"Contract: {ETH_ADDRESS}"

    def test_unicode_ellipsis_restores_url(self):
        """Long URL with `…` should restore."""
        sub = URLSubstitutor()
        url = "https://example.com/docs/api/v2/reference/contracts/very/long/path/to/resource/page"
        mappings = _prepare(sub, url)
        shortened = mappings[url]

        corrupted = shortened.replace("...", "\u2026")
        restored = sub.restore(corrupted)
        assert restored == url

    def test_unicode_ellipsis_restores_path(self):
        """Path with `…` should restore."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        text = f"({path})"
        _prepare(sub, text)
        mappings = sub.get_mappings()
        shortened = mappings[path]

        corrupted = shortened.replace("...", "\u2026")
        result = sub.restore(f"({corrupted})")
        assert result == text

    def test_unicode_ellipsis_multiple_patterns(self):
        """Multiple patterns all with `…` should all restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS, ETH_ADDRESS_2, SOL_ADDRESS)

        parts = []
        for orig in [ETH_ADDRESS, ETH_ADDRESS_2, SOL_ADDRESS]:
            corrupted = mappings[orig].replace("...", "\u2026")
            parts.append(corrupted)

        text = " ".join(parts)
        restored = sub.restore(text)
        assert ETH_ADDRESS in restored
        assert ETH_ADDRESS_2 in restored
        assert SOL_ADDRESS in restored


class TestCaseInsensitiveRestore:
    """Category 2: Case-insensitive restoration."""

    def test_lowercase_address_restores(self):
        """Lowercased shortened address should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        lowered = shortened.lower()
        restored = sub.restore(lowered)
        assert restored == ETH_ADDRESS

    def test_uppercase_address_restores(self):
        """Uppercased shortened address should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        uppered = shortened.upper()
        restored = sub.restore(uppered)
        assert restored == ETH_ADDRESS

    def test_case_change_url_restores(self):
        """Case-changed shortened URL should restore."""
        sub = URLSubstitutor()
        url = "https://example.com/docs/api/v2/reference/contracts/very/long/path/to/resource/page"
        mappings = _prepare(sub, url)
        shortened = mappings[url]

        lowered = shortened.lower()
        restored = sub.restore(lowered)
        assert restored == url


class TestCombinedCorruption:
    """Category 3: Unicode ellipsis + case change combined."""

    def test_ellipsis_plus_lowercase(self):
        """Lowercased AND ellipsis-corrupted address should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        corrupted = shortened.replace("...", "\u2026").lower()
        restored = sub.restore(corrupted)
        assert restored == ETH_ADDRESS

    def test_ellipsis_plus_lowercase_url(self):
        """Lowercased AND ellipsis-corrupted URL should restore."""
        sub = URLSubstitutor()
        url = "https://example.com/docs/api/v2/reference/contracts/very/long/path/to/resource/page"
        mappings = _prepare(sub, url)
        shortened = mappings[url]

        corrupted = shortened.replace("...", "\u2026").lower()
        restored = sub.restore(corrupted)
        assert restored == url


class TestURLWithEmbeddedPatternsCorrupted:
    """Category 4: URLs with embedded patterns, corruption applied."""

    def test_url_embedded_eth_ellipsis(self):
        """Etherscan URL with inner address ellipsis-corrupted should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, LONG_URL)
        shortened_url = mappings[LONG_URL]

        corrupted = shortened_url.replace("...", "\u2026")
        restored = sub.restore(corrupted)
        assert restored == LONG_URL

    def test_standalone_and_url_both_corrupted(self):
        """Same address corrupted in text and in URL should both restore."""
        sub = URLSubstitutor()
        eth = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth}"
        text = f"Contract {eth} at {url}"
        mappings = _prepare(sub, text)

        short_eth = mappings[eth]
        short_url = mappings[url]
        corrupted_eth = short_eth.replace("...", "\u2026")
        corrupted_url = short_url.replace("...", "\u2026")

        result = sub.restore(f"Contract {corrupted_eth} at {corrupted_url}")
        assert eth in result
        assert url in result


class TestFalsePositiveResistance:
    """Category 5: Natural text should not be falsely restored."""

    def test_natural_text_not_falsely_restored(self):
        """Random text with hex-like chars should not be replaced."""
        sub = URLSubstitutor()
        _prepare(sub, ETH_ADDRESS)

        innocent_text = "The value 0x7a25abc1488D is unrelated"
        restored = sub.restore(innocent_text)
        assert restored == innocent_text

    def test_natural_ellipsis_not_falsely_matched(self):
        """Natural `...` in prose should not be mistaken for a shortened form."""
        sub = URLSubstitutor()
        _prepare(sub, ETH_ADDRESS)

        text = "Loading... please wait"
        restored = sub.restore(text)
        assert restored == text

    def test_natural_unicode_ellipsis_not_falsely_matched(self):
        """Natural `…` in prose should not be mistaken for a shortened form."""
        sub = URLSubstitutor()
        _prepare(sub, ETH_ADDRESS)

        text = "Loading\u2026 please wait"
        restored = sub.restore(text)
        # Unicode ellipsis gets normalized to ... but shouldn't match any pattern
        assert "Loading" in restored
        assert "please wait" in restored


class TestIntegrationWithSubstitute:
    """Category 6: Integration with substitute() — corrupted forms should not interfere."""

    def test_corrupted_form_passes_through_substitute(self):
        """Corrupted shortened form is too short for pattern regexes — substitute should not touch it."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]
        corrupted = shortened.replace("...", "\u2026")

        result = sub.substitute(corrupted)
        # The corrupted form should pass through (it's not in _forward)
        assert result == corrupted

    def test_substitute_after_corrupted_restore(self):
        """Restored original should re-shorten correctly."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        corrupted = shortened.replace("...", "\u2026")
        restored = sub.restore(corrupted)
        assert restored == ETH_ADDRESS

        re_shortened = sub.substitute(restored)
        assert re_shortened == shortened

    def test_roundtrip_through_corruption(self):
        """Full cycle: substitute → corrupt address only → restore → re-substitute should be stable."""
        sub = URLSubstitutor()
        text = f"Contract {ETH_ADDRESS} deployed"
        sub.prepare([text])

        shortened = sub.substitute(text)
        addr_short = sub.get_mappings()[ETH_ADDRESS]

        # Corrupt only the shortened address form (not the surrounding text)
        corrupted_addr = addr_short.replace("...", "\u2026").lower()
        corrupted = shortened.replace(addr_short, corrupted_addr)

        restored = sub.restore(corrupted)
        assert ETH_ADDRESS in restored

        re_shortened = sub.substitute(restored)
        assert re_shortened == shortened
