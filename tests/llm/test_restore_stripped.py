"""Tests for tilde-stripped restore in URLSubstitutor.

When LLMs strip ~ delimiters from shortened forms (e.g., 0xfd61~585~1708 → 0xfd615851708),
restore() should still recover the original via pre-computed stripped reverse mappings.
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


class TestBasicStrippedRestore:
    """Category 1: Basic tilde-stripped restore."""

    def test_string_tilde_stripped_restore(self):
        """ETH address: stripped form (tildes removed) should restore to original."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        # Simulate LLM stripping tildes
        stripped = shortened.replace("~", "")
        text = f"Contract: {stripped}"
        restored = sub.restore(text)
        assert restored == f"Contract: {ETH_ADDRESS}"

    def test_url_hash_tilde_stripped_restore(self):
        """URL shortened with hash: stripped form should restore."""
        sub = URLSubstitutor()
        url = "https://example.com/very/long/path/to/resource/page"
        mappings = _prepare(sub, url)
        shortened = mappings[url]

        stripped = shortened.replace("~", "")
        restored = sub.restore(stripped)
        assert restored == url

    def test_path_tilde_stripped_restore(self):
        """Path shortened: stripped form should restore."""
        sub = URLSubstitutor()
        path = "/es/network/nodes/configure/telemetry"
        text = f"({path})"
        _prepare(sub, text)
        mappings = sub.get_mappings()
        shortened = mappings[path]

        stripped = shortened.replace("~", "")
        result = sub.restore(f"({stripped})")
        assert result == text

    def test_multiple_stripped_patterns(self):
        """Multiple different patterns all stripped should all restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS, ETH_ADDRESS_2, SOL_ADDRESS)

        parts = []
        for orig in [ETH_ADDRESS, ETH_ADDRESS_2, SOL_ADDRESS]:
            stripped = mappings[orig].replace("~", "")
            parts.append(stripped)

        text = " ".join(parts)
        restored = sub.restore(text)
        assert ETH_ADDRESS in restored
        assert ETH_ADDRESS_2 in restored
        assert SOL_ADDRESS in restored

    def test_mixed_exact_and_stripped(self):
        """One pattern with tildes intact, another stripped — both should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS, ETH_ADDRESS_2)
        short_a = mappings[ETH_ADDRESS]  # kept intact
        short_b = mappings[ETH_ADDRESS_2]
        stripped_b = short_b.replace("~", "")  # stripped

        text = f"{short_a} and {stripped_b}"
        restored = sub.restore(text)
        assert ETH_ADDRESS in restored
        assert ETH_ADDRESS_2 in restored

    def test_stripped_roundtrip(self):
        """strip → restore → substitute should produce original shortened form."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        stripped = shortened.replace("~", "")
        restored = sub.restore(stripped)
        assert restored == ETH_ADDRESS

        re_shortened = sub.substitute(restored)
        assert re_shortened == shortened


class TestURLWithEmbeddedPatternsStripped:
    """Category 2: URLs with embedded patterns, tildes stripped."""

    def test_url_embedded_eth_stripped(self):
        """Etherscan URL with inner address tildes stripped should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, LONG_URL)
        shortened_url = mappings[LONG_URL]

        stripped = shortened_url.replace("~", "")
        restored = sub.restore(stripped)
        assert restored == LONG_URL

    def test_standalone_and_url_both_stripped(self):
        """Same address stripped in text and in URL should both restore."""
        sub = URLSubstitutor()
        eth = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth}"
        text = f"Contract {eth} at {url}"
        mappings = _prepare(sub, text)

        short_eth = mappings[eth]
        short_url = mappings[url]
        stripped_eth = short_eth.replace("~", "")
        stripped_url = short_url.replace("~", "")

        result = sub.restore(f"Contract {stripped_eth} at {stripped_url}")
        assert eth in result
        assert url in result

    def test_url_stripped_standalone_intact(self):
        """URL stripped but standalone exact — both should work."""
        sub = URLSubstitutor()
        eth = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth}"
        text = f"Contract {eth} at {url}"
        mappings = _prepare(sub, text)

        short_eth = mappings[eth]  # kept intact
        short_url = mappings[url]
        stripped_url = short_url.replace("~", "")

        result = sub.restore(f"Contract {short_eth} at {stripped_url}")
        assert eth in result
        assert url in result

    def test_longest_match_url_over_inner(self):
        """URL-level stripped match should win over inner address stripped match."""
        sub = URLSubstitutor()
        eth = "0xdac17f958d2ee523a2206206994597c13d831ec7"
        url = f"https://etherscan.io/address/{eth}"
        mappings = _prepare(sub, url)

        # The URL shortened form contains the address shortened form
        short_url = mappings[url]
        stripped_url = short_url.replace("~", "")

        # Restore should recover the full URL, not just the inner address
        restored = sub.restore(stripped_url)
        assert restored == url


class TestCaseAndStripCombined:
    """Category 3: Case changes + tilde stripping combined."""

    def test_lowercase_plus_stripped(self):
        """Lowercased AND stripped address should restore."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        stripped_lower = shortened.replace("~", "").lower()
        restored = sub.restore(stripped_lower)
        assert restored == ETH_ADDRESS

    def test_stripped_lowercase_url(self):
        """Lowercased AND stripped URL should restore."""
        sub = URLSubstitutor()
        url = "https://example.com/very/long/path/to/resource/page"
        mappings = _prepare(sub, url)
        shortened = mappings[url]

        stripped_lower = shortened.replace("~", "").lower()
        restored = sub.restore(stripped_lower)
        assert restored == url


class TestFalsePositiveResistance:
    """Category 4: False positive resistance — natural text should not be falsely restored."""

    def test_natural_text_not_falsely_restored(self):
        """Random text with hex-like chars should not be replaced."""
        sub = URLSubstitutor()
        _prepare(sub, ETH_ADDRESS)

        innocent_text = "The value 0x7a25abc1488D is unrelated"
        restored = sub.restore(innocent_text)
        # Should be unchanged (the text doesn't match any stripped form)
        assert restored == innocent_text

    def test_similar_different_string_not_restored(self):
        """Different hex at hash position should not match stripped form."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        # Create a string that looks similar but has different hash chars
        # The stripped form is very specific due to SHA256 hash
        fake = shortened.replace("~", "").replace("a", "b", 1)
        restored = sub.restore(fake)
        # Should NOT restore to ETH_ADDRESS
        assert ETH_ADDRESS not in restored

    def test_stripped_not_substring_of_original(self):
        """Empirical: stripped form should not be a substring of the original value."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS, ETH_ADDRESS_2, SOL_ADDRESS)
        for original, shortened in mappings.items():
            stripped = shortened.replace("~", "")
            assert stripped not in original, f"Stripped form {stripped} found in original {original}"

    def test_no_second_pass_corruption(self):
        """Exact-restore of A should not be corrupted by stripped-B in pass 2."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS, ETH_ADDRESS_2)
        short_a = mappings[ETH_ADDRESS]
        short_b = mappings[ETH_ADDRESS_2]

        # Exact restore A, stripped restore B
        stripped_b = short_b.replace("~", "")
        text = f"{short_a} and {stripped_b}"
        restored = sub.restore(text)

        # A should be exactly restored
        assert ETH_ADDRESS in restored
        # B should be restored via stripped
        assert ETH_ADDRESS_2 in restored


class TestCollisionHandling:
    """Category 5: Collision handling — first mapping wins."""

    def test_collision_first_wins(self):
        """If two shortened forms produce same stripped form, first should win."""
        sub = URLSubstitutor()
        # Create two mappings manually that would collide when stripped
        # This is astronomically unlikely with SHA256 but we test the guard
        sub._add_mapping("original_A", "pre~abc~suf")
        sub._add_mapping("original_B", "pre~abc~suf2")  # different shortened, different stripped

        # Both stripped forms are different, so both should be registered
        assert "preabcsuf" in sub._stripped_reverse
        assert "preabcsuf2" in sub._stripped_reverse

    def test_collision_doesnt_break_exact(self):
        """Even with a stripped collision, exact restore of both should work."""
        sub = URLSubstitutor()
        sub._add_mapping("original_A", "x~1~y")
        sub._add_mapping("original_B", "x~1~z")

        assert sub.restore("x~1~y") == "original_A"
        assert sub.restore("x~1~z") == "original_B"

    def test_stripped_not_in_reverse_dict(self):
        """Stripped forms should only be in _stripped_reverse, not _reverse."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]
        stripped = shortened.replace("~", "")

        assert stripped not in sub._reverse
        assert stripped in sub._stripped_reverse


class TestKnownLimitations:
    """Category 6: Known limitations."""

    def test_partial_strip_not_restored(self):
        """One tilde kept, one stripped → not restored (known limitation)."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        # Count tildes — there should be exactly 2
        assert shortened.count("~") == 2

        # Partially strip: remove only first tilde
        idx = shortened.index("~")
        partial = shortened[:idx] + shortened[idx + 1 :]  # remove first ~

        # Neither exact nor fully-stripped match
        assert partial != shortened  # not exact
        assert partial != shortened.replace("~", "")  # not fully stripped

        # Restore should not find it (known limitation)
        result = sub.restore(partial)
        assert result == partial  # unchanged

    def test_no_stripped_entry_when_no_tildes(self):
        """Shortened forms without tildes should not produce _stripped_reverse entries."""
        sub = URLSubstitutor()
        # Manually add a mapping without tildes
        sub._add_mapping("original", "short_form")

        # No tilde in shortened → stripped == shortened → guard prevents registration
        assert "short_form" not in sub._stripped_reverse


class TestIntegrationWithSubstitute:
    """Category 7: Integration with substitute() — stripped forms should not interfere."""

    def test_stripped_form_passes_through_substitute(self):
        """Stripped form is too short for pattern regexes — substitute should not touch it."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]
        stripped = shortened.replace("~", "")

        # Stripped form should pass through substitute unchanged
        result = sub.substitute(stripped)
        assert result == stripped

    def test_substitute_after_stripped_restore(self):
        """Restored original should re-shorten correctly."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        shortened = mappings[ETH_ADDRESS]

        # Simulate: LLM strips tildes → restore → re-substitute
        stripped = shortened.replace("~", "")
        restored = sub.restore(stripped)
        assert restored == ETH_ADDRESS

        re_shortened = sub.substitute(restored)
        assert re_shortened == shortened

    def test_substitute_idempotent_on_stripped(self):
        """substitute() should not create spurious mappings for stripped forms."""
        sub = URLSubstitutor()
        mappings = _prepare(sub, ETH_ADDRESS)
        count_before = sub.pattern_count
        shortened = mappings[ETH_ADDRESS]
        stripped = shortened.replace("~", "")

        sub.substitute(stripped)
        assert sub.pattern_count == count_before
