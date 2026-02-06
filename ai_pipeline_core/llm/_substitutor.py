"""Content shortener for URLs and high-entropy strings.

Reduces LLM token usage by shortening long URLs and blockchain identifiers.
Uses `~` as truncation marker. Round-trip: substitute() → LLM → restore().

Generic entropy-based detection works across all chains without chain-specific patterns.
Zero-width characters are normalized for detection but preserved for exact round-trip.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Zero-width and invisible characters (normalized for detection, preserved in output)
_INVISIBLE_CHARS = frozenset("\u200b\u200c\u200d\ufeff\u2060\u180e")

# Generic detection patterns (order: more specific first)
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 0x-prefixed hex (ETH, Sui, Aptos, etc.) - 40+ hex chars after 0x
    ("hex_prefixed", re.compile(r"\b0x[a-fA-F0-9]{40,}\b")),
    # High-entropy alphanumeric (Solana, Stellar, Polkadot, Tron, Cardano, Cosmos, Base58, Base32)
    ("high_entropy", re.compile(r"\b[A-Za-z0-9]{26,}\b")),
    # Base64 with special chars (explicit +/= markers)
    ("base64", re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b")),
    # Plain hex string without 0x prefix (32+ chars for SHA256 etc)
    ("hex", re.compile(r"\b[a-fA-F0-9]{32,}\b")),
    # Relative paths inside delimiters (markdown links, HTML attributes, XML tags)
    ("path", re.compile(r"/[a-zA-Z0-9_\-\./]+")),
    # URLs
    ("url", re.compile(r"https?://[^\s<>\"'`\[\]{}|\\^]+", re.IGNORECASE)),
]

# Config per pattern: (prefix_len, suffix_len, min_entropy, min_diversity)
_CONFIG: dict[str, tuple[int, int, float, int]] = {
    "hex_prefixed": (6, 4, 3.0, 8),
    "high_entropy": (4, 4, 3.5, 12),
    "base64": (6, 4, 3.5, 10),
    "hex": (6, 4, 3.0, 8),
}

# JWT protection (base64-like but should not be shortened)
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")

# Code context keywords (false positive prevention)
_CODE_KEYWORDS = ("install ", "npm ", "yarn ", "pnpm ", "pip ", "require(", "import ")

# Delimiter pairs for path shortening validation
_DELIMITER_PAIRS: dict[str, str] = {"(": ")", "[": "]", '"': '"', "'": "'", ">": "<"}

# Content-addressed URL schemes (hash IS the identifier - must not be shortened)
_CONTENT_ADDRESSED_SCHEMES = ("ipfs://", "ipns://", "magnet:", "data:")

# File extensions to preserve in shortened URLs
FILE_EXTENSIONS = frozenset({
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".csv",
    ".txt",
    ".md",
    ".zip",
    ".tar",
    ".gz",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
})


def _normalize(s: str) -> str:
    """Remove invisible chars for pattern matching (not from source text)."""
    if not any(c in s for c in _INVISIBLE_CHARS):
        return s
    return "".join(c for c in s if c not in _INVISIBLE_CHARS)


def _build_normalized_view(text: str) -> tuple[str, list[int] | None]:
    """Build normalized text and index map for pattern detection.

    Returns (normalized_text, index_map) where index_map[norm_idx] = orig_idx.
    If no invisible chars, returns (text, None) for fast path.
    """
    if not any(c in text for c in _INVISIBLE_CHARS):
        return text, None

    chars: list[str] = []
    mapping: list[int] = []

    for i, c in enumerate(text):
        if c not in _INVISIBLE_CHARS:
            chars.append(c)
            mapping.append(i)

    # Add sentinel for end-of-string mapping
    mapping.append(len(text))

    return "".join(chars), mapping


def _map_span_to_original(index_map: list[int] | None, start: int, end: int) -> tuple[int, int]:
    """Map normalized span back to original text positions."""
    if index_map is None:
        return start, end
    return index_map[start], index_map[end]


def _entropy(s: str) -> float:
    """Shannon entropy - high values indicate randomness."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _trim_url(url: str) -> str:
    """Trim trailing punctuation from URL, preserving balanced brackets."""
    while url and url[-1] in ".,;:!?)]}'\"'`":
        if url[-1] == ")" and url.count("(") >= url.count(")"):
            break
        if url[-1] == "]" and url.count("[") >= url.count("]"):
            break
        url = url[:-1]
    return url


def _rstrip_to_alnum(s: str) -> str:
    """Strip trailing non-alphanumeric to avoid ugly junctions like -~ or =~."""
    i = len(s)
    while i > 0 and not s[i - 1].isalnum():
        i -= 1
    return s[:i]


def _is_in_url(text: str, pos: int) -> bool:
    """Check if position is inside a URL."""
    before = text[max(0, pos - 200) : pos]
    if "://" not in before:
        return False
    last = max(before.rfind("http://"), before.rfind("https://"))
    return last != -1 and not any(c in before[last:] for c in " \t\n\r")


def _is_in_content_addressed_url(context_before: str) -> bool:
    """Check if position is inside a content-addressed URL (ipfs://, data:, etc.)."""
    lower = context_before.lower()
    for scheme in _CONTENT_ADDRESSED_SCHEMES:
        if scheme in lower:
            # Check if scheme is the most recent URL-like thing
            last_scheme_pos = lower.rfind(scheme)
            # No whitespace between scheme and current position
            after_scheme = context_before[last_scheme_pos:]
            if not any(c in after_scheme for c in " \t\n\r"):
                return True
    return False


def _is_valid_path(value: str, context_before: str, context_after: str) -> bool:
    """Validate a path pattern: must be long, multi-segment, and inside delimiters."""
    if value.startswith("//") or len(value) < 30 or value.count("/") < 3:
        return False
    ctx = _normalize(context_before)
    ctx_a = _normalize(context_after)
    if not ctx or not ctx_a:
        return False
    return ctx[-1] in _DELIMITER_PAIRS and _DELIMITER_PAIRS[ctx[-1]] == ctx_a[0]


def _is_valid_high_entropy(value: str, ctx_before: str, ctx_after: str) -> bool:
    """Extra validation for high-entropy alphanumeric patterns."""
    if ctx_before.rstrip().endswith("@"):
        return False
    if any(kw in ctx_before.lower() for kw in _CODE_KEYWORDS):
        return False
    if ctx_before.endswith(".") or ctx_after.startswith("."):
        return False
    return not (value.isalpha() and len(value) < 40)


def _is_valid_base64(value: str, ctx_before: str, ctx_after: str) -> bool:
    """Extra validation for base64 patterns."""
    if ctx_before.endswith(".") or ctx_after.startswith("."):
        return False
    # URL paths match base64 charset because '/' is valid Base64.
    # Real Base64 with '/' also contains '+' or '=' padding — URL paths never do.
    if "/" in value and "+" not in value and not value.endswith("="):
        return False
    has_base64_chars = "+" in value or "/" in value or value.endswith("=")
    return has_base64_chars or len(value) >= 64


def _is_valid_pattern(normalized_value: str, kind: str, context_before: str, context_after: str) -> bool:
    """Validate pattern using entropy and diversity thresholds on normalized value."""
    if _is_in_content_addressed_url(context_before):
        return False

    if kind == "path":
        return _is_valid_path(normalized_value, context_before, context_after)

    if kind not in _CONFIG:
        return True

    _, _, min_entropy, min_diversity = _CONFIG[kind]
    if _entropy(normalized_value) < min_entropy or len(set(normalized_value)) < min_diversity:
        return False

    ctx_before = _normalize(context_before)
    ctx_after = _normalize(context_after)

    if kind == "high_entropy":
        return _is_valid_high_entropy(normalized_value, ctx_before, ctx_after)
    return kind != "base64" or _is_valid_base64(normalized_value, ctx_before, ctx_after)


@dataclass(slots=True)
class URLSubstitutor:
    """Bidirectional URL/address shortener for LLM context.

    Uses entropy-based detection to work across all blockchain chains
    without chain-specific patterns. Zero-width characters are preserved
    for exact round-trip while still being detected in patterns.

    Usage:
        >>> sub = URLSubstitutor()
        >>> await sub.prepare(["Check https://example.com/long/path"])
        >>> shortened = sub.substitute("Check https://example.com/long/path")
        >>> original = sub.restore(shortened)
    """

    _forward: dict[str, str] = field(default_factory=dict)
    _reverse: dict[str, str] = field(default_factory=dict)
    _hashes: dict[str, str] = field(default_factory=dict)
    _prepared: bool = False

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    @property
    def pattern_count(self) -> int:
        return len(self._forward)

    def get_mappings(self) -> dict[str, str]:
        """Return a copy of all original-to-shortened mappings."""
        return dict(self._forward)

    def _add_mapping(self, original: str, shortened: str) -> None:
        """Add bidirectional mapping with lowercase variant for LLM resilience."""
        self._forward[original] = shortened
        self._reverse[shortened] = original
        # Add lowercase variant for LLM that lowercase output
        lower = shortened.lower()
        if lower != shortened and lower not in self._reverse:
            self._reverse[lower] = original

    def _generate_hash(self, value: str, start_length: int = 3) -> str:
        """Generate unique hash suffix, extending on collision."""
        # Use normalized value for consistent hashing
        normalized = _normalize(value)
        full = hashlib.sha256(normalized.encode()).hexdigest()
        for length in range(start_length, len(full) + 1):
            h = full[:length]
            if h not in self._hashes or self._hashes[h] == normalized:
                self._hashes[h] = normalized
                return h
        return full  # Full hash as fallback (never truncated)

    def _shorten_url(self, original: str) -> str:
        """Shorten URL, reusing pattern shortenings for embedded high-entropy strings.

        If the URL contains patterns that would be shortened on their own (e.g., ETH addresses),
        those patterns are shortened first, preserving the URL structure. Only if no patterns
        are found does the method fall back to hash-based URL shortening.
        """
        if original in self._forward:
            return self._forward[original]

        normalized = _normalize(original)
        if len(normalized) < 40:
            return original

        # Phase 1: Try to shorten embedded patterns within the URL
        modified = self._shorten_patterns_in_url(original)
        if modified != original and len(modified) <= 100:
            self._add_mapping(original, modified)
            return modified

        # Phase 2: Fall back to hash-based URL shortening (no patterns found or still too long)
        return self._shorten_url_with_hash(original, normalized)

    def _shorten_patterns_in_url(self, url: str) -> str:
        """Find and shorten high-entropy patterns within a URL.

        Returns the URL with patterns replaced by their shortened forms,
        or the original URL if no patterns were found.
        """
        normalized = _normalize(url)
        result = url
        found_any = False

        # Check each non-URL, non-path pattern
        for kind, pattern in _PATTERNS:
            if kind in {"url", "path"}:
                continue

            for m in pattern.finditer(normalized):
                match_text = m.group()

                # Validate the pattern
                start, end = m.start(), m.end()
                ctx_before = normalized[max(0, start - 30) : start]
                ctx_after = normalized[end : min(len(normalized), end + 30)]

                if not _is_valid_pattern(match_text, kind, ctx_before, ctx_after):
                    continue

                # Get or create shortened form
                if match_text in self._forward:
                    short = self._forward[match_text]
                else:
                    short = self._shorten_string(match_text, kind)

                if short != match_text:
                    result = result.replace(match_text, short)
                    found_any = True

        return result if found_any else url

    def _shorten_url_with_hash(self, original: str, normalized: str) -> str:
        """Shorten URL using hash-based approach (fallback when no patterns found)."""
        parsed = urlparse(normalized)
        domain = parsed.netloc.removeprefix("www.")
        parts = [p for p in parsed.path.split("/") if p]

        segment = _rstrip_to_alnum(parts[0][:10]) if parts else ""

        ext = ""
        if parts and "." in parts[-1]:
            e = parts[-1][parts[-1].rfind(".") :]
            if e.lower() in FILE_EXTENSIONS:
                ext = e

        h = self._generate_hash(original, 4)
        scheme = f"{parsed.scheme}://"

        if segment:
            short = f"{scheme}{domain}/{segment}~{h}{ext}"
        else:
            short = f"{scheme}{domain}~{h}"

        self._add_mapping(original, short)
        return short

    def _shorten_path(self, original: str) -> str:
        """Shorten relative path to /first/~hash/last format."""
        if original in self._forward:
            return self._forward[original]

        normalized = _normalize(original)
        parts = [p for p in normalized.split("/") if p]

        if len(parts) < 2:
            return original

        first = parts[0]
        last = parts[-1]
        h = self._generate_hash(original, 4)
        short = f"/{first}/~{h}/{last}"
        self._add_mapping(original, short)
        return short

    def _shorten_string(self, original: str, kind: str) -> str:
        """Shorten high-entropy string to prefix~hash~suffix."""
        if original in self._forward:
            return self._forward[original]

        normalized = _normalize(original)
        if len(normalized) < 20:
            return original

        prefix_len, suffix_len = 4, 4
        if kind in _CONFIG:
            prefix_len, suffix_len = _CONFIG[kind][0], _CONFIG[kind][1]

        h = self._generate_hash(original)
        # Use normalized value for prefix/suffix (clean visible output)
        short = f"{normalized[:prefix_len]}~{h}~{normalized[-suffix_len:]}"
        self._add_mapping(original, short)
        return short

    def prepare(self, texts: Sequence[str]) -> None:
        """Pre-process texts to build substitution mappings."""
        for text in texts:
            if text:
                self._scan(text)
        self._prepared = True

    def _scan(self, text: str) -> None:
        """Scan text and build mappings (preserves invisible chars in originals)."""
        # Build normalized view for pattern detection
        normalized, index_map = _build_normalized_view(text)

        # Find JWT spans in normalized text
        jwt_spans = [(m.start(), m.end()) for m in _JWT_PATTERN.finditer(normalized)]

        for kind, pattern in _PATTERNS:
            for m in pattern.finditer(normalized):
                norm_value = m.group()
                norm_start, norm_end = m.start(), m.end()

                # Trim URL in normalized space
                if kind == "url":
                    norm_value = _trim_url(norm_value)
                    norm_end = norm_start + len(norm_value)

                # Map back to original positions
                orig_start, orig_end = _map_span_to_original(index_map, norm_start, norm_end)
                original_value = text[orig_start:orig_end]

                if original_value in self._forward or norm_value in self._reverse:
                    continue

                if kind == "url":
                    self._shorten_url(original_value)
                    continue

                # Skip patterns inside URLs (check in normalized text)
                if _is_in_url(normalized, norm_start):
                    continue

                # Skip inside JWT tokens
                if any(s <= norm_start < e for s, e in jwt_spans):
                    continue

                # Validate using normalized value and context
                ctx_before = text[max(0, orig_start - 30) : orig_start]
                ctx_after = text[orig_end : min(len(text), orig_end + 30)]
                if not _is_valid_pattern(norm_value, kind, ctx_before, ctx_after):
                    continue

                if kind == "path":
                    self._shorten_path(original_value)
                else:
                    self._shorten_string(original_value, kind)

    def _ensure_mapping(self, original_value: str, norm_value: str, kind: str, ctx_before: str, ctx_after: str) -> None:
        """Create shortened mapping for a pattern if one doesn't exist yet."""
        if original_value in self._forward:
            return
        if kind == "url":
            self._shorten_url(original_value)
            return
        if not _is_valid_pattern(norm_value, kind, ctx_before, ctx_after):
            return
        if kind == "path":
            self._shorten_path(original_value)
        else:
            self._shorten_string(original_value, kind)

    def substitute(self, text: str) -> str:
        """Replace patterns with shortened forms (preserves unmatched text exactly)."""
        if not text:
            return text

        normalized, index_map = _build_normalized_view(text)
        jwt_spans = [(m.start(), m.end()) for m in _JWT_PATTERN.finditer(normalized)]
        replacements: list[tuple[int, int, str]] = []

        for kind, pattern in _PATTERNS:
            for m in pattern.finditer(normalized):
                norm_value = m.group()
                norm_start, norm_end = m.start(), m.end()

                if kind == "url":
                    norm_value = _trim_url(norm_value)
                    norm_end = norm_start + len(norm_value)

                orig_start, orig_end = _map_span_to_original(index_map, norm_start, norm_end)
                original_value = text[orig_start:orig_end]

                if norm_value in self._reverse:
                    continue
                if any(s <= norm_start < e for s, e in jwt_spans):
                    continue
                if kind != "url" and _is_in_url(normalized, norm_start):
                    continue

                ctx_before = text[max(0, orig_start - 30) : orig_start]
                ctx_after = text[orig_end : min(len(text), orig_end + 30)]
                self._ensure_mapping(original_value, norm_value, kind, ctx_before, ctx_after)

                if original_value in self._forward and self._forward[original_value] != original_value:
                    replacements.append((orig_start, orig_end, self._forward[original_value]))

        return self._apply_replacements(text, replacements)

    @staticmethod
    def _apply_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
        """Apply non-overlapping replacements to text, longest-first at each position."""
        if not replacements:
            return text

        replacements.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        filtered = []
        last_end = 0
        for start, end, short in replacements:
            if start >= last_end:
                filtered.append((start, end, short))
                last_end = end

        result = text
        for start, end, short in reversed(filtered):
            result = result[:start] + short + result[end:]
        return result

    def restore(self, text: str) -> str:
        """Restore shortened forms to originals."""
        if not text or not self._reverse:
            return text

        result = text

        # Exact match (longest first to avoid partial matches)
        for short in sorted(self._reverse, key=len, reverse=True):
            if short in result:
                result = result.replace(short, self._reverse[short])

        return result
