"""Post-response LLM output degeneration detection.

Detects token repetition loops where models output the same substring
thousands of times. Scans the tail of the response for any pattern
(length 1-100) that repeats consecutively beyond a threshold.

Consecutive-only detection: all documented degeneration produces
byte-identical consecutive repetitions. The model enters a token loop
and emits the same tokens back-to-back with no gaps or filler.
"""

_TAIL_CHARS: int = 8_000
_MIN_CONTENT_LENGTH: int = 200
_MAX_PATTERN_LENGTH: int = 100
_MIN_TOTAL_REPETITION_CHARS: int = 300
_MIN_REPETITION_COUNT: int = 10


def _repetition_threshold(pattern_length: int) -> int:
    """Minimum consecutive repetition count to classify as degeneration.

    Requires ~300 chars of consecutive repetition regardless of pattern length,
    with a floor of 10 repetitions for long patterns.

    Examples:
        L=1:  300 repetitions (300 chars) — no separator line is 300 chars
        L=3:  100 repetitions (300 chars) — "bau" x 100
        L=8:   38 repetitions (304 chars)
        L=30:  10 repetitions (300 chars)
        L=50:  10 repetitions (500 chars)
        L=100: 10 repetitions (1000 chars)
    """
    return max(_MIN_TOTAL_REPETITION_CHARS // pattern_length, _MIN_REPETITION_COUNT)


def detect_output_degeneration(content: str) -> str | None:
    """Detect LLM output degeneration in response text.

    Scans the tail of the response for any substring (length 1-100)
    that repeats consecutively enough times to indicate a generation loop.

    Returns an explanation string if degeneration is detected, None otherwise.

    Algorithm:
        For each pattern length L (1 to 100):
            Scan the tail for consecutive runs of the same L-char substring.
            If a run reaches the threshold, return explanation.

    Performance:
        O(100 * n) where n = min(len(content), 8000). Uses str.startswith()
        for in-place comparison — O(1) on first-byte mismatch. Typical
        execution: 20-50ms on non-degenerate text, ~2ms on degenerate text
        (early return at short pattern lengths).
    """
    if len(content) < _MIN_CONTENT_LENGTH:
        return None

    tail = content[-_TAIL_CHARS:] if len(content) > _TAIL_CHARS else content
    tail_len = len(tail)

    for pattern_len in range(1, _MAX_PATTERN_LENGTH + 1):
        if pattern_len > tail_len:
            break

        threshold = _repetition_threshold(pattern_len)
        # If the tail can't possibly contain enough repetitions, skip this length
        if tail_len < pattern_len * threshold:
            continue

        i = 0
        scan_end = tail_len - pattern_len
        while i <= scan_end:
            # Count consecutive repetitions of tail[i:i+pattern_len]
            pattern = tail[i : i + pattern_len]
            j = i + pattern_len
            while j + pattern_len <= tail_len and tail.startswith(pattern, j):
                j += pattern_len

            count = (j - i) // pattern_len
            if count >= threshold:
                total_chars = count * pattern_len
                if pattern_len <= 40:
                    display = repr(pattern)
                else:
                    display = repr(pattern[:40]) + "..."
                return f"substring {display} (length {pattern_len}) repeated {count} times ({total_chars} chars)"

            # Skip past the chain (if any), otherwise advance by 1
            i = j if count > 1 else i + 1

    return None
