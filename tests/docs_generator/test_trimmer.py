from ai_pipeline_core.docs_generator.extractor import ClassInfo
from ai_pipeline_core.docs_generator.guide_builder import GuideData, render_guide
from ai_pipeline_core.docs_generator.trimmer import (
    MAX_GUIDE_SIZE,
    _measure,
    manage_guide_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_guide() -> GuideData:
    """Minimal guide that renders small."""
    cls = ClassInfo(
        name="Small",
        bases=(),
        docstring="Tiny.",
        is_public=True,
        class_vars=(),
        methods=(),
        validators=(),
        module_path="mod",
    )
    return GuideData(
        module_name="small",
        classes=[cls],
        functions=[],
        rules=[],
        external_bases=set(),
        normal_examples=[],
        error_examples=[],
    )


# ---------------------------------------------------------------------------
# manage_guide_size
# ---------------------------------------------------------------------------


def test_manage_guide_size_under_limit_returns_content():
    data = _small_guide()
    content = render_guide(data)
    result = manage_guide_size(data, content, max_size=100_000)
    assert result == content


def test_manage_guide_size_uses_default_limit():
    data = _small_guide()
    content = render_guide(data)
    # Should not warn -- small guide is well under 51,200 bytes
    result = manage_guide_size(data, content)
    assert result == content


def test_manage_guide_size_over_limit_returns_content_unchanged():
    data = _small_guide()
    content = render_guide(data)
    result = manage_guide_size(data, content, max_size=1)
    assert result == content


def test_manage_guide_size_at_exact_limit():
    data = _small_guide()
    content = render_guide(data)
    exact = _measure(content)
    result = manage_guide_size(data, content, max_size=exact)
    assert result == content


def test_manage_guide_size_over_limit_logs_warning(caplog):
    data = _small_guide()
    content = render_guide(data)
    with caplog.at_level("WARNING"):
        manage_guide_size(data, content, max_size=1)
    assert "small" in caplog.text


def test_manage_guide_size_under_limit_no_warning(caplog):
    data = _small_guide()
    content = render_guide(data)
    with caplog.at_level("WARNING"):
        manage_guide_size(data, content, max_size=100_000)
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# _measure
# ---------------------------------------------------------------------------


def test_measure_ascii():
    assert _measure("hello") == 5


def test_measure_unicode_multibyte():
    # Unicode character U+1F600 is 4 bytes in UTF-8
    assert _measure("\U0001f600") == 4


def test_measure_mixed():
    # 'a' = 1 byte, emoji = 4 bytes, 'b' = 1 byte
    assert _measure("a\U0001f600b") == 6


def test_measure_empty():
    assert _measure("") == 0


def test_max_guide_size_constant():
    assert MAX_GUIDE_SIZE == 51_200
