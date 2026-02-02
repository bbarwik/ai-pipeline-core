"""Tests for tracking recursion prevention."""

from ai_pipeline_core.observability._tracking._internal import (
    internal_tracking_context,
    is_internal_tracking,
)


class TestInternalTrackingContext:
    """Test thread-local recursion flag."""

    def test_default_is_false(self):
        assert is_internal_tracking() is False

    def test_context_sets_flag(self):
        with internal_tracking_context():
            assert is_internal_tracking() is True
        assert is_internal_tracking() is False

    def test_nested_contexts(self):
        with internal_tracking_context():
            assert is_internal_tracking() is True
            with internal_tracking_context():
                assert is_internal_tracking() is True
            # Outer context still active after inner exits
            assert is_internal_tracking() is True
        assert is_internal_tracking() is False

    def test_exception_resets_flag(self):
        try:
            with internal_tracking_context():
                raise ValueError("test")
        except ValueError:
            pass
        assert is_internal_tracking() is False
