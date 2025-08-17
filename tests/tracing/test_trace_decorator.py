"""Tests for trace decorator."""

import asyncio
import inspect
import os
from unittest.mock import Mock, patch

from ai_pipeline_core.tracing import TraceInfo, trace


class TestTraceInfo:
    """Test TraceInfo metadata container."""

    def test_default_values(self):
        """Test default TraceInfo values."""
        info = TraceInfo()
        assert info.session_id is None
        assert info.user_id is None
        assert info.metadata == {}
        assert info.tags == []

    def test_get_observe_kwargs_empty(self):
        """Test get_observe_kwargs with empty TraceInfo."""
        info = TraceInfo()
        kwargs = info.get_observe_kwargs()
        assert kwargs == {}

    def test_get_observe_kwargs_with_values(self):
        """Test get_observe_kwargs with populated values."""
        info = TraceInfo(
            session_id="session-123",
            user_id="user-456",
            metadata={"key": "value"},
            tags=["test", "production"],
        )
        kwargs = info.get_observe_kwargs()

        assert kwargs["session_id"] == "session-123"
        assert kwargs["user_id"] == "user-456"
        assert kwargs["metadata"] == {"key": "value"}
        assert kwargs["tags"] == ["test", "production"]

    @patch.dict(os.environ, {"LMNR_SESSION_ID": "env-session", "LMNR_USER_ID": "env-user"})
    def test_get_observe_kwargs_env_fallback(self):
        """Test environment variable fallback."""
        info = TraceInfo()
        kwargs = info.get_observe_kwargs()

        assert kwargs["session_id"] == "env-session"
        assert kwargs["user_id"] == "env-user"

    @patch.dict(os.environ, {"LMNR_SESSION_ID": "env-session"})
    def test_get_observe_kwargs_explicit_overrides_env(self):
        """Test that explicit values override environment variables."""
        info = TraceInfo(session_id="explicit-session")
        kwargs = info.get_observe_kwargs()

        assert kwargs["session_id"] == "explicit-session"  # Not env-session


class TestTraceDecorator:
    """Test trace decorator functionality."""

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_basic_sync_function(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test tracing a basic synchronous function."""
        mock_observe.return_value = lambda f: f  # Pass through

        @trace
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

        # Observe should have been called
        mock_observe.assert_called()

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    async def test_basic_async_function(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test tracing an async function."""
        mock_observe.return_value = lambda f: f  # Pass through

        @trace
        async def test_func(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        result = await test_func(5)
        assert result == 10

        mock_observe.assert_called()

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_custom_name(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test custom name for traced function."""
        mock_observe.return_value = lambda f: f

        @trace(name="custom_operation")
        def test_func():
            return "result"

        test_func()

        # Check that observe was called with custom name
        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["name"] == "custom_operation"

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_test_flag_adds_tag(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test that test=True adds test tag."""
        mock_observe.return_value = lambda f: f

        @trace(test=True)
        def test_func():
            return "result"

        test_func()

        call_kwargs = mock_observe.call_args[1]
        assert "tags" in call_kwargs
        assert "test" in call_kwargs["tags"]

    @patch.dict(os.environ, {"LMNR_DEBUG": "false"})
    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_debug_only_without_env(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test debug_only=True without LMNR_DEBUG env var."""

        @trace(debug_only=True)
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

        # Observe should NOT have been called
        mock_observe.assert_not_called()

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})
    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_debug_only_with_env(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test debug_only=True with LMNR_DEBUG=true."""
        mock_observe.return_value = lambda f: f

        @trace(debug_only=True)
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

        # Observe SHOULD have been called
        mock_observe.assert_called()

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_ignore_flags(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test ignore input/output flags."""
        mock_observe.return_value = lambda f: f

        @trace(ignore_input=True, ignore_output=True, ignore_inputs=["password", "secret"])
        def test_func():
            return "result"

        test_func()

        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["ignore_input"] is True
        assert call_kwargs["ignore_output"] is True
        assert call_kwargs["ignore_inputs"] == ["password", "secret"]

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_formatters(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test input and output formatters."""
        mock_observe.return_value = lambda f: f

        def input_fmt(*args, **kwargs):
            return f"Input: {args}"

        def output_fmt(result):
            return f"Output: {result}"

        @trace(input_formatter=input_fmt, output_formatter=output_fmt)
        def test_func(x):
            return x * 2

        test_func(5)

        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["input_formatter"] == input_fmt
        assert call_kwargs["output_formatter"] == output_fmt

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_trace_info_injection(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test TraceInfo injection into function kwargs."""
        mock_observe.return_value = lambda f: f

        @trace
        def test_func(x: int, trace_info: TraceInfo) -> int:
            # Should receive a TraceInfo instance
            assert isinstance(trace_info, TraceInfo)
            return x * 2

        # Call without providing trace_info (decorator will inject it)
        result = test_func(5)  # type: ignore[call-arg]
        assert result == 10

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_runtime_test_flag(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test runtime test flag in kwargs."""
        mock_observe.return_value = lambda f: f

        @trace
        def test_func(x: int, test: bool = False) -> int:
            return x * 2

        # Call with test=True
        test_func(5, test=True)

        call_kwargs = mock_observe.call_args[1]
        assert "tags" in call_kwargs
        assert "test" in call_kwargs["tags"]

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_signature_preservation(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test that function signature is preserved."""
        mock_observe.return_value = lambda f: f

        def original_func(x: int, y: str = "default") -> str:
            return f"{x}-{y}"

        traced_func = trace(original_func)

        # Check signature is preserved
        original_sig = inspect.signature(original_func)
        traced_sig = inspect.signature(traced_func)

        assert str(original_sig) == str(traced_sig)

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_multiple_decorators(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test trace with multiple decorator syntaxes."""
        mock_observe.return_value = lambda f: f

        # Bare decorator
        @trace
        def func1():
            return "func1"

        # With parentheses but no args
        @trace()
        def func2():
            return "func2"

        # With arguments
        @trace(name="func3_custom")
        def func3():
            return "func3"

        assert func1() == "func1"
        assert func2() == "func2"
        assert func3() == "func3"

        assert mock_observe.call_count == 3

    @patch("ai_pipeline_core.tracing.observe")
    @patch("ai_pipeline_core.tracing.Laminar.initialize")
    def test_trace_info_with_metadata(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test TraceInfo with custom metadata."""
        mock_observe.return_value = lambda f: f

        @trace
        def test_func(trace_info: TraceInfo):
            return "result"

        # Call with custom TraceInfo
        custom_info = TraceInfo(session_id="session-123", metadata={"workflow": "test"})
        test_func(trace_info=custom_info)

        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["session_id"] == "session-123"
        assert call_kwargs["metadata"] == {"workflow": "test"}
