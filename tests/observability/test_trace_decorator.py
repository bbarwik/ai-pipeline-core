"""Tests for trace decorator and tracing utilities."""

import asyncio
import inspect
import os
from unittest.mock import Mock, patch

import pytest

from ai_pipeline_core.observability.tracing import TraceInfo, set_trace_cost, trace


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

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_basic_sync_function(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test tracing a basic synchronous function."""
        mock_observe.return_value = lambda f: f  # Pass through

        @trace(level="debug")
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

        # Observe should have been called
        mock_observe.assert_called()

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    async def test_basic_async_function(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test tracing an async function."""
        mock_observe.return_value = lambda f: f  # Pass through

        @trace(level="debug")
        async def test_func(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        result = await test_func(5)
        assert result == 10

        mock_observe.assert_called()

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_custom_name(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test custom name for traced function."""
        mock_observe.return_value = lambda f: f

        @trace(level="debug", name="custom_operation")
        def test_func():
            return "result"

        test_func()

        # Check that observe was called with custom name
        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["name"] == "custom_operation"

    # Note: test parameter was removed from trace decorator

    @patch.dict(os.environ, {"LMNR_DEBUG": "false"})
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_debug_level_with_env_false(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test level='debug' with LMNR_DEBUG=false should NOT trace."""
        mock_observe.return_value = lambda f: f

        @trace(level="debug")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

        # Observe should NOT have been called (doesn't trace when LMNR_DEBUG != "true")
        mock_observe.assert_not_called()

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_debug_level_with_env_true(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test level='debug' with LMNR_DEBUG=true should trace."""
        mock_observe.return_value = lambda f: f

        @trace(level="debug")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

        # Observe SHOULD have been called (traces when LMNR_DEBUG is "true")
        mock_observe.assert_called()

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_ignore_flags(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test ignore input/output flags."""
        mock_observe.return_value = lambda f: f

        @trace(
            level="debug",
            ignore_input=True,
            ignore_output=True,
            ignore_inputs=["password", "secret"],
        )
        def test_func():
            return "result"

        test_func()

        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["ignore_input"] is True
        assert call_kwargs["ignore_output"] is True
        assert call_kwargs["ignore_inputs"] == ["password", "secret"]

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_formatters(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test input and output formatters."""
        mock_observe.return_value = lambda f: f

        def input_fmt(*args, **kwargs):
            return f"Input: {args}"

        def output_fmt(result):
            return f"Output: {result}"

        @trace(
            level="debug",
            input_formatter=input_fmt,
            output_formatter=output_fmt,
            trim_documents=False,
        )
        def test_func(x):
            return x * 2

        test_func(5)

        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["input_formatter"] == input_fmt
        assert call_kwargs["output_formatter"] == output_fmt

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_trace_info_injection(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test TraceInfo injection into function kwargs."""
        mock_observe.return_value = lambda f: f

        @trace(level="debug")
        def test_func(x: int, trace_info: TraceInfo) -> int:
            # Should receive a TraceInfo instance
            assert isinstance(trace_info, TraceInfo)
            return x * 2

        # Call without providing trace_info (decorator will inject it)
        result = test_func(5)  # type: ignore[call-arg]
        assert result == 10

    # Note: runtime test flag functionality was removed

    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
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

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_multiple_decorators(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test trace with multiple decorator syntaxes."""
        mock_observe.return_value = lambda f: f

        # Bare decorator
        @trace(level="debug")
        def func1():
            return "func1"

        # With parentheses but no args
        @trace(level="debug")
        def func2():
            return "func2"

        # With arguments
        @trace(level="debug", name="func3_custom")
        def func3():
            return "func3"

        assert func1() == "func1"
        assert func2() == "func2"
        assert func3() == "func3"

        assert mock_observe.call_count == 3

    @patch.dict(os.environ, {"LMNR_DEBUG": "true"})  # Enable debug tracing
    @patch("ai_pipeline_core.observability.tracing.observe")
    @patch("ai_pipeline_core.observability.tracing.Laminar.initialize")
    def test_trace_info_with_metadata(self, mock_init: Mock, mock_observe: Mock) -> None:
        """Test TraceInfo with custom metadata."""
        mock_observe.return_value = lambda f: f

        @trace(level="debug")
        def test_func(trace_info: TraceInfo):
            return "result"

        # Call with custom TraceInfo
        custom_info = TraceInfo(session_id="session-123", metadata={"workflow": "test"})
        test_func(trace_info=custom_info)

        call_kwargs = mock_observe.call_args[1]
        assert call_kwargs["session_id"] == "session-123"
        assert call_kwargs["metadata"] == {"workflow": "test"}


class TestSetTraceCost:
    """Test set_trace_cost function."""

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_positive(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with positive cost value."""
        set_trace_cost(0.05)

        mock_set_attrs.assert_called_once_with({
            "gen_ai.usage.output_cost": 0.05,
            "gen_ai.usage.cost": 0.05,
            "cost": 0.05,
        })

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_zero(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with zero cost (should not call Laminar)."""
        set_trace_cost(0.0)
        mock_set_attrs.assert_not_called()

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_negative(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with negative cost (should not call Laminar)."""
        set_trace_cost(-0.05)
        mock_set_attrs.assert_not_called()

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_usd_string_format(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with USD string format."""
        set_trace_cost("$0.50")

        mock_set_attrs.assert_called_once_with({
            "gen_ai.usage.output_cost": 0.50,
            "gen_ai.usage.cost": 0.50,
            "cost": 0.50,
        })

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_usd_string_with_spaces(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with USD string format with spaces."""
        set_trace_cost("  $1.25  ")

        mock_set_attrs.assert_called_once_with({
            "gen_ai.usage.output_cost": 1.25,
            "gen_ai.usage.cost": 1.25,
            "cost": 1.25,
        })

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_usd_string_zero(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with USD string format of zero (should not call Laminar)."""
        set_trace_cost("$0.00")
        mock_set_attrs.assert_not_called()

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_usd_string_negative(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with negative USD string (should not call Laminar)."""
        set_trace_cost("$-0.50")
        mock_set_attrs.assert_not_called()

    def test_set_trace_cost_invalid_string_no_dollar(self) -> None:
        """Test set_trace_cost with invalid string (no dollar sign)."""
        with pytest.raises(ValueError, match=r"Invalid USD format.*Must start with"):
            set_trace_cost("0.50")

    def test_set_trace_cost_invalid_string_not_number(self) -> None:
        """Test set_trace_cost with invalid string (not a number)."""
        with pytest.raises(ValueError, match=r"Invalid USD format.*Must be a valid number"):
            set_trace_cost("$abc")

    def test_set_trace_cost_invalid_string_empty_after_dollar(self) -> None:
        """Test set_trace_cost with invalid string (empty after dollar)."""
        with pytest.raises(ValueError, match=r"Invalid USD format.*Must be a valid number"):
            set_trace_cost("$")

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_multiple_calls(self, mock_set_attrs: Mock) -> None:
        """Test multiple calls to set_trace_cost within the same function."""

        @trace
        def traced_func():
            # First cost
            set_trace_cost(0.01)
            # Update cost
            set_trace_cost("$0.02")
            # Final cost
            set_trace_cost(0.03)
            return "result"

        traced_func()

        # Verify all three calls were made
        assert mock_set_attrs.call_count == 3

        # Check each call
        calls = mock_set_attrs.call_args_list
        assert calls[0][0][0] == {
            "gen_ai.usage.output_cost": 0.01,
            "gen_ai.usage.cost": 0.01,
            "cost": 0.01,
        }
        assert calls[1][0][0] == {
            "gen_ai.usage.output_cost": 0.02,
            "gen_ai.usage.cost": 0.02,
            "cost": 0.02,
        }
        assert calls[2][0][0] == {
            "gen_ai.usage.output_cost": 0.03,
            "gen_ai.usage.cost": 0.03,
            "cost": 0.03,
        }

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    def test_set_trace_cost_within_traced_function(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost called within a traced function."""

        @trace
        def traced_func():
            # Calculate dynamic cost
            cost = 0.001 * 10  # Some calculation
            set_trace_cost(cost)
            return "result"

        # Note: In real usage, this would be within a Laminar span context
        # For testing, we just verify the call was made
        traced_func()

        mock_set_attrs.assert_called_with({
            "gen_ai.usage.output_cost": 0.01,
            "gen_ai.usage.cost": 0.01,
            "cost": 0.01,
        })

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    async def test_set_trace_cost_in_async_function(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost in async traced function."""

        @trace
        async def async_traced_func(items: list[int]) -> int:
            # Dynamic cost based on input
            cost_per_item = 0.002
            total_cost = len(items) * cost_per_item
            set_trace_cost(total_cost)
            await asyncio.sleep(0.001)
            return len(items)

        result = await async_traced_func([1, 2, 3])
        assert result == 3

        mock_set_attrs.assert_called_with({
            "gen_ai.usage.output_cost": 0.006,
            "gen_ai.usage.cost": 0.006,
            "cost": 0.006,
        })

    @patch("ai_pipeline_core.observability.tracing.Laminar.set_span_attributes")
    async def test_set_trace_cost_with_usd_string_in_async(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost with USD string in async traced function."""

        @trace
        async def async_traced_func(amount: float) -> str:
            # Format cost as USD string
            cost_str = f"${amount:.2f}"
            set_trace_cost(cost_str)
            await asyncio.sleep(0.001)
            return cost_str

        result = await async_traced_func(3.456)
        assert result == "$3.46"

        mock_set_attrs.assert_called_with({
            "gen_ai.usage.output_cost": 3.46,
            "gen_ai.usage.cost": 3.46,
            "cost": 3.46,
        })

    @patch(
        "ai_pipeline_core.observability.tracing.Laminar.set_span_attributes",
        side_effect=Exception("Not in traced context"),
    )
    def test_set_trace_cost_outside_context_no_error(self, mock_set_attrs: Mock) -> None:
        """Test set_trace_cost silently handles exception when not in traced context."""
        # Should NOT raise an exception - silently ignores the error
        set_trace_cost(0.05)  # This should not raise

        # Verify it tried to call
        mock_set_attrs.assert_called_once()
