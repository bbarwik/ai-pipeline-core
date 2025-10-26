"""Comprehensive tests for remote_deployment decorator."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ai_pipeline_core import DocumentList, FlowDocument
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.utils.remote_deployment import remote_deployment


class OutputDoc(FlowDocument):
    """Output document for testing."""

    pass


class TestRemoteDeploymentDecorator:
    """Test the remote_deployment decorator functionality."""

    async def test_basic_remote_deployment(self):
        """Test @remote_deployment with basic usage."""

        @remote_deployment(output_document_type=OutputDoc)
        async def my_remote_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass  # Body is never executed  # type: ignore[return]

        # Mock the remote deployment execution
        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            # Mock return value - list of dicts representing documents
            mock_run.return_value = [{"name": "output.txt", "content": b"result", "sources": []}]

            docs = DocumentList([])
            options = FlowOptions()
            result = await my_remote_flow("test-project", docs, options)

            # Verify result
            assert isinstance(result, DocumentList)
            assert len(result) == 1
            assert isinstance(result[0], OutputDoc)
            assert result[0].name == "output.txt"
            assert result[0].content == b"result"

            # Verify deployment was called with correct parameters
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1]["deployment_name"] == "my-remote-flow/my_remote_flow"
            assert call_args[1]["parameters"]["project_name"] == "test-project"

    async def test_document_list_serialization(self):
        """Test that DocumentList parameters are serialized correctly."""

        class InputDoc(FlowDocument):
            pass

        @remote_deployment(output_document_type=OutputDoc)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = [{"name": "output.txt", "content": b"result", "sources": []}]

            # Create input documents
            input_docs = DocumentList([
                InputDoc(name="input1.txt", content=b"content1"),
                InputDoc(name="input2.txt", content=b"content2"),
            ])
            options = FlowOptions()

            await my_flow("project", input_docs, options)

            # Verify documents were serialized to list
            call_args = mock_run.call_args
            serialized_docs = call_args[1]["parameters"]["documents"]

            assert isinstance(serialized_docs, list)
            assert len(serialized_docs) == 2
            # Documents are serialized via iteration
            assert serialized_docs[0].name == "input1.txt"
            assert serialized_docs[1].name == "input2.txt"

    async def test_deployment_name_derivation(self):
        """Test that deployment name is derived correctly from function name."""

        @remote_deployment(output_document_type=OutputDoc)
        async def my_complex_flow_name(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = [{"name": "output.txt", "content": b"result", "sources": []}]

            docs = DocumentList([])
            options = FlowOptions()
            await my_complex_flow_name("project", docs, options)

            # Verify underscores are converted to hyphens
            call_args = mock_run.call_args
            deployment_name = call_args[1]["deployment_name"]
            assert deployment_name == "my-complex-flow-name/my_complex_flow_name"

    async def test_trace_level_off(self):
        """Test @remote_deployment with trace_level='off'."""

        @remote_deployment(output_document_type=OutputDoc, trace_level="off")
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = [{"name": "output.txt", "content": b"result", "sources": []}]

            docs = DocumentList([])
            options = FlowOptions()
            result = await my_flow("project", docs, options)

            assert isinstance(result, DocumentList)
            assert len(result) == 1

    async def test_trace_cost(self):
        """Test @remote_deployment with trace_cost parameter."""

        with patch("ai_pipeline_core.utils.remote_deployment.set_trace_cost") as mock_set_cost:

            @remote_deployment(output_document_type=OutputDoc, trace_cost=0.05)
            async def my_flow(
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:  # pyright: ignore[reportReturnType]
                pass

            with patch(
                "ai_pipeline_core.utils.remote_deployment.run_remote_deployment"
            ) as mock_run:
                mock_run.return_value = [
                    {"name": "output.txt", "content": b"result", "sources": []}
                ]

                docs = DocumentList([])
                options = FlowOptions()
                result = await my_flow("project", docs, options)

                assert isinstance(result, DocumentList)
                # Verify set_trace_cost was called with correct value
                mock_set_cost.assert_called_once_with(0.05)

    async def test_trace_cost_zero_not_called(self):
        """Test that trace_cost=0 doesn't call set_trace_cost."""

        with patch("ai_pipeline_core.utils.remote_deployment.set_trace_cost") as mock_set_cost:

            @remote_deployment(output_document_type=OutputDoc, trace_cost=0)
            async def my_flow(
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:  # pyright: ignore[reportReturnType]
                pass

            with patch(
                "ai_pipeline_core.utils.remote_deployment.run_remote_deployment"
            ) as mock_run:
                mock_run.return_value = [
                    {"name": "output.txt", "content": b"result", "sources": []}
                ]

                docs = DocumentList([])
                options = FlowOptions()
                await my_flow("project", docs, options)

                # Verify set_trace_cost was NOT called for zero cost
                mock_set_cost.assert_not_called()

    async def test_trace_parameters(self):
        """Test @remote_deployment with various trace parameters."""

        def format_input(*args: Any, **kwargs: Any) -> str:
            return f"Input: {args}"

        def format_output(result: Any) -> str:
            return f"Output: {result}"

        @remote_deployment(
            output_document_type=OutputDoc,
            name="custom_name",
            trace_level="debug",
            trace_ignore_input=True,
            trace_ignore_output=True,
            trace_ignore_inputs=["secret"],
            trace_input_formatter=format_input,
            trace_output_formatter=format_output,
            trace_trim_documents=False,
        )
        async def my_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
            secret: str = "password",
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = [{"name": "output.txt", "content": b"result", "sources": []}]

            docs = DocumentList([])
            options = FlowOptions()
            result = await my_flow("project", docs, options, secret="hidden")

            assert isinstance(result, DocumentList)

    async def test_already_traced_raises_error(self):
        """Test that applying @trace before @remote_deployment raises TypeError."""
        from ai_pipeline_core import trace

        with pytest.raises(TypeError, match="already decorated.*with @trace"):

            @remote_deployment(output_document_type=OutputDoc)
            @trace
            async def my_flow(  # pyright: ignore[reportUnusedFunction]
                project_name: str, documents: DocumentList, flow_options: FlowOptions
            ) -> DocumentList:  # pyright: ignore[reportReturnType]
                pass

    async def test_multiple_parameters_passed_correctly(self):
        """Test that all parameters are passed to remote deployment correctly."""

        @remote_deployment(output_document_type=OutputDoc)
        async def my_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: FlowOptions,
            extra_param: str = "default",
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            mock_run.return_value = [{"name": "output.txt", "content": b"result", "sources": []}]

            docs = DocumentList([])
            options = FlowOptions()
            await my_flow("project", docs, options, extra_param="custom_value")

            # Verify all parameters were passed
            call_args = mock_run.call_args
            params = call_args[1]["parameters"]
            assert params["project_name"] == "project"
            assert params["extra_param"] == "custom_value"

    async def test_result_deserialization_with_multiple_documents(self):
        """Test deserialization of multiple documents from remote result."""

        @remote_deployment(output_document_type=OutputDoc)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            # Return multiple documents
            mock_run.return_value = [
                {"name": "output1.txt", "content": b"result1", "sources": []},
                {"name": "output2.txt", "content": b"result2", "sources": []},
                {"name": "output3.txt", "content": b"result3", "sources": []},
            ]

            docs = DocumentList([])
            options = FlowOptions()
            result = await my_flow("project", docs, options)

            # Verify all documents are deserialized correctly
            assert len(result) == 3
            assert all(isinstance(doc, OutputDoc) for doc in result)
            assert result[0].name == "output1.txt"
            assert result[1].name == "output2.txt"
            assert result[2].name == "output3.txt"

    async def test_empty_result_list(self):
        """Test handling of empty result from remote deployment."""

        @remote_deployment(output_document_type=OutputDoc)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            # Return empty list
            mock_run.return_value = []

            docs = DocumentList([])
            options = FlowOptions()
            result = await my_flow("project", docs, options)

            # Verify empty DocumentList is returned
            assert isinstance(result, DocumentList)
            assert len(result) == 0

    async def test_deployment_not_found_raises_error(self):
        """Test that deployment not found raises ValueError."""

        @remote_deployment(output_document_type=OutputDoc)
        async def my_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:  # pyright: ignore[reportReturnType]
            pass

        with patch("ai_pipeline_core.utils.remote_deployment.run_remote_deployment") as mock_run:
            # Simulate deployment not found
            mock_run.side_effect = ValueError("deployment not found")

            docs = DocumentList([])
            options = FlowOptions()

            with pytest.raises(ValueError, match="deployment not found"):
                await my_flow("project", docs, options)


class TestRunRemoteDeployment:
    """Test run_remote_deployment function error handling."""

    async def test_run_remote_deployment_not_found_no_api_url(self):
        """Test run_remote_deployment when deployment not found and no API URL."""
        from prefect.exceptions import ObjectNotFound

        from ai_pipeline_core.utils.remote_deployment import run_remote_deployment

        with patch("ai_pipeline_core.utils.remote_deployment.get_client") as mock_get_client:
            # Local client raises ObjectNotFound
            mock_local_client = AsyncMock()
            mock_local_client.__aenter__.return_value = mock_local_client
            mock_local_client.__aexit__.return_value = None
            mock_local_client.read_deployment_by_name = AsyncMock(
                side_effect=ObjectNotFound(http_exc=Exception("Not found"))
            )
            mock_get_client.return_value = mock_local_client

            with patch("ai_pipeline_core.utils.remote_deployment.settings") as mock_settings:
                mock_settings.prefect_api_url = None

                with pytest.raises(
                    ValueError, match="deployment not found, PREFECT_API_URL is not set"
                ):
                    await run_remote_deployment("test-deployment", {"param": "value"})

    async def test_run_remote_deployment_not_found_anywhere(self):
        """Test run_remote_deployment when deployment not found locally or remotely."""
        from prefect.exceptions import ObjectNotFound

        from ai_pipeline_core.utils.remote_deployment import run_remote_deployment

        with patch("ai_pipeline_core.utils.remote_deployment.get_client") as mock_get_client:
            # Local client raises ObjectNotFound
            mock_local_client = AsyncMock()
            mock_local_client.__aenter__.return_value = mock_local_client
            mock_local_client.__aexit__.return_value = None
            mock_local_client.read_deployment_by_name = AsyncMock(
                side_effect=ObjectNotFound(http_exc=Exception("Not found"))
            )
            mock_get_client.return_value = mock_local_client

            with patch(
                "ai_pipeline_core.utils.remote_deployment.PrefectClient"
            ) as mock_prefect_client:
                # Remote client also raises ObjectNotFound
                mock_remote_client = AsyncMock()
                mock_remote_client.__aenter__.return_value = mock_remote_client
                mock_remote_client.__aexit__.return_value = None
                mock_remote_client.read_deployment_by_name = AsyncMock(
                    side_effect=ObjectNotFound(http_exc=Exception("Not found remotely"))
                )
                mock_prefect_client.return_value = mock_remote_client

                with patch("ai_pipeline_core.utils.remote_deployment.settings") as mock_settings:
                    mock_settings.prefect_api_url = "http://api.example.com"
                    mock_settings.prefect_api_key = "key"
                    mock_settings.prefect_api_auth_string = "auth"

                    with pytest.raises(ValueError, match="deployment not found"):
                        await run_remote_deployment("test-deployment", {"param": "value"})


class TestRemoteDeploymentUtilities:
    """Test utility functions in remote_deployment module."""

    def test_callable_name_with_named_function(self):
        """Test _callable_name with a named function."""
        from ai_pipeline_core.utils.remote_deployment import (
            _callable_name,  # type: ignore[reportPrivateUsage]
        )

        def my_func():
            pass

        assert _callable_name(my_func, "fallback") == "my_func"

    def test_callable_name_with_fallback(self):
        """Test _callable_name with fallback for objects without __name__."""
        from ai_pipeline_core.utils.remote_deployment import (
            _callable_name,  # type: ignore[reportPrivateUsage]
        )

        obj = object()
        assert _callable_name(obj, "fallback") == "fallback"

    def test_callable_name_with_exception(self):
        """Test _callable_name with object that raises exception on attribute access."""
        from ai_pipeline_core.utils.remote_deployment import (
            _callable_name,  # type: ignore[reportPrivateUsage]
        )

        class BadObject:
            @property
            def __name__(self):
                raise RuntimeError("Cannot access __name__")

        obj = BadObject()
        # Should return fallback when exception occurs
        assert _callable_name(obj, "fallback") == "fallback"

    def test_is_already_traced_false_for_untraced(self):
        """Test _is_already_traced returns False for untraced function."""
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        async def my_func():
            pass

        assert _is_already_traced(my_func) is False

    def test_is_already_traced_true_for_traced(self):
        """Test _is_already_traced returns True for traced function."""
        from ai_pipeline_core import trace
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")  # Use "always" to ensure trace is applied
        async def my_func():
            pass

        assert _is_already_traced(my_func) is True

    def test_is_already_traced_detects_nested_trace(self):
        """Test _is_already_traced detects trace in __wrapped__ chain."""
        from ai_pipeline_core import trace
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        @trace(level="always")
        @trace(level="always")  # Both use "always" to ensure both are applied
        async def my_func():
            pass

        assert _is_already_traced(my_func) is True

    def test_is_already_traced_deep_wrapped_chain(self):
        """Test _is_already_traced with deep __wrapped__ chain."""
        from functools import wraps

        from ai_pipeline_core import trace
        from ai_pipeline_core.utils.remote_deployment import (
            _is_already_traced,  # type: ignore[reportPrivateUsage]
        )

        # Create a traced function
        @trace(level="always")
        async def base_func():
            pass

        # Wrap it multiple times to create a chain
        @wraps(base_func)
        async def wrapper1():
            pass

        wrapper1.__wrapped__ = base_func  # type: ignore[attr-defined]

        @wraps(wrapper1)
        async def wrapper2():
            pass

        wrapper2.__wrapped__ = wrapper1  # type: ignore[attr-defined]

        # Should detect trace in the chain
        assert _is_already_traced(wrapper2) is True
