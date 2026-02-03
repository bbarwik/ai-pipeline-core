"""Tests for FlowOptions inheritance and pipeline_flow compatibility."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from ai_pipeline_core.documents import Document
from ai_pipeline_core.pipeline import FlowOptions, pipeline_flow


class TestFlowOptionsInheritance:
    """Test FlowOptions can be inherited and extended."""

    def test_base_flow_options_is_empty(self):
        """Test that base FlowOptions has no predefined fields."""
        options = FlowOptions()
        assert not hasattr(options, "core_model")
        assert not hasattr(options, "small_model")

    def test_base_flow_options_accepts_extra(self):
        """Test that base FlowOptions accepts extra fields (extra='allow')."""
        options = FlowOptions(unknown_field="value")
        assert options.unknown_field == "value"

    def test_flow_options_is_frozen(self):
        """Test that FlowOptions instances are immutable."""

        class SimpleOptions(FlowOptions):
            core_model: str = "default"

        options = SimpleOptions()
        with pytest.raises(ValidationError):
            options.core_model = "new-model"

    def test_inherited_flow_options_basic(self):
        """Test basic inheritance from FlowOptions."""

        class ProjectFlowOptions(FlowOptions):
            """Project-specific flow options."""

            core_model: str = "gemini-3-pro"
            small_model: str = "grok-4.1-fast"
            batch_max_chars: int = Field(default=100_000, gt=0)
            batch_max_files: int = Field(default=25, gt=0)
            enable_caching: bool = Field(default=True)

        # Test with defaults
        options = ProjectFlowOptions()
        assert options.core_model == "gemini-3-pro"
        assert options.small_model == "grok-4.1-fast"
        assert options.batch_max_chars == 100_000
        assert options.batch_max_files == 25
        assert options.enable_caching is True

        # Test with custom values
        options = ProjectFlowOptions(core_model="custom-model", batch_max_chars=200_000, enable_caching=False)
        assert options.core_model == "custom-model"
        assert options.batch_max_chars == 200_000
        assert options.enable_caching is False

    def test_inherited_flow_options_with_lists(self):
        """Test inheritance with list fields."""

        class ExtendedFlowOptions(FlowOptions):
            """Extended options with model lists."""

            supporting_models: list[str] = Field(default_factory=list)
            search_models: list[str] = Field(default_factory=lambda: ["sonar-pro-search"])
            tags: list[str] = Field(default_factory=list)

        options = ExtendedFlowOptions(supporting_models=["model1", "model2"], tags=["tag1", "tag2"])
        assert options.supporting_models == ["model1", "model2"]
        assert options.search_models == ["sonar-pro-search"]
        assert options.tags == ["tag1", "tag2"]

    def test_inherited_flow_options_with_nested_models(self):
        """Test inheritance with nested Pydantic models."""

        class DatabaseConfig(BaseModel):
            host: str = "localhost"
            port: int = 5432
            database: str = "ai_pipeline"

        class AdvancedFlowOptions(FlowOptions):
            """Options with nested configuration."""

            database: DatabaseConfig = Field(default_factory=DatabaseConfig)
            max_retries: int = Field(default=3, ge=0)

        options = AdvancedFlowOptions()
        assert options.database.host == "localhost"
        assert options.database.port == 5432
        assert options.max_retries == 3

        # Test with custom database config
        custom_db = DatabaseConfig(host="remote", port=3306, database="custom")
        options = AdvancedFlowOptions(database=custom_db, max_retries=5)
        assert options.database.host == "remote"
        assert options.database.port == 3306
        assert options.max_retries == 5

    def test_inherited_flow_options_maintains_frozen(self):
        """Test that inherited classes maintain frozen configuration."""

        class CustomFlowOptions(FlowOptions):
            custom_field: str = "default"
            # Inherits frozen=True from parent

        options = CustomFlowOptions()
        with pytest.raises(ValidationError):
            options.custom_field = "new_value"

    def test_inherited_flow_options_with_validators(self):
        """Test inheritance with custom validators."""

        class ValidatedFlowOptions(FlowOptions):
            """Options with custom validation."""

            core_model: str = "gemini-3-pro"
            small_model: str = "grok-4.1-fast"
            temperature: float = Field(default=0.7, ge=0.0, le=2.0)

            @model_validator(mode="after")
            def validate_temperature_model_combination(self) -> "ValidatedFlowOptions":
                # Example validation: high temperature requires core model
                if self.temperature > 1.5 and self.core_model == self.small_model:
                    raise ValueError("High temperature requires different core and small models")
                return self

        # Valid options
        options = ValidatedFlowOptions(temperature=0.5)
        assert options.temperature == 0.5

        # Valid high temperature with different models
        options = ValidatedFlowOptions(temperature=1.8, core_model="gpt-5.1", small_model="gpt-5-mini")
        assert options.temperature == 1.8

        # Invalid temperature
        with pytest.raises(ValidationError):
            ValidatedFlowOptions(temperature=2.5)


class FlowInputDocument(Document):
    """Input document for flow options inheritance tests."""


class FlowOutputDocument(Document):
    """Output document for flow options inheritance tests."""


class TestDocumentsFlowWithInheritedOptions:
    """Test that pipeline_flow works with inherited FlowOptions."""

    def test_documents_flow_with_base_options(self):
        """Test pipeline_flow with base FlowOptions."""

        @pipeline_flow()
        async def test_flow(project_name: str, documents: list[Document], flow_options: FlowOptions) -> list[Document]:
            assert isinstance(project_name, str)
            assert isinstance(flow_options, FlowOptions)
            return list([FlowOutputDocument(name="output", content=b"test")])

        result = asyncio.run(
            test_flow(project_name="test", documents=list([]), flow_options=FlowOptions())  # type: ignore[call-overload]
        )
        assert isinstance(result, list)

    def test_documents_flow_with_inherited_options(self):
        """Test pipeline_flow with inherited FlowOptions."""

        class CustomFlowOptions(FlowOptions):
            core_model: str = "gemini-3-pro"
            batch_size: int = Field(default=10, gt=0)
            enable_logging: bool = Field(default=True)

        @pipeline_flow()  # type: ignore[arg-type]
        async def test_flow(
            project_name: str,
            documents: list[Document],
            flow_options: CustomFlowOptions,
        ) -> list[Document]:
            assert isinstance(flow_options, CustomFlowOptions)
            assert isinstance(flow_options, FlowOptions)
            assert flow_options.core_model == "custom-core"
            assert flow_options.batch_size == 20
            assert flow_options.enable_logging is False
            return list([FlowOutputDocument(name="output", content=b"test")])

        custom_options = CustomFlowOptions(core_model="custom-core", batch_size=20, enable_logging=False)

        result = asyncio.run(
            test_flow(project_name="test", documents=list([]), flow_options=custom_options)  # type: ignore[call-overload]
        )
        assert isinstance(result, list)

    def test_documents_flow_with_complex_inherited_options(self):
        """Test pipeline_flow with complex inherited options including nested models."""

        class APIConfig(BaseModel):
            endpoint: str = "https://api.example.com"
            timeout: int = 30
            retry_count: int = 3

        class AdvancedFlowOptions(FlowOptions):
            core_model: str = "gemini-3-pro"
            small_model: str = "grok-4.1-fast"
            api_config: APIConfig = Field(default_factory=APIConfig)
            processing_modes: list[str] = Field(default_factory=lambda: ["fast", "accurate"])
            metadata: dict[str, Any] = Field(default_factory=dict)

        @pipeline_flow()  # type: ignore[arg-type]
        async def advanced_flow(project_name: str, documents: list[Document], flow_options: AdvancedFlowOptions) -> list[Document]:
            assert flow_options.core_model == "gemini-3-pro"
            assert flow_options.small_model == "custom-small"
            assert flow_options.api_config.endpoint == "https://custom.api.com"
            assert flow_options.api_config.timeout == 60
            assert "fast" in flow_options.processing_modes
            assert "parallel" in flow_options.processing_modes
            assert flow_options.metadata["version"] == "2.0"
            return list([FlowOutputDocument(name="output", content=b"test")])

        api_config = APIConfig(endpoint="https://custom.api.com", timeout=60, retry_count=5)
        options = AdvancedFlowOptions(
            small_model="custom-small",
            api_config=api_config,
            processing_modes=["fast", "parallel"],
            metadata={"version": "2.0", "author": "test"},
        )

        result = asyncio.run(
            advanced_flow(  # type: ignore[call-overload]
                project_name="advanced-test", documents=list([]), flow_options=options
            )
        )
        assert isinstance(result, list)

    def test_documents_flow_type_checking(self):
        """Test that pipeline_flow properly validates FlowOptions types."""

        class StrictFlowOptions(FlowOptions):
            required_field: str  # No default - required field

        @pipeline_flow()  # type: ignore[arg-type]
        async def strict_flow(project_name: str, documents: list[Document], flow_options: StrictFlowOptions) -> list[Document]:
            assert flow_options.required_field == "test-value"
            return list([FlowOutputDocument(name="output", content=b"test")])

        options = StrictFlowOptions(required_field="test-value")
        result = asyncio.run(
            strict_flow(project_name="test", documents=list([]), flow_options=options)  # type: ignore[call-overload]
        )
        assert isinstance(result, list)

        with pytest.raises(ValidationError):
            StrictFlowOptions()  # type: ignore[call-arg]

    def test_multiple_inheritance_levels(self):
        """Test FlowOptions with multiple inheritance levels."""

        class BaseProjectOptions(FlowOptions):
            core_model: str = "gemini-3-pro"
            organization: str = "default-org"
            environment: str = "development"

        class SpecificProjectOptions(BaseProjectOptions):
            feature_flags: dict[str, bool] = Field(default_factory=dict)

        @pipeline_flow()  # type: ignore[arg-type]
        async def multi_level_flow(project_name: str, documents: list[Document], flow_options: SpecificProjectOptions) -> list[Document]:
            assert flow_options.core_model == "gemini-3-pro"
            assert flow_options.organization == "custom-org"
            assert flow_options.environment == "production"
            assert flow_options.feature_flags["new_feature"] is True
            return list([FlowOutputDocument(name="output", content=b"test")])

        options = SpecificProjectOptions(
            organization="custom-org",
            environment="production",
            feature_flags={"new_feature": True, "beta_feature": False},
        )

        result = asyncio.run(
            multi_level_flow(  # type: ignore[call-overload]
                project_name="multi-level", documents=list([]), flow_options=options
            )
        )
        assert isinstance(result, list)

    def test_flow_options_with_pydantic_fields(self):
        """Test FlowOptions with Pydantic Field definitions."""
        PRIMARY_MODELS = ["gpt-5.1", "gpt-5-mini"]
        SMALL_MODELS = ["gpt-5-mini", "gemini-3-flash"]
        SEARCH_MODELS = ["sonar", "gemini-3-flash-search"]

        class ProjectFlowOptions(FlowOptions):
            primary_models: list[str] = Field(default_factory=lambda: PRIMARY_MODELS.copy())
            small_models_list: list[str] = Field(default_factory=lambda: SMALL_MODELS.copy())
            search_models: list[str] = Field(default_factory=lambda: SEARCH_MODELS.copy())

        @pipeline_flow()
        async def convert_input_documents(
            project_name: str,
            documents: list[Document],
            flow_options: ProjectFlowOptions,
        ) -> list[Document]:
            assert isinstance(flow_options.primary_models, list)
            assert isinstance(flow_options.small_models_list, list)
            assert isinstance(flow_options.search_models, list)
            return list([FlowOutputDocument(name="output", content=b"test")])

        options = ProjectFlowOptions()
        result = asyncio.run(convert_input_documents(project_name="test", documents=list([]), flow_options=options))
        assert isinstance(result, list)
