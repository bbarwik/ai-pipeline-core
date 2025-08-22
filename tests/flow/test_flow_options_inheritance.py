"""Tests for FlowOptions inheritance and documents_flow compatibility."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from ai_pipeline_core.documents import DocumentList, FlowDocument
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.pipeline import pipeline_flow


class TestFlowOptionsInheritance:
    """Test FlowOptions can be inherited and extended."""

    def test_base_flow_options_defaults(self):
        """Test that base FlowOptions has correct defaults."""
        options = FlowOptions()
        assert options.core_model == "gpt-5"
        assert options.small_model == "gpt-5-mini"

    def test_base_flow_options_custom_values(self):
        """Test setting custom values for base FlowOptions."""
        options = FlowOptions(core_model="custom-core-model", small_model="custom-small-model")
        assert options.core_model == "custom-core-model"
        assert options.small_model == "custom-small-model"

    def test_flow_options_accepts_model_name_type(self):
        """Test that FlowOptions accepts ModelName type literals."""
        options = FlowOptions(
            core_model="gpt-5",  # This is a valid ModelName
            small_model="gpt-5-mini",  # This is a valid ModelName
        )
        assert options.core_model == "gpt-5"
        assert options.small_model == "gpt-5-mini"

    def test_flow_options_accepts_arbitrary_strings(self):
        """Test that FlowOptions accepts arbitrary string model names."""
        options = FlowOptions(
            core_model="anthropic/claude-3-opus", small_model="anthropic/claude-3-haiku"
        )
        assert options.core_model == "anthropic/claude-3-opus"
        assert options.small_model == "anthropic/claude-3-haiku"

    def test_flow_options_is_frozen(self):
        """Test that FlowOptions instances are immutable."""
        options = FlowOptions()
        with pytest.raises(ValidationError):
            options.core_model = "new-model"

    def test_inherited_flow_options_basic(self):
        """Test basic inheritance from FlowOptions."""

        class ProjectFlowOptions(FlowOptions):
            """Project-specific flow options."""

            batch_max_chars: int = Field(default=100_000, gt=0)
            batch_max_files: int = Field(default=25, gt=0)
            enable_caching: bool = Field(default=True)

        # Test with defaults
        options = ProjectFlowOptions()
        assert options.core_model == "gpt-5"
        assert options.small_model == "gpt-5-mini"
        assert options.batch_max_chars == 100_000
        assert options.batch_max_files == 25
        assert options.enable_caching is True

        # Test with custom values
        options = ProjectFlowOptions(
            core_model="custom-model", batch_max_chars=200_000, enable_caching=False
        )
        assert options.core_model == "custom-model"
        assert options.batch_max_chars == 200_000
        assert options.enable_caching is False

    def test_inherited_flow_options_with_lists(self):
        """Test inheritance with list fields."""

        class ExtendedFlowOptions(FlowOptions):
            """Extended options with model lists."""

            supporting_models: list[str] = Field(default_factory=list)
            search_models: list[str] = Field(default_factory=lambda: ["gpt-4o-search"])
            tags: list[str] = Field(default_factory=list)

        options = ExtendedFlowOptions(supporting_models=["model1", "model2"], tags=["tag1", "tag2"])
        assert options.supporting_models == ["model1", "model2"]
        assert options.search_models == ["gpt-4o-search"]
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
        with pytest.raises(ValidationError):
            options.core_model = "new_model"

    def test_inherited_flow_options_with_validators(self):
        """Test inheritance with custom validators."""

        class ValidatedFlowOptions(FlowOptions):
            """Options with custom validation."""

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
        options = ValidatedFlowOptions(
            temperature=1.8, core_model="gpt-5", small_model="gpt-5-mini"
        )
        assert options.temperature == 1.8

        # Invalid temperature
        with pytest.raises(ValidationError):
            ValidatedFlowOptions(temperature=2.5)


class TestDocumentsFlowWithInheritedOptions:
    """Test that documents_flow works with inherited FlowOptions."""

    def test_documents_flow_with_base_options(self):
        """Test documents_flow with base FlowOptions."""

        class SampleDocument(FlowDocument):
            pass

        @pipeline_flow
        async def test_flow(
            project_name: str, documents: DocumentList, flow_options: FlowOptions
        ) -> DocumentList:
            # Verify we received the correct types
            assert isinstance(project_name, str)
            assert isinstance(documents, DocumentList)
            assert isinstance(flow_options, FlowOptions)
            assert flow_options.core_model == "gpt-5"
            assert flow_options.small_model == "gpt-5-mini"
            # Use SampleDocument to avoid unused warning
            if SampleDocument:
                pass
            return documents

        # Run the flow
        result = asyncio.run(
            test_flow(project_name="test", documents=DocumentList([]), flow_options=FlowOptions())  # type: ignore[call-overload]
        )
        assert isinstance(result, DocumentList)

    def test_documents_flow_with_inherited_options(self):
        """Test documents_flow with inherited FlowOptions."""

        class CustomFlowOptions(FlowOptions):
            batch_size: int = Field(default=10, gt=0)
            enable_logging: bool = Field(default=True)

        class SampleDocument(FlowDocument):
            pass

        @pipeline_flow  # type: ignore[arg-type]
        async def test_flow(
            project_name: str,
            documents: DocumentList,
            flow_options: CustomFlowOptions,  # Using inherited type
        ) -> DocumentList:
            # Verify we received the correct types
            assert isinstance(flow_options, CustomFlowOptions)
            assert isinstance(flow_options, FlowOptions)  # Also instance of parent
            assert flow_options.core_model == "custom-core"
            assert flow_options.batch_size == 20
            assert flow_options.enable_logging is False
            # Use SampleDocument to avoid unused warning
            if SampleDocument:
                pass
            return documents

        # Run the flow with custom options
        custom_options = CustomFlowOptions(
            core_model="custom-core", batch_size=20, enable_logging=False
        )

        result = asyncio.run(
            test_flow(project_name="test", documents=DocumentList([]), flow_options=custom_options)  # type: ignore[call-overload]
        )
        assert isinstance(result, DocumentList)

    def test_documents_flow_with_complex_inherited_options(self):
        """Test documents_flow with complex inherited options including nested models."""

        class APIConfig(BaseModel):
            endpoint: str = "https://api.example.com"
            timeout: int = 30
            retry_count: int = 3

        class AdvancedFlowOptions(FlowOptions):
            api_config: APIConfig = Field(default_factory=APIConfig)
            processing_modes: list[str] = Field(default_factory=lambda: ["fast", "accurate"])
            metadata: dict[str, Any] = Field(default_factory=dict)

        class SampleDocument(FlowDocument):
            pass

        @pipeline_flow  # type: ignore[arg-type]
        async def advanced_flow(
            project_name: str, documents: DocumentList, flow_options: AdvancedFlowOptions
        ) -> DocumentList:
            # Access inherited fields
            assert flow_options.core_model == "gpt-5"
            assert flow_options.small_model == "custom-small"

            # Access custom fields
            assert flow_options.api_config.endpoint == "https://custom.api.com"
            assert flow_options.api_config.timeout == 60
            assert "fast" in flow_options.processing_modes
            assert "parallel" in flow_options.processing_modes
            assert flow_options.metadata["version"] == "2.0"
            # Use SampleDocument to avoid unused warning
            if SampleDocument:
                pass
            return documents

        # Create complex options
        api_config = APIConfig(endpoint="https://custom.api.com", timeout=60, retry_count=5)

        options = AdvancedFlowOptions(
            small_model="custom-small",
            api_config=api_config,
            processing_modes=["fast", "parallel"],
            metadata={"version": "2.0", "author": "test"},
        )

        result = asyncio.run(
            advanced_flow(  # type: ignore[call-overload]
                project_name="advanced-test", documents=DocumentList([]), flow_options=options
            )
        )
        assert isinstance(result, DocumentList)

    def test_documents_flow_type_checking(self):
        """Test that documents_flow properly validates FlowOptions types."""

        class StrictFlowOptions(FlowOptions):
            required_field: str  # No default - required field

        class SampleDocument(FlowDocument):
            pass

        @pipeline_flow  # type: ignore[arg-type]
        async def strict_flow(
            project_name: str, documents: DocumentList, flow_options: StrictFlowOptions
        ) -> DocumentList:
            assert flow_options.required_field == "test-value"
            # Use SampleDocument to avoid unused warning
            if SampleDocument:
                pass
            return documents

        # Should work with proper options
        options = StrictFlowOptions(required_field="test-value")
        result = asyncio.run(
            strict_flow(project_name="test", documents=DocumentList([]), flow_options=options)  # type: ignore[call-overload]
        )
        assert isinstance(result, DocumentList)

        # Should fail to create options without required field
        with pytest.raises(ValidationError):
            StrictFlowOptions()  # type: ignore[call-arg]

    def test_multiple_inheritance_levels(self):
        """Test FlowOptions with multiple inheritance levels."""

        class BaseProjectOptions(FlowOptions):
            """First level of inheritance."""

            organization: str = "default-org"
            environment: str = "development"

        class SpecificProjectOptions(BaseProjectOptions):
            """Second level of inheritance."""

            feature_flags: dict[str, bool] = Field(default_factory=dict)

        class SampleDocument(FlowDocument):
            pass

        @pipeline_flow  # type: ignore[arg-type]
        async def multi_level_flow(
            project_name: str, documents: DocumentList, flow_options: SpecificProjectOptions
        ) -> DocumentList:
            # Can access all levels of inheritance
            assert flow_options.core_model == "gpt-5"  # From FlowOptions
            assert flow_options.organization == "custom-org"  # From BaseProjectOptions
            assert flow_options.environment == "production"  # From BaseProjectOptions
            assert flow_options.feature_flags["new_feature"] is True  # From SpecificProjectOptions
            # Use SampleDocument to avoid unused warning
            if SampleDocument:
                pass
            return documents

        options = SpecificProjectOptions(
            organization="custom-org",
            environment="production",
            feature_flags={"new_feature": True, "beta_feature": False},
        )

        result = asyncio.run(
            multi_level_flow(  # type: ignore[call-overload]
                project_name="multi-level", documents=DocumentList([]), flow_options=options
            )
        )
        assert isinstance(result, DocumentList)

    def test_flow_options_with_pydantic_fields(self):
        """Test FlowOptions with Pydantic Field definitions (reproduces user issue)."""

        # Define constants like the user has
        PRIMARY_MODELS = ["gpt-5", "gpt-5-mini"]
        SMALL_MODELS = ["gpt-5-mini", "gemini-2.5-flash"]
        SEARCH_MODELS = ["sonar", "gemini-2.5-flash-search"]

        class ProjectFlowOptions(FlowOptions):
            """Project-specific flow options extending the base FlowOptions."""

            # Using Pydantic Field() like the user
            primary_models: list[str] = Field(default_factory=lambda: PRIMARY_MODELS.copy())
            small_models_list: list[str] = Field(default_factory=lambda: SMALL_MODELS.copy())
            search_models: list[str] = Field(default_factory=lambda: SEARCH_MODELS.copy())

        class SampleDocument(FlowDocument):
            pass

        # This should reproduce the type error
        @pipeline_flow
        async def convert_input_documents(
            project_name: str,
            documents: DocumentList,
            flow_options: ProjectFlowOptions,
        ) -> DocumentList:
            # Verify we can access all fields
            assert isinstance(flow_options.primary_models, list)
            assert isinstance(flow_options.small_models_list, list)
            assert isinstance(flow_options.search_models, list)
            # Use SampleDocument to avoid unused warning
            if SampleDocument:
                pass
            return documents

        # Try to run the flow
        options = ProjectFlowOptions()
        result = asyncio.run(
            convert_input_documents(
                project_name="test", documents=DocumentList([]), flow_options=options
            )
        )
        assert isinstance(result, DocumentList)
