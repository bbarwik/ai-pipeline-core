"""Flow configuration system for type-safe pipeline definitions.

@public

This module provides the FlowConfig abstract base class that enforces
type safety for flow inputs and outputs in the pipeline system.

Best Practice:
    Always finish @pipeline_flow functions with create_and_validate_output()
    to ensure type safety and proper validation of output documents.
"""

import json
from abc import ABC
from typing import Any, ClassVar, Iterable

from ai_pipeline_core.documents import Document, DocumentList, FlowDocument
from ai_pipeline_core.exceptions import DocumentValidationError
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.storage import Storage

logger = get_pipeline_logger(__name__)


class FlowConfig(ABC):
    """Abstract base class for type-safe flow configuration.

    @public

    FlowConfig defines the contract for flow inputs and outputs, ensuring
    type safety and preventing circular dependencies in pipeline flows.
    Each flow must have a corresponding FlowConfig subclass that specifies
    its input document types and output document type.

    CRITICAL RULE: OUTPUT_DOCUMENT_TYPE must NEVER be in INPUT_DOCUMENT_TYPES!
        This prevents circular dependencies as flows chain together.
        Each flow transforms input types to a DIFFERENT output type.

    Class Variables:
        INPUT_DOCUMENT_TYPES: List of FlowDocument types this flow accepts
        OUTPUT_DOCUMENT_TYPE: Single FlowDocument type this flow produces

    Validation Rules:
        - INPUT_DOCUMENT_TYPES and OUTPUT_DOCUMENT_TYPE must be defined
        - OUTPUT_DOCUMENT_TYPE cannot be in INPUT_DOCUMENT_TYPES (prevents cycles)
        - Field names must be exact (common typos are detected)

    Why this matters:
        Flows connect in pipelines where one flow's output becomes another's input.
        Same input/output types would create infinite loops or circular dependencies.

    Example:
        >>> # CORRECT - Different output type from inputs
        >>> class ProcessingFlowConfig(FlowConfig):
        ...     INPUT_DOCUMENT_TYPES = [RawDataDocument]
        ...     OUTPUT_DOCUMENT_TYPE = ProcessedDocument  # Different type!
        >>>
        >>> # Use in @pipeline_flow - RECOMMENDED PATTERN
        >>> @pipeline_flow(config=ProcessingFlowConfig, name="processing")
        >>> async def process(
        ...     project_name: str, docs: DocumentList, flow_options: FlowOptions
        ... ) -> DocumentList:
        ...     outputs = []
        ...     # ... processing logic ...
        ...     return config.create_and_validate_output(outputs)

        >>> # WRONG - Will raise TypeError
        >>> class BadConfig(FlowConfig):
        ...     INPUT_DOCUMENT_TYPES = [DataDocument]
        ...     OUTPUT_DOCUMENT_TYPE = DataDocument  # SAME TYPE - NOT ALLOWED!

    Note:
        - Validation happens at class definition time
        - Helps catch configuration errors early
        - Used by simple_runner to manage document flow
    """

    INPUT_DOCUMENT_TYPES: ClassVar[list[type[FlowDocument]]]
    OUTPUT_DOCUMENT_TYPE: ClassVar[type[FlowDocument]]

    def __init_subclass__(cls, **kwargs: Any):
        """Validate flow configuration at subclass definition time.

        Performs comprehensive validation when a FlowConfig subclass is defined:
        1. Checks for common field name mistakes (typos)
        2. Ensures required fields are defined
        3. Prevents circular dependencies (output != input)

        Args:
            **kwargs: Additional arguments for parent __init_subclass__.

        Raises:
            TypeError: If configuration violates any validation rules:
                      - Missing required fields
                      - Incorrect field names
                      - Circular dependency detected

        Note:
            This runs at class definition time, not instantiation,
            providing immediate feedback during development.
        """
        super().__init_subclass__(**kwargs)

        # Skip validation for the abstract base class itself
        if cls.__name__ == "FlowConfig":
            return

        # Check for invalid field names (common mistakes)
        allowed_fields = {"INPUT_DOCUMENT_TYPES", "OUTPUT_DOCUMENT_TYPE"}
        class_attrs = {name for name in dir(cls) if not name.startswith("_") and name.isupper()}

        # Find fields that look like they might be mistakes
        suspicious_fields = class_attrs - allowed_fields
        common_mistakes = {
            "OUTPUT_DOCUMENT_TYPES": "OUTPUT_DOCUMENT_TYPE",
            "INPUT_DOCUMENT_TYPE": "INPUT_DOCUMENT_TYPES",
        }

        for field in suspicious_fields:
            # Skip inherited attributes from parent classes
            if any(hasattr(base, field) for base in cls.__bases__):
                continue

            if field in common_mistakes:
                raise TypeError(
                    f"FlowConfig {cls.__name__}: Found '{field}' but expected "
                    f"'{common_mistakes[field]}'. Please use the correct field name."
                )
            elif "DOCUMENT" in field:
                raise TypeError(
                    f"FlowConfig {cls.__name__}: Invalid field '{field}'. "
                    f"Only 'INPUT_DOCUMENT_TYPES' and 'OUTPUT_DOCUMENT_TYPE' are allowed."
                )

        # Ensure required attributes are defined
        if not hasattr(cls, "INPUT_DOCUMENT_TYPES"):
            raise TypeError(f"FlowConfig {cls.__name__} must define INPUT_DOCUMENT_TYPES")
        if not hasattr(cls, "OUTPUT_DOCUMENT_TYPE"):
            raise TypeError(f"FlowConfig {cls.__name__} must define OUTPUT_DOCUMENT_TYPE")

        # Validate that output type is not in input types
        if cls.OUTPUT_DOCUMENT_TYPE in cls.INPUT_DOCUMENT_TYPES:
            raise TypeError(
                f"FlowConfig {cls.__name__}: OUTPUT_DOCUMENT_TYPE "
                f"({cls.OUTPUT_DOCUMENT_TYPE.__name__}) cannot be in INPUT_DOCUMENT_TYPES"
            )

    @classmethod
    def get_input_document_types(cls) -> list[type[FlowDocument]]:
        """Get the list of input document types this flow accepts.

        Returns:
            List of FlowDocument subclasses that this flow requires
            as input.

        Example:
            >>> types = MyFlowConfig.get_input_document_types()
            >>> print([t.__name__ for t in types])
            ['InputDoc', 'ConfigDoc']
        """
        return cls.INPUT_DOCUMENT_TYPES

    @classmethod
    def get_output_document_type(cls) -> type[FlowDocument]:
        """Get the output document type this flow produces.

        Returns:
            Single FlowDocument subclass that this flow outputs.

        Example:
            >>> output_type = MyFlowConfig.get_output_document_type()
            >>> print(output_type.__name__)
            'ProcessedDataDocument'
        """
        return cls.OUTPUT_DOCUMENT_TYPE

    @classmethod
    def has_input_documents(cls, documents: DocumentList) -> bool:
        """Check if all required input documents are present.

        Verifies that the document list contains at least one instance
        of each required input document type.

        Args:
            documents: DocumentList to check for required inputs.

        Returns:
            True if all required document types are present,
            False if any are missing.

        Example:
            >>> docs = DocumentList([input_doc, config_doc])
            >>> if MyFlowConfig.has_input_documents(docs):
            ...     # Safe to proceed with flow
            ...     pass

        Note:
            Use this before get_input_documents() to avoid exceptions.
        """
        for doc_cls in cls.INPUT_DOCUMENT_TYPES:
            if not any(isinstance(doc, doc_cls) for doc in documents):
                return False
        return True

    @classmethod
    def get_input_documents(cls, documents: DocumentList) -> DocumentList:
        """Extract and return all required input documents.

        Filters the provided document list to return only documents
        matching the required input types. Returns all matching documents,
        not just the first of each type.

        Args:
            documents: DocumentList containing mixed document types.

        Returns:
            DocumentList containing only the required input documents.

        Raises:
            ValueError: If any required document type is missing.

        Example:
            >>> all_docs = DocumentList([input1, input2, other_doc])
            >>> input_docs = MyFlowConfig.get_input_documents(all_docs)
            >>> len(input_docs)  # Contains only input1 and input2
            2

        Note:
            Call has_input_documents() first to check availability.
        """
        input_documents = DocumentList()
        for doc_cls in cls.INPUT_DOCUMENT_TYPES:
            filtered_documents = [doc for doc in documents if isinstance(doc, doc_cls)]
            if not filtered_documents:
                raise ValueError(f"No input document found for class {doc_cls.__name__}")
            input_documents.extend(filtered_documents)
        return input_documents

    @classmethod
    def validate_output_documents(cls, documents: Any) -> None:
        """Validate that output documents match the expected type.

        Ensures all documents in the list are instances of the
        declared OUTPUT_DOCUMENT_TYPE.

        Args:
            documents: DocumentList to validate.

        Raises:
            DocumentValidationError: If documents is not a DocumentList or if any
                document has incorrect type.

        Example:
            >>> output = DocumentList([ProcessedDoc(...)])
            >>> MyFlowConfig.validate_output_documents(output)
            >>> # No exception means valid

        Note:
            Used internally by create_and_validate_output().
            Uses explicit exceptions for validation (works with python -O).
        """
        if not isinstance(documents, DocumentList):
            raise DocumentValidationError("Documents must be a DocumentList")

        output_document_class = cls.get_output_document_type()

        for doc in documents:
            if not isinstance(doc, output_document_class):
                raise DocumentValidationError(
                    f"Document '{doc.name}' has incorrect type. "
                    f"Expected: {output_document_class.__name__}, "
                    f"Got: {type(doc).__name__}"
                )

    @classmethod
    def create_and_validate_output(
        cls, output: FlowDocument | Iterable[FlowDocument] | DocumentList
    ) -> DocumentList:
        """Create and validate flow output documents.

        @public

        RECOMMENDED: Always use this method at the end of @pipeline_flow functions
        to ensure type safety and proper output validation.

        Convenience method that wraps output in a DocumentList if needed
        and validates it matches the expected OUTPUT_DOCUMENT_TYPE.

        Args:
            output: Single document, iterable of documents, or DocumentList.

        Returns:
            Validated DocumentList containing the output documents.

        Raises:
            DocumentValidationError: If output type doesn't match OUTPUT_DOCUMENT_TYPE.

        Example:
            >>> @pipeline_flow(config=MyFlowConfig, name="my_flow")
            >>> async def process_flow(
            ...     project_name: str, documents: DocumentList, flow_options: FlowOptions
            ... ) -> DocumentList:
            >>>     outputs = []
            >>>     # ... processing logic ...
            >>>     outputs.append(OutputDoc(...))
            >>>
            >>>     # Always finish with this validation
            >>>     return config.create_and_validate_output(outputs)

        Note:
            This is the recommended pattern for all @pipeline_flow functions.
            It ensures type safety and catches output errors immediately.
        """
        documents: DocumentList
        if isinstance(output, FlowDocument):
            documents = DocumentList([output])
        elif isinstance(output, DocumentList):
            documents = output
        else:
            # Handle any iterable of FlowDocuments
            documents = DocumentList(list(output))  # type: ignore[arg-type]
        cls.validate_output_documents(documents)
        return documents

    @classmethod
    async def load_documents(
        cls,
        uri: str,
    ) -> DocumentList:
        """Load documents from storage matching INPUT_DOCUMENT_TYPES.

        Loads documents from a storage location based on the class's INPUT_DOCUMENT_TYPES.
        Supports both local filesystem and Google Cloud Storage backends.
        Automatically loads metadata (.description.md and .sources.json) when present.

        Args:
            uri: Storage URI (file://, gs://, or local path)

        Returns:
            DocumentList containing loaded documents matching INPUT_DOCUMENT_TYPES

        Example:
            >>> # Load from local filesystem
            >>> docs = await MyFlowConfig.load_documents("./data")
            >>>
            >>> # Load from GCS (uses GCS_SERVICE_ACCOUNT_FILE from settings if configured)
            >>> docs = await MyFlowConfig.load_documents("gs://bucket/data")
        """
        # Use INPUT_DOCUMENT_TYPES if not specified
        storage = await Storage.from_uri(uri)
        loaded_documents = DocumentList()

        # Process each document type
        for doc_type in cls.INPUT_DOCUMENT_TYPES:
            canonical_name = doc_type.canonical_name()
            doc_storage = storage.with_base(canonical_name)

            # Check if subdirectory exists
            if not await doc_storage.exists(""):
                logger.debug(f"Subdirectory {canonical_name} not found, skipping")
                continue

            # List files in subdirectory
            objects = await doc_storage.list("", recursive=False, include_dirs=False)

            # Create lookup set for metadata files
            object_keys = {obj.key for obj in objects}

            # Filter out metadata files
            doc_files = [
                obj
                for obj in objects
                if not obj.key.endswith(Document.DESCRIPTION_EXTENSION)
                and not obj.key.endswith(Document.SOURCES_EXTENSION)
            ]

            for obj in doc_files:
                try:
                    # Load document content
                    content = await doc_storage.read_bytes(obj.key)

                    # Load metadata if present
                    description = None
                    sources: list[str] = []

                    # Check for description in objects list
                    desc_path = f"{obj.key}{Document.DESCRIPTION_EXTENSION}"
                    if desc_path in object_keys:
                        try:
                            description = await doc_storage.read_text(desc_path)
                        except Exception as e:
                            logger.warning(f"Failed to load description for {obj.key}: {e}")

                    # Check for sources in objects list
                    sources_path = f"{obj.key}{Document.SOURCES_EXTENSION}"
                    if sources_path in object_keys:
                        try:
                            sources_text = await doc_storage.read_text(sources_path)
                            sources = json.loads(sources_text)
                        except Exception as e:
                            logger.warning(f"Failed to load sources for {obj.key}: {e}")

                    # Create document instance
                    doc = doc_type(
                        name=obj.key,
                        content=content,
                        description=description,
                        sources=sources,
                    )

                    loaded_documents.append(doc)
                    logger.debug(f"Loaded {doc_type.__name__} document: {obj.key}")
                except Exception as e:
                    logger.error(f"Failed to load {doc_type.__name__} document {obj.key}: {e}")

        logger.info(f"Loaded {len(loaded_documents)} documents from {uri}")
        return loaded_documents

    @classmethod
    async def save_documents(
        cls,
        uri: str,
        documents: DocumentList,
        *,
        validate_output_type: bool = True,
    ) -> None:
        """Save documents to storage with metadata.

        Saves FlowDocument instances to a storage location with their content
        and metadata files (Document.DESCRIPTION_EXTENSION and Document.SOURCES_EXTENSION).
        Non-FlowDocument instances (TaskDocument, TemporaryDocument) are skipped.

        Args:
            uri: Storage URI (file://, gs://, or local path)
            documents: DocumentList to save
            validate_output_type: If True, validate documents match cls.OUTPUT_DOCUMENT_TYPE

        Raises:
            DocumentValidationError: If validate_output_type=True and documents don't match
                                   OUTPUT_DOCUMENT_TYPE

        Example:
            >>> # Save to local filesystem
            >>> await MyFlowConfig.save_documents("./output", docs)
            >>>
            >>> # Save to GCS (uses GCS_SERVICE_ACCOUNT_FILE from settings if configured)
            >>> await MyFlowConfig.save_documents("gs://bucket/output", docs)
        """
        # Validate output type if requested
        if validate_output_type:
            cls.validate_output_documents(documents)

        storage = await Storage.from_uri(uri)
        saved_count = 0

        for doc in documents:
            # Skip non-FlowDocument instances
            if not isinstance(doc, FlowDocument):
                logger.warning(f"Skipping non-FlowDocument: {type(doc).__name__}")
                continue

            # Get canonical name for subdirectory
            canonical_name = doc.canonical_name()
            doc_storage = storage.with_base(canonical_name)

            # Save document content
            await doc_storage.write_bytes(doc.name, doc.content)
            saved_count += 1

            # Save description if present
            if doc.description:
                desc_path = f"{doc.name}{Document.DESCRIPTION_EXTENSION}"
                await doc_storage.write_text(desc_path, doc.description)

            # Save sources if present
            if doc.sources:
                sources_path = f"{doc.name}{Document.SOURCES_EXTENSION}"
                sources_json = json.dumps(doc.sources, indent=2)
                await doc_storage.write_text(sources_path, sources_json)

            logger.debug(f"Saved {type(doc).__name__} document: {doc.name}")

        logger.info(f"Saved {saved_count} documents to {uri}")
