"""Simple pipeline runner for local flow execution.

This module provides the core functionality for running AI pipeline flows
locally without full Prefect orchestration. It handles document I/O,
flow sequencing, and error management.

Key components:
    - Document I/O from/to filesystem directories
    - Single and multi-flow execution
    - Automatic document validation and passing between flows
    - Step-based execution control (start/end steps)

Directory structure:
    working_dir/
    ├── InputDocument/       # Documents of type InputDocument
    │   ├── file1.txt
    │   └── file1.txt.description.md   # Optional description
    └── OutputDocument/      # Documents of type OutputDocument
        └── result.json

Example:
    >>> from ai_pipeline_core.simple_runner import run_pipeline
    >>>
    >>> # Run single flow
    >>> results = await run_pipeline(
    ...     flow_func=MyFlow,
    ...     config=MyConfig,
    ...     project_name="test",
    ...     output_dir=Path("./output"),
    ...     flow_options=options
    ... )

Note:
    Document directories are named using the canonical_name() method
    of each document type for consistent organization.
"""

import json
from pathlib import Path
from typing import Any, Callable, Sequence, Type

from ai_pipeline_core.documents import Document, DocumentList, FlowDocument
from ai_pipeline_core.flow.config import FlowConfig
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

FlowSequence = Sequence[Callable[..., Any]]
"""Type alias for a sequence of flow functions."""

ConfigSequence = Sequence[Type[FlowConfig]]
"""Type alias for a sequence of flow configuration classes."""


def load_documents_from_directory(
    base_dir: Path, document_types: Sequence[Type[FlowDocument]]
) -> DocumentList:
    """Load documents from filesystem directories by type.

    Scans subdirectories of base_dir for documents matching the provided
    types. Each document type has its own subdirectory named after its
    canonical_name().

    Args:
        base_dir: Base directory containing document subdirectories.
        document_types: Sequence of FlowDocument subclasses to load.
                       Each type corresponds to a subdirectory.

    Returns:
        DocumentList containing all successfully loaded documents.
        Empty list if no documents found or directories don't exist.

    Directory structure:
        base_dir/
        ├── DocumentTypeA/     # canonical_name() of DocumentTypeA
        │   ├── doc1.txt
        │   ├── doc1.txt.description.md  # Optional description file
        │   └── doc2.json
        └── DocumentTypeB/
            └── data.csv

    File handling:
        - Document content is read as bytes
        - Optional .description.md files provide document descriptions
        - Failed loads are logged but don't stop processing
        - Non-file entries are skipped

    Example:
        >>> from my_docs import InputDoc, ConfigDoc
        >>> docs = load_documents_from_directory(
        ...     Path("./data"),
        ...     [InputDoc, ConfigDoc]
        ... )
        >>> print(f"Loaded {len(docs)} documents")

    Note:
        - Uses canonical_name() for directory names (e.g., "InputDocument")
        - Descriptions are loaded from "{filename}.description.md" files
        - All file types are supported (determined by document class)
    """
    documents = DocumentList()

    for doc_class in document_types:
        dir_name = doc_class.canonical_name()
        type_dir = base_dir / dir_name

        if not type_dir.exists() or not type_dir.is_dir():
            continue

        logger.info(f"Loading documents from {type_dir.relative_to(base_dir)}")

        for file_path in type_dir.iterdir():
            if not file_path.is_file() or file_path.name.endswith(Document.DESCRIPTION_EXTENSION):
                continue

            # Skip .sources.json files - they are metadata, not documents
            if file_path.name.endswith(".sources.json"):
                continue

            try:
                content = file_path.read_bytes()

                # Load sources if .sources.json exists
                sources = []
                sources_file = file_path.with_name(file_path.name + ".sources.json")
                if sources_file.exists():
                    sources = json.loads(sources_file.read_text(encoding="utf-8"))

                doc = doc_class(name=file_path.name, content=content, sources=sources)

                desc_file = file_path.with_name(file_path.name + Document.DESCRIPTION_EXTENSION)
                if desc_file.exists():
                    object.__setattr__(doc, "description", desc_file.read_text(encoding="utf-8"))

                documents.append(doc)
            except Exception as e:
                logger.error(
                    f"  Failed to load {file_path.name} as {doc_class.__name__}: {e}", exc_info=True
                )

    return documents


def save_documents_to_directory(base_dir: Path, documents: DocumentList) -> None:
    """Save documents to filesystem directories by type.

    Creates subdirectories under base_dir for each document type and
    saves documents with their original filenames. Only FlowDocument
    instances are saved (temporary documents are skipped).

    Args:
        base_dir: Base directory for saving document subdirectories.
                 Created if it doesn't exist.
        documents: DocumentList containing documents to save.
                  Non-FlowDocument instances are silently skipped.

    Side effects:
        - Creates base_dir and subdirectories as needed
        - Overwrites existing files with the same name
        - Logs each saved document
        - Creates .description.md files for documents with descriptions

    Directory structure created:
        base_dir/
        └── DocumentType/      # canonical_name() of document
            ├── output.json    # Document content
            └── output.json.description.md  # Optional description

    Example:
        >>> docs = DocumentList([
        ...     OutputDoc(name="result.txt", content=b"data"),
        ...     OutputDoc(name="stats.json", content=b'{...}')
        ... ])
        >>> save_documents_to_directory(Path("./output"), docs)
        >>> # Creates ./output/OutputDocument/result.txt
        >>> #     and ./output/OutputDocument/stats.json

    Note:
        - Only FlowDocument subclasses are saved
        - TaskDocument and other temporary documents are skipped
        - Descriptions are saved as separate .description.md files
    """
    for document in documents:
        if not isinstance(document, FlowDocument):
            continue

        dir_name = document.canonical_name()
        document_dir = base_dir / dir_name
        document_dir.mkdir(parents=True, exist_ok=True)

        file_path = document_dir / document.name
        file_path.write_bytes(document.content)
        logger.info(f"Saved: {dir_name}/{document.name}")

        if document.description:
            desc_file = file_path.with_name(file_path.name + Document.DESCRIPTION_EXTENSION)
            desc_file.write_text(document.description, encoding="utf-8")

        # Save sources to .sources.json if present
        if document.sources:
            sources_file = file_path.with_name(file_path.name + ".sources.json")
            sources_file.write_text(json.dumps(document.sources, indent=2), encoding="utf-8")


async def run_pipeline(
    flow_func: Callable[..., Any],
    config: Type[FlowConfig],
    project_name: str,
    output_dir: Path,
    flow_options: FlowOptions,
    flow_name: str | None = None,
) -> DocumentList:
    """Execute a single pipeline flow with document I/O.

    Runs a flow function with automatic document loading, validation,
    and saving. The flow receives input documents from the filesystem
    and saves its output for subsequent flows.

    The execution proceeds through these steps:
    1. Load input documents from output_dir subdirectories
    2. Validate input documents against config requirements
    3. Execute flow function with documents and options
    4. Validate output documents match config.OUTPUT_DOCUMENT_TYPE
    5. Save output documents to output_dir subdirectories

    Args:
        flow_func: Async flow function decorated with @pipeline_flow.
                  Must accept (project_name, documents, flow_options).

        config: FlowConfig subclass defining input/output document types.
               Used for validation and directory organization.

        project_name: Name of the project/pipeline for logging and tracking.

        output_dir: Directory for loading input and saving output documents.
                   Document subdirectories are created as needed.

        flow_options: Configuration options passed to the flow function.
                     Can be FlowOptions or any subclass.

        flow_name: Optional display name for logging. If None, uses
                  flow_func.name or flow_func.__name__.

    Returns:
        DocumentList containing the flow's output documents.

    Raises:
        RuntimeError: If required input documents are missing.

    Example:
        >>> from my_flows import AnalysisFlow, AnalysisConfig
        >>>
        >>> results = await run_pipeline(
        ...     flow_func=AnalysisFlow,
        ...     config=AnalysisConfig,
        ...     project_name="analysis_001",
        ...     output_dir=Path("./results"),
        ...     flow_options=FlowOptions(temperature=0.7)
        ... )
        >>> print(f"Generated {len(results)} documents")

    Note:
        - Flow must be async (decorated with @pipeline_flow)
        - Input documents are loaded based on config.INPUT_DOCUMENT_TYPES
        - Output is validated against config.OUTPUT_DOCUMENT_TYPE
        - All I/O is logged for debugging
    """
    if flow_name is None:
        # For Prefect Flow objects, use their name attribute
        # For regular functions, fall back to __name__
        flow_name = getattr(flow_func, "name", None) or getattr(flow_func, "__name__", "flow")

    logger.info(f"Running Flow: {flow_name}")

    input_documents = load_documents_from_directory(output_dir, config.INPUT_DOCUMENT_TYPES)

    if not config.has_input_documents(input_documents):
        raise RuntimeError(f"Missing input documents for flow {flow_name}")

    result_documents = await flow_func(project_name, input_documents, flow_options)

    config.validate_output_documents(result_documents)

    save_documents_to_directory(output_dir, result_documents)

    logger.info(f"Completed Flow: {flow_name}")

    return result_documents


async def run_pipelines(
    project_name: str,
    output_dir: Path,
    flows: FlowSequence,
    flow_configs: ConfigSequence,
    flow_options: FlowOptions,
    start_step: int = 1,
    end_step: int | None = None,
) -> None:
    """Execute multiple pipeline flows in sequence.

    Runs a series of flows where each flow's output becomes the input
    for the next flow. Supports partial execution with start/end steps
    for debugging and resuming failed pipelines.

    Execution proceeds by:
    1. Validating step indices and sequence lengths
    2. For each flow in range [start_step, end_step]:
       a. Loading input documents from output_dir
       b. Executing flow with documents
       c. Saving output documents to output_dir
       d. Output becomes input for next flow
    3. Logging progress and any failures

    Steps are 1-based for user convenience. Step 1 is the first flow,
    Step N is the Nth flow. Use start_step > 1 to skip initial flows
    and end_step < N to stop early.

    Args:
        project_name: Name of the overall pipeline/project.
        output_dir: Directory for document I/O between flows.
                   Shared by all flows in the sequence.
        flows: Sequence of flow functions to execute in order.
              Must all be async functions decorated with @pipeline_flow.
        flow_configs: Sequence of FlowConfig classes corresponding to flows.
                     Must have same length as flows sequence.
        flow_options: Options passed to all flows in the sequence.
                     Individual flows can use different fields.
        start_step: First flow to execute (1-based index).
                   Default 1 starts from the beginning.
        end_step: Last flow to execute (1-based index).
                 None runs through the last flow.

    Raises:
        ValueError: If flows and configs have different lengths, or if
                   start_step or end_step are out of range.

    Example:
        >>> # Run full pipeline
        >>> await run_pipelines(
        ...     project_name="analysis",
        ...     output_dir=Path("./work"),
        ...     flows=[ExtractFlow, AnalyzeFlow, SummarizeFlow],
        ...     flow_configs=[ExtractConfig, AnalyzeConfig, SummaryConfig],
        ...     flow_options=options
        ... )
        >>>
        >>> # Run only steps 2-3 (skip extraction)
        >>> await run_pipelines(
        ...     ...,
        ...     start_step=2,
        ...     end_step=3
        ... )

    Note:
        - Each flow's output must match the next flow's input types
        - Failed flows stop the entire pipeline
        - Progress is logged with step numbers for debugging
        - Documents persist in output_dir between runs
    """
    if len(flows) != len(flow_configs):
        raise ValueError("The number of flows and flow configs must match.")

    num_steps = len(flows)
    start_index = start_step - 1
    end_index = (end_step if end_step is not None else num_steps) - 1

    if (
        not (0 <= start_index < num_steps)
        or not (0 <= end_index < num_steps)
        or start_index > end_index
    ):
        raise ValueError("Invalid start/end steps.")

    logger.info(f"Starting pipeline '{project_name}' (Steps {start_step} to {end_index + 1})")

    for i in range(start_index, end_index + 1):
        flow_func = flows[i]
        config = flow_configs[i]
        # For Prefect Flow objects, use their name attribute; for functions, use __name__
        flow_name = getattr(flow_func, "name", None) or getattr(
            flow_func, "__name__", f"flow_{i + 1}"
        )

        logger.info(f"--- [Step {i + 1}/{num_steps}] Running Flow: {flow_name} ---")

        try:
            await run_pipeline(
                flow_func=flow_func,
                config=config,
                project_name=project_name,
                output_dir=output_dir,
                flow_options=flow_options,
                flow_name=f"[Step {i + 1}/{num_steps}] {flow_name}",
            )

        except Exception as e:
            logger.error(
                f"--- [Step {i + 1}/{num_steps}] Flow {flow_name} Failed: {e} ---", exc_info=True
            )
            raise
