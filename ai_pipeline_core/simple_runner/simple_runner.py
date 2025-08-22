from pathlib import Path
from typing import Any, Callable, Sequence, Type

from ai_pipeline_core.documents import Document, DocumentList, FlowDocument
from ai_pipeline_core.flow.config import FlowConfig
from ai_pipeline_core.flow.options import FlowOptions
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

FlowSequence = Sequence[Callable[..., Any]]
ConfigSequence = Sequence[Type[FlowConfig]]


def load_documents_from_directory(
    base_dir: Path, document_types: Sequence[Type[FlowDocument]]
) -> DocumentList:
    """Loads documents using canonical_name."""
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

            try:
                content = file_path.read_bytes()
                doc = doc_class(name=file_path.name, content=content)

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
    """Saves documents using canonical_name."""
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


async def run_pipeline(
    flow_func: Callable[..., Any],
    config: Type[FlowConfig],
    project_name: str,
    output_dir: Path,
    flow_options: FlowOptions,
    flow_name: str | None = None,
) -> DocumentList:
    """Execute a single pipeline flow."""
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
    """Executes multiple pipeline flows sequentially."""
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
