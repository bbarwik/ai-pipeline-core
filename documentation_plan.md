# Documentation Plan for ai_pipeline_core

## Overview
This document lists all classes, methods, functions, and properties in ai_pipeline_core that require docstrings.

## ai_pipeline_core/__init__.py
- No functions/classes (only imports and exports)

## ai_pipeline_core/documents/__init__.py
- No functions/classes (only imports and exports)

## ai_pipeline_core/documents/document.py
### Functions
- `_is_str_list(content: Any) -> TypeGuard[list[str]]`
- `_is_basemodel_list(content: Any) -> TypeGuard[list[BaseModel]]`

### Classes
- `Document(BaseModel, ABC)`
  - `__init_subclass__(cls, **kwargs: Any) -> None`
  - `__init__(self, **data: Any) -> None`
  - `get_base_type(self) -> Literal["flow", "task", "temporary"]` (abstract)
  - `base_type` (property)
  - `is_flow` (property)
  - `is_task` (property)
  - `is_temporary` (property)
  - `get_expected_files(cls) -> list[str] | None` (classmethod)
  - `validate_file_name(cls, name: str) -> None` (classmethod)
  - `validate_name(cls, v: str) -> str` (validator)
  - `validate_content(cls, v: bytes) -> bytes` (validator)
  - `serialize_content(self, v: bytes) -> str` (serializer)
  - `id` (property)
  - `sha256` (cached_property)
  - `size` (property)
  - `detected_mime_type` (cached_property)
  - `mime_type` (property)
  - `is_text` (property)
  - `is_pdf` (property)
  - `is_image` (property)
  - `canonical_name(cls) -> str` (classmethod)
  - `text` (property) -> str
  - `as_yaml(self) -> Any`
  - `as_json(self) -> Any`
  - `as_pydantic_model(self, model_type) -> TModel | list[TModel]`
  - `as_markdown_list(self) -> list[str]`
  - `create(cls, name: str, ...) -> Self` (classmethod, overloaded)
  - `create_as_markdown_list(cls, name: str, description: str | None, items: list[str]) -> Self` (classmethod)
  - `create_as_json(cls, name: str, description: str | None, data: Any) -> Self` (classmethod)
  - `create_as_yaml(cls, name: str, description: str | None, data: Any) -> Self` (classmethod)
  - `serialize_model(self) -> dict[str, Any]`
  - `from_dict(cls, data: dict[str, Any]) -> Self` (classmethod)

## ai_pipeline_core/documents/document_list.py
### Classes
- `DocumentList(list[Document])`
  - `__init__(self, documents: list[Document] | None = None, validate_same_type: bool = False, validate_duplicates: bool = False) -> None`
  - `_validate_no_duplicates(self) -> None`
  - `_validate_no_description_files(self) -> None`
  - `_validate_types(self) -> None`
  - `_validate(self) -> None`
  - `append(self, document: Document) -> None`
  - `extend(self, documents: Iterable[Document]) -> None`
  - `insert(self, index: SupportsIndex, document: Document) -> None`
  - `__setitem__(self, index: Union[SupportsIndex, slice], value: Any) -> None`
  - `__iadd__(self, other: Any) -> Self`
  - `filter_by_type(self, document_type: type[Document]) -> DocumentList`
  - `filter_by_types(self, document_types: list[type[Document]]) -> DocumentList`
  - `get_by_name(self, name: str) -> Document | None`

## ai_pipeline_core/documents/flow_document.py
### Classes
- `FlowDocument(Document)`
  - `__init__(self, **data: Any) -> None`
  - `get_base_type(self) -> Literal["flow"]`

## ai_pipeline_core/documents/task_document.py
### Classes
- `TaskDocument(Document)`
  - `__init__(self, **data: Any) -> None`
  - `get_base_type(self) -> Literal["task"]`

## ai_pipeline_core/documents/temporary_document.py
### Classes
- `TemporaryDocument(Document)`
  - `get_base_type(self) -> Literal["temporary"]`

## ai_pipeline_core/documents/utils.py
### Functions
- `sanitize_url(url: str) -> str`
- `camel_to_snake(name: str) -> str`
- `canonical_name_key(obj_or_name: Type[Any] | str, *, max_parent_suffixes: int = 3, extra_suffixes: Iterable[str] = ()) -> str`

## ai_pipeline_core/documents/mime_type.py
### Functions
- `detect_mime_type(content: bytes, name: str) -> str`
- `mime_type_from_extension(name: str) -> str`
- `is_text_mime_type(mime_type: str) -> bool`
- `is_json_mime_type(mime_type: str) -> bool`
- `is_yaml_mime_type(mime_type: str) -> bool`
- `is_pdf_mime_type(mime_type: str) -> bool`
- `is_image_mime_type(mime_type: str) -> bool`

## ai_pipeline_core/exceptions.py
### Classes
- `PipelineCoreError(Exception)`
- `DocumentError(PipelineCoreError)`
- `DocumentValidationError(DocumentError)`
- `DocumentSizeError(DocumentValidationError)`
- `DocumentNameError(DocumentValidationError)`
- `LLMError(PipelineCoreError)`
- `PromptError(PipelineCoreError)`
- `PromptRenderError(PromptError)`
- `PromptNotFoundError(PromptError)`
- `MimeTypeError(DocumentError)`

## ai_pipeline_core/flow/__init__.py
- No functions/classes (only imports)

## ai_pipeline_core/flow/config.py
### Classes
- `FlowConfig(ABC)`
  - `__init_subclass__(cls, **kwargs: Any)`
  - `get_input_document_types(cls) -> list[type[FlowDocument]]` (classmethod)
  - `get_output_document_type(cls) -> type[FlowDocument]` (classmethod)
  - `has_input_documents(cls, documents: DocumentList) -> bool` (classmethod)
  - `get_input_documents(cls, documents: DocumentList) -> DocumentList` (classmethod)
  - `validate_output_documents(cls, documents: DocumentList) -> None` (classmethod)
  - `create_and_validate_output(cls, output: FlowDocument | list[FlowDocument] | DocumentList) -> DocumentList` (classmethod)

## ai_pipeline_core/flow/options.py
### Classes
- `FlowOptions(BaseSettings)`

## ai_pipeline_core/llm/__init__.py
- No functions/classes (only imports)

## ai_pipeline_core/llm/ai_messages.py
### Classes
- `AIMessages(list[AIMessageType])`
  - `get_last_message(self) -> AIMessageType`
  - `get_last_message_as_str(self) -> str`
  - `to_prompt(self) -> list[ChatCompletionMessageParam]`
  - `to_tracing_log(self) -> list[str]`
  - `get_prompt_cache_key(self, system_prompt: str | None = None) -> str`
  - `document_to_prompt(document: Document) -> list[ChatCompletionContentPartParam]` (staticmethod)

## ai_pipeline_core/llm/client.py
### Functions
- `_process_messages(context: AIMessages, messages: AIMessages, system_prompt: str | None = None) -> list[ChatCompletionMessageParam]`
- `_generate(model: str, messages: list[ChatCompletionMessageParam], completion_kwargs: dict[str, Any]) -> ModelResponse` (async)
- `_generate_with_retry(model: str, context: AIMessages, messages: AIMessages, options: ModelOptions) -> ModelResponse` (async)
- `generate(model: ModelName | str, *, context: AIMessages = AIMessages(), messages: AIMessages | str, options: ModelOptions = ModelOptions()) -> ModelResponse` (async)
- `generate_structured(model: ModelName | str, response_format: type[T], *, context: AIMessages = AIMessages(), messages: AIMessages | str, options: ModelOptions = ModelOptions()) -> StructuredModelResponse[T]` (async)

## ai_pipeline_core/llm/model_options.py
### Classes
- `ModelOptions(BaseModel)`
  - `to_openai_completion_kwargs(self) -> dict[str, Any]`

## ai_pipeline_core/llm/model_response.py
### Classes
- `ModelResponse(ChatCompletion)`
  - `__init__(self, chat_completion: ChatCompletion | None = None, **kwargs: Any) -> None`
  - `content` (property)
  - `set_model_options(self, options: dict[str, Any]) -> None`
  - `set_headers(self, headers: dict[str, str]) -> None`
  - `get_laminar_metadata(self) -> dict[str, str | int | float]`

- `StructuredModelResponse(ModelResponse, Generic[T])`
  - `__init__(self, chat_completion: ChatCompletion | None = None, parsed_value: T | None = None, **kwargs: Any) -> None`
  - `parsed` (property)

## ai_pipeline_core/llm/model_types.py
- Type alias only, no functions/classes

## ai_pipeline_core/logging/__init__.py
- No functions/classes (only imports)

## ai_pipeline_core/logging/logging_config.py
### Classes
- `LoggingConfig`
  - `__init__(self, config_path: Optional[Path] = None)`
  - `_get_default_config_path() -> Optional[Path]` (staticmethod)
  - `load_config(self) -> Dict[str, Any]`
  - `_get_default_config() -> Dict[str, Any]` (staticmethod)
  - `apply(self)`

### Functions
- `setup_logging(config_path: Optional[Path] = None, level: Optional[str] = None)`
- `get_pipeline_logger(name: str)`

## ai_pipeline_core/logging/logging_mixin.py
### Classes
- `LoggerMixin`
  - `logger` (cached_property)
  - `_get_run_logger(self)`
  - `log_debug(self, message: str, **kwargs: Any) -> None`
  - `log_info(self, message: str, **kwargs: Any) -> None`
  - `log_warning(self, message: str, **kwargs: Any) -> None`
  - `log_error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None`
  - `log_critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None`
  - `log_with_context(self, level: str, message: str, context: Dict[str, Any]) -> None`

- `StructuredLoggerMixin(LoggerMixin)`
  - `log_event(self, event: str, **kwargs: Any) -> None`
  - `log_metric(self, metric_name: str, value: float, unit: str = "", **tags: Any) -> None`
  - `log_span(self, operation: str, duration_ms: float, **attributes: Any) -> None`
  - `log_operation(self, operation: str, **context: Any) -> Generator[None, None, None]` (contextmanager)

- `PrefectLoggerMixin(StructuredLoggerMixin)`
  - `log_flow_start(self, flow_name: str, parameters: Dict[str, Any]) -> None`
  - `log_flow_end(self, flow_name: str, status: str, duration_ms: float) -> None`
  - `log_task_start(self, task_name: str, inputs: Dict[str, Any]) -> None`
  - `log_task_end(self, task_name: str, status: str, duration_ms: float) -> None`
  - `log_retry(self, operation: str, attempt: int, max_attempts: int, error: str) -> None`
  - `log_checkpoint(self, checkpoint_name: str, **data: Any) -> None`

## ai_pipeline_core/pipeline.py
### Functions
- `_callable_name(obj: Any, fallback: str) -> str`
- `pipeline_task(__fn: Callable[..., Coroutine[Any, Any, R_co]] | None = None, ...) -> _TaskLike[R_co] | Callable[...]`
- `pipeline_flow(__fn: _DocumentsFlowCallable[FO_contra] | None = None, ...) -> _FlowLike[FO_contra] | Callable[...]`

### Protocols
- `_TaskLike(Protocol[R_co])`
- `_DocumentsFlowCallable(Protocol[FO_contra])`
- `_FlowLike(Protocol[FO_contra])`

## ai_pipeline_core/prefect.py
- No functions/classes (only imports/exports)

## ai_pipeline_core/prompt_manager.py
### Classes
- `PromptManager`
  - `__init__(self, current_dir: str, prompts_dir: str = "prompts")`
  - `get(self, prompt_path: str, **kwargs: Any) -> str`

## ai_pipeline_core/settings.py
### Classes
- `Settings(BaseSettings)`

## ai_pipeline_core/simple_runner/__init__.py
- No functions/classes (only imports)

## ai_pipeline_core/simple_runner/cli.py
### Functions
- `_initialize_environment() -> None`
- `_running_under_pytest() -> bool`
- `run_cli(*, flows: FlowSequence, flow_configs: ConfigSequence, options_cls: Type[TOptions], initializer: InitializerFunc = None, trace_name: str | None = None) -> None`

## ai_pipeline_core/simple_runner/simple_runner.py
### Functions
- `load_documents_from_directory(base_dir: Path, document_types: Sequence[Type[FlowDocument]]) -> DocumentList`
- `save_documents_to_directory(base_dir: Path, documents: DocumentList) -> None`
- `run_pipeline(flow_func: Callable[..., Any], config: Type[FlowConfig], project_name: str, output_dir: Path, flow_options: FlowOptions, flow_name: str | None = None) -> DocumentList` (async)
- `run_pipelines(project_name: str, output_dir: Path, flows: FlowSequence, flow_configs: ConfigSequence, flow_options: FlowOptions, start_step: int = 1, end_step: int | None = None) -> None` (async)

## ai_pipeline_core/tracing.py
### Classes
- `TraceInfo(BaseModel)`
  - `get_observe_kwargs(self) -> dict[str, Any]`

### Functions
- `_initialise_laminar() -> None`
- `trace(func: Callable[P, R] | None = None, ...) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]`
