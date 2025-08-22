# AI Pipeline Core API Reference (v0.1.8)

## Version 0.1.8 Breaking Changes
- **Pipeline decorators now require async functions only** - `@pipeline_flow` and `@pipeline_task` will raise `TypeError` for sync functions
- **Document class name validation** - Subclasses starting with "Test" are rejected at definition time
- **FlowConfig validation** - `OUTPUT_DOCUMENT_TYPE` cannot be in `INPUT_DOCUMENT_TYPES`
- **Temperature field added** to `ModelOptions` for explicit temperature control

## Table of Contents

1. **Settings Configuration**
2. **Logging System**
3. **Document System**
4. **Flow Configuration**
5. **Pipeline Decorators**
6. **Prefect Utilities**
7. **LLM Module**
8. **Tracing**
9. **Utilities**
10. **Usage Patterns**
11. **Best Practices**

---

## 1. Settings Configuration

### `settings`
Singleton instance of `Settings` class providing environment-based configuration with Pydantic validation.

**Class:** `Settings(BaseSettings)`
- **Model Config:** `SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")`

**Properties:**
- `openai_base_url: str = ""` - Base URL for OpenAI-compatible API (LiteLLM proxy)
- `openai_api_key: str = ""` - API key for authentication
- `prefect_api_url: str = ""` - Prefect server URL
- `prefect_api_key: str = ""` - Prefect authentication key
- `lmnr_project_api_key: str = ""` - Laminar observability API key

**Usage:**
```python
from ai_pipeline_core import settings
# Access as: settings.openai_api_key
```

---

## 2. Logging System

### `get_pipeline_logger(name: str)`
Creates a Prefect-aware logger instance using Prefect's `get_logger`.

**Parameters:**
- `name: str` - Logger name (typically `__name__`)

**Returns:** Logger instance that automatically uses Prefect's run logger when in flow/task context

**Behavior:**
- Ensures logging is setup if not already initialized
- Returns Prefect logger instance

### `setup_logging(config_path: Path | None = None, level: str | None = None)`
Initializes logging configuration for the pipeline.

**Parameters:**
- `config_path: Path | None` - Optional YAML config file path
- `level: str | None` - Override log level (DEBUG, INFO, WARNING, ERROR)

**Behavior:**
- Applies logging configuration from file or defaults
- Sets Prefect environment variables if configured
- Overrides level for all pipeline loggers if provided

### `LoggingConfig`
Manages logging configuration from YAML files.

**Constructor:**
- `config_path: Path | None` - Path to YAML config file

**Methods:**
- `load_config() -> dict` - Load configuration from file or defaults
- `apply()` - Apply configuration using `logging.config.dictConfig`
- `_get_default_config() -> dict` - Returns default logging configuration

**Default Log Levels:**
```python
DEFAULT_LOG_LEVELS = {
    "ai_pipeline_core": "INFO",
    "ai_pipeline_core.documents": "INFO",
    "ai_pipeline_core.llm": "INFO",
    "ai_pipeline_core.flow": "INFO",
    "ai_pipeline_core.testing": "DEBUG",
}
```

### `LoggerMixin`
Mix-in class providing logging methods with automatic context detection.

**Cached Properties:**
- `logger` - Returns `get_run_logger()` in flow/task context, otherwise `get_logger()`

**Methods:**
- `log_debug(message: str, **kwargs)` - Debug level with extra kwargs
- `log_info(message: str, **kwargs)` - Info level with extra kwargs
- `log_warning(message: str, **kwargs)` - Warning level with extra kwargs
- `log_error(message: str, exc_info: bool = False, **kwargs)` - Error with optional exception
- `log_critical(message: str, exc_info: bool = False, **kwargs)` - Critical with optional exception
- `log_with_context(level: str, message: str, context: dict)` - Structured logging with context dict

**Private Methods:**
- `_get_run_logger()` - Attempts to get Prefect run logger, returns None if not in context

### `StructuredLoggerMixin`
Extends LoggerMixin with structured logging capabilities.

**Additional Methods:**
- `log_event(event: str, **kwargs)` - Log structured events with `structured=True` flag
- `log_metric(metric_name: str, value: float, unit: str = "", **tags)` - Log metrics with tags
- `log_span(operation: str, duration_ms: float, **attributes)` - Log operation spans
- `log_operation(operation: str, **context) -> Generator` - Context manager for timed operations

**Context Manager Behavior:**
- Logs start with debug level
- Logs completion with info level and duration
- Logs failure with error level, exception info, and duration
- Always re-raises exceptions

---

## 3. Document System

### `Document` (Abstract Base Class)
Base class for all document types. Cannot be instantiated directly.

**Class Variables:**
- `MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024` - 25MB default limit
- `DESCRIPTION_EXTENSION: ClassVar[str] = ".description.md"` - Reserved extension
- `MARKDOWN_LIST_SEPARATOR: ClassVar[str] = "\n\n---\n\n"` - Markdown list separator
- `FILES: ClassVar[type[StrEnum] | None] = None` - Optional filename restrictions

**Constructor Validation:**
- Prevents direct instantiation of abstract class
- **v0.1.8:** Prevents subclass names starting with "Test" (pytest conflict)
  - Raises `TypeError` at class definition time
  - Suggests alternatives: `SampleDocument`, `ExampleDocument`, `DemoDocument`, `MockDocument`

**Properties (Pydantic Fields):**
- `name: str` - Validated filename (no path traversal, no reserved extensions)
- `description: str | None` - Optional description
- `content: bytes` - Binary content (size-validated)

**Computed Properties:**
- `id: str` - First 6 chars of SHA256 hash (base32 encoded, uppercase)
- `sha256: str` - Full SHA256 hash (base32 encoded, uppercase)
- `size: int` - Content size in bytes
- `detected_mime_type: str` - MIME type from content + extension
- `mime_type: str` - Alias for detected_mime_type
- `is_text: bool` - True if text MIME type
- `is_pdf: bool` - True if PDF MIME type
- `is_image: bool` - True if image MIME type
- `base_type: Literal["flow", "task"]` - Alias for get_base_type()
- `is_flow: bool` - True if flow document
- `is_task: bool` - True if task document

**Abstract Methods:**
- `get_base_type() -> Literal["flow", "task"]` - Must be implemented by subclasses

**Instance Methods:**
- `as_text() -> str` - Decode as UTF-8 (raises ValueError if not text)
- `as_yaml() -> Any` - Parse using ruamel.yaml
- `as_json() -> Any` - Parse using json.loads
- `as_pydantic_model(model_type: type[T]) -> T` - Parse and validate as Pydantic model
- `as_markdown_list() -> list[str]` - Split by MARKDOWN_LIST_SEPARATOR
- `serialize_model() -> dict` - Serialize with metadata and smart encoding
- `from_dict(data: dict) -> Self` - Deserialize from dict with encoding support

**Class Methods:**
- `canonical_name() -> str` - Get canonical snake_case name
- `get_expected_files() -> list[str] | None` - Get allowed filenames from FILES enum
- `validate_file_name(name: str)` - Validate against FILES enum if defined
- `create(name: str, description: str | None, content: bytes | str | BaseModel | list[str] | Any) -> Self` - Smart factory method
- `create_as_markdown_list(name: str, description: str | None, items: list[str]) -> Self` - Create markdown with separators
- `create_as_json(name: str, description: str | None, data: Any) -> Self` - Create JSON with 2-space indent
- `create_as_yaml(name: str, description: str | None, data: Any) -> Self` - Create YAML with proper formatting

**Validators:**
- `validate_name` - Checks for path traversal, reserved extensions, whitespace
- `validate_content` - Enforces MAX_CONTENT_SIZE limit

**Serializers:**
- `serialize_content` - Encodes to UTF-8 or base64 for JSON serialization

### `FlowDocument` (Abstract Base Class)
Base for flow-specific documents. Persistent across Prefect flow runs.

**Inheritance:** All Document properties and methods
**Override:** `get_base_type()` returns `"flow"` (marked as `@final`)
**Constructor:** Prevents direct instantiation of abstract class

### `TaskDocument` (Abstract Base Class)
Base for task-specific documents. Temporary within task execution.

**Inheritance:** All Document properties and methods
**Override:** `get_base_type()` returns `"task"` (marked as `@final`)
**Constructor:** Prevents direct instantiation of abstract class

### `DocumentList`
Type-safe container extending `list[Document]` with validation.

**Constructor Parameters:**
- `documents: list[Document] | None = None` - Initial documents
- `validate_same_type: bool = False` - Enforce same document type
- `validate_duplicates: bool = False` - Prevent duplicate filenames

**Overridden Methods (with validation):**
- `append(document: Document)` - Add with validation
- `extend(documents: Iterable[Document])` - Add multiple
- `insert(index: int, document: Document)` - Insert at index
- `__setitem__(index: int | slice, value: Document | Iterable[Document])` - Set with validation
- `__iadd__(other: Any) -> Self` - In-place addition

**Custom Methods:**
- `filter_by_type(document_type: type[Document]) -> DocumentList` - Filter by exact type (uses `type(doc) is`)
- `filter_by_types(document_types: list[type[Document]]) -> DocumentList` - Filter by multiple types
- `get_by_name(name: str) -> Document | None` - Find by filename

**Private Validation Methods:**
- `_validate()` - Runs all validations
- `_validate_no_duplicates()` - Checks for duplicate names
- `_validate_no_description_files()` - Prevents DESCRIPTION_EXTENSION files
- `_validate_types()` - Ensures same type if required

### Utility Functions

#### `canonical_name_key(obj_or_name: type | str, max_parent_suffixes: int = 3, extra_suffixes: Iterable[str] = ()) -> str`
Converts class names to canonical snake_case by stripping suffixes.

**Algorithm:**
1. Extracts class name or uses string directly
2. Collects up to `max_parent_suffixes` from MRO (excluding object)
3. Adds `extra_suffixes` to removal list
4. Iteratively removes longest matching suffix
5. Converts result to snake_case

#### `sanitize_url(url: str) -> str`
Sanitizes URLs for use in filenames.

**Behavior:**
1. Removes protocol (http://, https://)
2. Replaces invalid chars with underscore
3. Collapses multiple underscores
4. Strips leading/trailing underscores and dots
5. Limits to 100 characters
6. Falls back to "unnamed" if empty

---

## 4. Flow Configuration

### `FlowConfig` (Abstract Base Class)
Type-safe configuration for Prefect flows.

**Class Variables (must be defined by subclasses):**
- `INPUT_DOCUMENT_TYPES: ClassVar[list[type[FlowDocument]]]` - Required input types
- `OUTPUT_DOCUMENT_TYPE: ClassVar[type[FlowDocument]]` - Output document type

**Validation (v0.1.8):**
- Enforced at class definition time via `__init_subclass__`
- `OUTPUT_DOCUMENT_TYPE` cannot be in `INPUT_DOCUMENT_TYPES` (prevents circular dependencies)
- Raises `TypeError` with clear message if validation fails

**Class Methods:**
- `get_input_document_types() -> list[type[FlowDocument]]` - Returns INPUT_DOCUMENT_TYPES
- `get_output_document_type() -> type[FlowDocument]` - Returns OUTPUT_DOCUMENT_TYPE
- `has_input_documents(documents: DocumentList) -> bool` - Check all inputs present (uses isinstance)
- `get_input_documents(documents: DocumentList) -> DocumentList` - Extract inputs (raises ValueError if missing)
- `validate_output_documents(documents: DocumentList)` - Assert all outputs match type

### `FlowOptions`
Pydantic settings for flow configuration.

**Base Class:** `BaseSettings`
**Model Config:** `SettingsConfigDict(frozen=True, extra="ignore")`

**Fields:**
- `core_model: ModelName | str = Field(default="gpt-5", description="Primary model for complex analysis and generation tasks.")`
- `small_model: ModelName | str = Field(default="gpt-5-mini", description="Fast, cost-effective model for simple tasks and orchestration.")`

---

## 5. Pipeline Decorators

### `@pipeline_task`
Decorator combining Prefect task with LMNR tracing. Requires async functions. Returns the actual Prefect Task object typed as `_TaskLike` Protocol.

**Overloads:**
1. Direct decoration: `@pipeline_task`
2. With parameters: `@pipeline_task(...parameters...)`

**Tracing Parameters:**
- `trace_level: TraceLevel = "always"` - "always", "debug", or "off"
- `trace_ignore_input: bool = False` - Ignore all inputs
- `trace_ignore_output: bool = False` - Ignore output
- `trace_ignore_inputs: list[str] | None = None` - Specific params to ignore
- `trace_input_formatter: Callable[..., str] | None = None` - Custom formatter
- `trace_output_formatter: Callable[..., str] | None = None` - Custom formatter

**Prefect Parameters (all standard task parameters):**
- `name: str | None = None`
- `description: str | None = None`
- `tags: Iterable[str] | None = None`
- `version: str | None = None`
- `cache_policy: CachePolicy | type[NotSet] = NotSet`
- `cache_key_fn: Callable[[TaskRunContext, dict], str | None] | None = None`
- `cache_expiration: timedelta | None = None`
- `task_run_name: TaskRunNameValueOrCallable | None = None`
- `retries: int | None = None` (defaults to 0 if None)
- `retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None`
- `retry_jitter_factor: float | None = None`
- `persist_result: bool | None = None`
- `result_storage: ResultStorage | str | None = None`
- `result_serializer: ResultSerializer | str | None = None`
- `result_storage_key: str | None = None`
- `cache_result_in_memory: bool = True`
- `timeout_seconds: int | float | None = None`
- `log_prints: bool | None = False`
- `refresh_cache: bool | None = None`
- `on_completion: list[StateHookCallable] | None = None`
- `on_failure: list[StateHookCallable] | None = None`
- `retry_condition_fn: RetryConditionCallable | None = None`
- `viz_return_value: bool | None = None`
- `asset_deps: list[str | Asset] | None = None`

**Implementation:**
1. Validates function is coroutine (raises TypeError if not)
2. Applies trace decorator if trace_level != "off"
3. Wraps with Prefect's native task decorator
4. Returns cast to `_TaskLike[R_co]`

### `@pipeline_flow`
Decorator for async document-processing flows with tracing. Returns the actual Prefect Flow object typed as `_FlowLike` Protocol with contravariant FlowOptions support.

**Required Signature:**
```python
async def flow_name(
    project_name: str,
    documents: DocumentList,
    flow_options: FlowOptions,  # or any subclass
    *args,
    **kwargs
) -> DocumentList
```

**Type Safety:** The decorator properly handles FlowOptions subclasses as contravariant, allowing flows to accept more specific option types.

**Parameters:** Similar to `@pipeline_task` but for flows, including:
- All tracing parameters
- `flow_run_name: Callable[[], str] | str | None = None`
- `retries: int | None = None` (defaults to 0)
- `retry_delay_seconds: int | float | None = None`
- `task_runner: TaskRunner[PrefectFuture[Any]] | None = None`
- `timeout_seconds: int | float | None = None`
- `validate_parameters: bool = True`
- Flow state hooks: `on_completion`, `on_failure`, `on_cancellation`, `on_crashed`, `on_running`

**Validation:**
1. Must be async function
2. Must have at least 3 parameters
3. Internal wrapper validates return type is DocumentList

---

## 6. Prefect Utilities

### `prefect_test_harness`
Direct export of `prefect.testing.utilities.prefect_test_harness` for testing.

**Usage:** Context manager for creating isolated Prefect test environment.

### `disable_run_logger`
Direct export of `prefect.logging.disable_run_logger` for disabling run logging.

**Usage:** Context manager to suppress Prefect run logs.

### Clean Decorators (from `ai_pipeline_core.prefect`)
The module `ai_pipeline_core.prefect` provides direct re-exports of Prefect's native decorators:
- `@flow` - Prefect's flow decorator without tracing (supports both sync and async)
- `@task` - Prefect's task decorator without tracing (supports both sync and async)

**Note:** These are NOT exported from the main `ai_pipeline_core` module. Import them explicitly:
```python
from ai_pipeline_core.prefect import flow, task
```

---

## 7. LLM Module

### `ModelName`
Type alias for supported model literals:
```python
ModelName: TypeAlias = Literal[
    # Core models
    "gemini-2.5-pro", "gpt-5", "grok-4",
    # Small models
    "gemini-2.5-flash", "gpt-5-mini", "grok-3-mini",
    # Search models
    "gemini-2.5-flash-search", "sonar-pro-search",
    "gpt-4o-search", "grok-3-mini-search"
]
```

### `AIMessageType`
Type alias: `str | Document | ModelResponse`

### `ModelOptions`
Pydantic model for LLM generation configuration.

**Fields:**
- `temperature: float | None = None` - Controls randomness (0-2)
- `system_prompt: str | None = None`
- `search_context_size: Literal["low", "medium", "high"] | None = None`
- `reasoning_effort: Literal["low", "medium", "high"] | None = None`
- `retries: int = 3`
- `retry_delay_seconds: int = 10`
- `timeout: int = 300`
- `service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None`
- `max_completion_tokens: int | None = None`
- `response_format: type[BaseModel] | None = None` - Used internally by generate_structured

**Methods:**
- `to_openai_completion_kwargs() -> dict` - Converts to OpenAI API kwargs with proper extra_body handling, including temperature

### `AIMessages`
Type-safe container extending `list[AIMessageType]`.

**Methods:**
- `get_last_message() -> AIMessageType` - Returns `self[-1]`
- `get_last_message_as_str() -> str` - Returns last as string (raises ValueError if not str)
- `to_prompt() -> list[ChatCompletionMessageParam]` - Converts to OpenAI format
- `to_tracing_log() -> list[str]` - For logging (excludes document content)
- `get_prompt_cache_key(system_prompt: str | None = None) -> str` - SHA256 hash for caching

**Static Methods:**
- `document_to_prompt(document: Document) -> list[ChatCompletionContentPartParam]`
  - Wraps documents in XML-like tags
  - Handles text, image, and PDF documents
  - Returns proper multimodal format for images/PDFs

### `ModelResponse`
Response from LLM extending OpenAI's `ChatCompletion`.

**Additional Fields:**
- `headers: dict[str, str] = Field(default_factory=dict)`
- `model_options: dict[str, Any] = Field(default_factory=dict)`

**Properties:**
- `content: str` - Returns `choices[0].message.content or ""`

**Methods:**
- `set_model_options(options: dict)` - Deep copy and store (removes "messages" key)
- `set_headers(headers: dict)` - Deep copy and store
- `get_laminar_metadata() -> dict[str, str | int | float]` - Extract observability metadata including LiteLLM headers, usage, costs

### `StructuredModelResponse[T]`
Generic response with parsed structured output.

**Additional Fields:**
- `_parsed_value: T | None` - Private field for parsed value

**Properties:**
- `parsed: T` - Returns parsed Pydantic instance (raises ValueError if None)

**Constructor:**
- Handles both regular ChatCompletion and ParsedChatCompletion
- Extracts parsed value from completion if available

### Core Generation Functions

#### `generate`
```python
@trace(ignore_inputs=["context"])
async def generate(
    model: ModelName | str,
    *,
    context: AIMessages = AIMessages(),
    messages: AIMessages | str,
    options: ModelOptions = ModelOptions()
) -> ModelResponse
```

**Features:**
- Automatic retry with exponential backoff
- Context caching (2-minute TTL via cache_control)
- LMNR span creation and metadata
- Automatic prompt processing
- Timeout and error handling

**Internal Implementation:**
1. `_process_messages` - Formats messages with system prompt and cache control
2. `_generate` - Makes actual API call using AsyncOpenAI
3. `_generate_with_retry` - Handles retry logic with cache disabling on non-timeout errors

#### `generate_structured`
```python
@trace(ignore_inputs=["context"])
async def generate_structured(
    model: ModelName | str,
    response_format: type[T],
    *,
    context: AIMessages = AIMessages(),
    messages: AIMessages | str,
    options: ModelOptions = ModelOptions()
) -> StructuredModelResponse[T]
```

**Additional Behavior:**
- Sets `options.response_format` to provided type
- Uses `client.chat.completions.parse` for structured output
- Extracts and validates parsed value
- Returns StructuredModelResponse with typed parsed property

---

## 8. Tracing

### `TraceLevel`
Type alias: `Literal["always", "debug", "off"]`

### `TraceInfo`
Pydantic model for trace metadata.

**Fields:**
- `session_id: str | None = None`
- `user_id: str | None = None`
- `metadata: dict[str, str] = {}`
- `tags: list[str] = []`

**Methods:**
- `get_observe_kwargs() -> dict` - Converts to Laminar kwargs with env var fallbacks

### `@trace` Decorator
Adds Laminar observability to functions (sync or async).

**Overloads:**
1. Direct: `@trace`
2. With params: `@trace(...parameters...)`

**Parameters:**
- `func: Callable | None = None` - Function to trace
- `level: TraceLevel = "always"` - When to trace
- `name: str | None = None` - Custom span name (defaults to function name)
- `session_id: str | None = None` - Override session
- `user_id: str | None = None` - Override user
- `metadata: dict[str, Any] | None = None` - Additional metadata
- `tags: list[str] | None = None` - Additional tags
- `span_type: str | None = None` - Span type for Laminar
- `ignore_input: bool = False` - Don't log inputs
- `ignore_output: bool = False` - Don't log outputs
- `ignore_inputs: list[str] | None = None` - Specific params to ignore
- `input_formatter: Callable[..., str] | None = None` - Custom formatter
- `output_formatter: Callable[..., str] | None = None` - Custom formatter
- `ignore_exceptions: bool = False` - Don't trace exceptions
- `preserve_global_context: bool = True` - Maintain global context

**Behavior:**
- "off": Returns function unchanged
- "debug": Only traces when LMNR_DEBUG != "true"
- "always": Always traces
- Initializes Laminar once per process
- Injects TraceInfo if function accepts it
- Preserves function signature
- Handles both sync and async functions

**Internal Implementation:**
- `_initialise_laminar()` - One-time Laminar setup with disabled OpenAI instrument
- `_prepare_and_get_observe_params()` - Runtime parameter preparation
- Creates sync or async wrapper based on function type

---

## 9. Utilities

### `PromptManager`
Jinja2-based prompt template manager.

**Constructor:**
- `current_dir: str` - Required: calling module's `__file__` path
- `prompts_dir: str = "prompts"` - Directory name to search

**Search Algorithm:**
1. Current directory's `prompts/` subdirectory
2. Current directory itself
3. Parent directories' `prompts/` (up to 4 levels while `__init__.py` exists)

**Properties:**
- `search_paths: list[Path]` - All discovered search paths
- `env: jinja2.Environment` - Configured Jinja2 environment

**Jinja2 Configuration:**
- `loader`: FileSystemLoader with all search paths
- `trim_blocks`: True
- `lstrip_blocks`: True
- `autoescape`: False (for prompt engineering)

**Methods:**
- `get(prompt_path: str, **kwargs) -> str` - Render template
  - Auto-adds `.jinja2` or `.jinja` extension if missing
  - Raises `PromptNotFoundError` if not found
  - Raises `PromptRenderError` on template errors
  - Handles multiple error types with proper re-raising

---

## 10. Usage Patterns

### Creating Custom Documents
```python
from ai_pipeline_core import FlowDocument, TaskDocument
from enum import StrEnum

class AllowedFiles(StrEnum):
    CONFIG = "config.yaml"
    DATA = "data.json"

class InputDocument(FlowDocument):
    FILES = AllowedFiles  # Optional filename restriction

class TempDocument(TaskDocument):
    pass  # Temporary processing document
```

### Document Creation Patterns
```python
from ai_pipeline_core import FlowDocument
from pydantic import BaseModel

class MyDoc(FlowDocument):
    pass

# Smart creation based on extension and content type
doc1 = MyDoc.create("data.json", None, {"key": "value"})  # JSON
doc2 = MyDoc.create("config.yaml", None, {"host": "localhost"})  # YAML
doc3 = MyDoc.create("text.txt", None, "Hello")  # UTF-8
doc4 = MyDoc.create("sections.md", None, ["Part 1", "Part 2"])  # Markdown list

# Direct creation methods
config = BaseModel(...)
doc5 = MyDoc.create_as_json("output.json", "Results", config)
doc6 = MyDoc.create_as_yaml("settings.yaml", None, {"debug": True})
```

### Implementing Flows
```python
from ai_pipeline_core import pipeline_flow, FlowConfig, FlowOptions, DocumentList

class MyFlowConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [InputDocument]
    OUTPUT_DOCUMENT_TYPE = OutputDocument

class MyFlowOptions(FlowOptions):
    temperature: float = 0.7
    batch_size: int = 100

@pipeline_flow(trace_level="always", retries=3)
async def my_flow(
    project_name: str,
    documents: DocumentList,
    flow_options: MyFlowOptions
) -> DocumentList:
    inputs = MyFlowConfig.get_input_documents(documents)
    # Process...
    outputs = DocumentList([output_doc])
    MyFlowConfig.validate_output_documents(outputs)
    return outputs
```

### Using LLM Generation
```python
from ai_pipeline_core import (
    generate, generate_structured,
    AIMessages, ModelOptions, ModelName
)
from pydantic import BaseModel

# Text generation with context caching
response = await generate(
    "gpt-5",
    context=AIMessages([large_document]),  # Cached for 2 minutes
    messages=AIMessages(["Analyze this"]),  # Dynamic
    options=ModelOptions(
        system_prompt="You are an analyst",
        max_completion_tokens=4000,
        retries=5
    )
)
text = response.content

# Structured generation
class Analysis(BaseModel):
    summary: str
    score: float
    tags: list[str]

structured = await generate_structured(
    "gpt-5-mini",
    Analysis,
    messages="Analyze the sentiment",
    options=ModelOptions(timeout=60)
)
result: Analysis = structured.parsed
```

### Prompt Management
```python
from ai_pipeline_core import PromptManager

# At module level
prompts = PromptManager(__file__)

# In function
async def analyze(doc: Document):
    prompt = prompts.get(
        "analysis.jinja2",  # Extension optional
        document=doc,
        temperature=0.7
    )
    response = await generate(...)
```

### Tracing Patterns
```python
from ai_pipeline_core import trace, TraceInfo, TraceLevel

@trace(
    level="always",
    name="custom_operation",
    ignore_inputs=["sensitive_data"],
    span_type="TASK"
)
async def process_data(
    data: str,
    sensitive_data: str,
    trace_info: TraceInfo  # Automatically injected
) -> str:
    # trace_info contains session_id, user_id, metadata, tags
    return result

# For testing
@trace(level="debug")  # Only traces when LMNR_DEBUG != "true"
async def test_function():
    pass
```

---

## 11. Best Practices

### Document Naming Rules
- **NEVER** create Document subclasses starting with "Test" (pytest conflict)
- Use descriptive names: `ReportDocument`, not `TestDocument`
- Alternative prefixes: `Sample`, `Example`, `Demo`, `Mock`

### Async Requirements (v0.1.8 Breaking Change)
- `@pipeline_flow` requires `async def` functions only (will raise TypeError for sync)
- `@pipeline_task` requires `async def` functions only (will raise TypeError for sync)
- Clean `@flow` and `@task` from `ai_pipeline_core.prefect` support both sync and async (Prefect native)

### Performance Optimization
- Use `context` parameter in generate() for large, static content (2-minute cache)
- Use `messages` parameter for dynamic queries
- Batch document processing using Prefect's `.map()`

### Error Handling
- LLM functions include automatic retry with exponential backoff
- Wrap in `try/except LLMError` for transient issues
- For `generate_structured`, also catch `ValidationError`

### Type Safety
- Always use `DocumentList` at flow boundaries
- Define concrete Document classes for each file type
- Use `FlowConfig` to enforce input/output contracts
- Extend `FlowOptions` for custom configuration

### Observability
- Set `LMNR_PROJECT_API_KEY` for production tracing
- Use `trace_level="debug"` for development
- Use `ignore_inputs` for sensitive parameters
- Environment variables: `LMNR_SESSION_ID`, `LMNR_USER_ID`

### Testing
- Use `@trace(level="debug")` in tests
- Mock `generate` and `generate_structured` for unit tests
- Mark integration tests with `@pytest.mark.integration`
- Use `prefect_test_harness` for flow testing

### Common Pitfalls to Avoid
- Don't import `logging` directly - use `get_pipeline_logger()`
- Don't pass raw strings/bytes between tasks - use Documents
- Don't hardcode model names - pass via FlowOptions
- Don't manually serialize Pydantic models - use `Document.create()`
- Don't mix pipeline decorators with clean Prefect decorators
