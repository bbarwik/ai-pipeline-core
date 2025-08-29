# ai-pipeline-core API Reference

## Navigation Guide

**For Humans:**
- Use `grep -n '^##' API.md` to list all main sections with line numbers
- Use `grep -n '^###' API.md` to list all classes and functions
- Use `grep -n '^####' API.md` to list all methods and properties
- Search for specific features: `grep -n -i "ClassName" API.md` or `grep -n -i "function_name" API.md`

**For AI Assistants:**
- Use the Grep tool with pattern `^##` to list all module sections (e.g., `^## ai_pipeline_core.module`)
- Use pattern `^###` to find all classes and functions (e.g., `### ClassName`, `### function_name`)
- Use pattern `^####` to find all methods (e.g., `#### ClassName.method_name`)
- For specific lookups, use patterns like `class AIMessages` or `def generate` with output_mode="content" and -n=true for line numbers
- Use -C flag (context lines) to see surrounding content: `pattern="AIMessages", -C=5`
- Navigate directly to line numbers using Read tool with offset parameter once you know the location


## ai_pipeline_core

AI Pipeline Core - Production-ready framework for building AI pipelines with LLMs.

AI Pipeline Core is a high-performance async framework for building type-safe AI pipelines.
It combines document processing, LLM integration, and workflow orchestration into a unified
system designed for production use.

The framework enforces best practices through strong typing (Pydantic), automatic retries,
cost tracking, and distributed tracing. All I/O operations are async for maximum throughput.

Core Capabilities:
    - **Document Processing**: Type-safe handling of text, JSON, YAML, PDFs, and images
    - **LLM Integration**: Unified interface to any model via LiteLLM with caching
    - **Structured Output**: Type-safe generation with Pydantic model validation
    - **Workflow Orchestration**: Prefect-based flows and tasks with retries
    - **Observability**: Distributed tracing via Laminar (LMNR) for debugging
    - **Local Development**: Simple runner for testing without infrastructure

Quick Start:
    >>> from ai_pipeline_core import pipeline_flow, FlowDocument, DocumentList
    >>> from ai_pipeline_core.flow import FlowOptions
    >>> from ai_pipeline_core.llm import generate
    >>>
    >>> class OutputDoc(FlowDocument):
    ...     '''Analysis result document.'''
    >>>
    >>> @pipeline_flow
    >>> async def analyze_flow(
    ...     project_name: str,
    ...     documents: DocumentList,
    ...     flow_options: FlowOptions
    ... ) -> DocumentList:
    ...     response = await generate(
    ...         model="gpt-5",
    ...         messages=documents[0].text
    ...     )
    ...     result = OutputDoc.create(
    ...         name="analysis.txt",
    ...         content=response.content
    ...     )
    ...     return DocumentList([result])

Environment Variables (when using LiteLLM proxy):
    - OPENAI_BASE_URL: LiteLLM proxy endpoint (e.g., http://localhost:4000)
    - OPENAI_API_KEY: API key for LiteLLM proxy

Optional Environment Variables:
    - PREFECT_API_URL: Prefect server for orchestration
    - LMNR_PROJECT_API_KEY: Laminar (LMNR) API key for tracing


## ai_pipeline_core.exceptions

Exception hierarchy for AI Pipeline Core.

This module defines the exception hierarchy used throughout the AI Pipeline Core library.
All exceptions inherit from PipelineCoreError, providing a consistent error handling interface.

### PipelineCoreError

```python
class PipelineCoreError(Exception)
```

Base exception for all AI Pipeline Core errors.

### DocumentError

```python
class DocumentError(PipelineCoreError)
```

Base exception for document-related errors.

### DocumentValidationError

```python
class DocumentValidationError(DocumentError)
```

Raised when document validation fails.

### DocumentSizeError

```python
class DocumentSizeError(DocumentValidationError)
```

Raised when document content exceeds MAX_CONTENT_SIZE limit.

### DocumentNameError

```python
class DocumentNameError(DocumentValidationError)
```

Raised when document name contains invalid characters or patterns.

### LLMError

```python
class LLMError(PipelineCoreError)
```

Raised when LLM generation fails after all retries.

### PromptError

```python
class PromptError(PipelineCoreError)
```

Base exception for prompt template errors.

### PromptRenderError

```python
class PromptRenderError(PromptError)
```

Raised when Jinja2 template rendering fails.

### PromptNotFoundError

```python
class PromptNotFoundError(PromptError)
```

Raised when prompt template file is not found in search paths.

### MimeTypeError

```python
class MimeTypeError(DocumentError)
```

Raised when MIME type detection or validation fails.


## ai_pipeline_core.llm.model_response

Model response structures for LLM interactions.

Provides enhanced response classes that wrap OpenAI API responses
with additional metadata, cost tracking, and structured output support.

### ModelResponse

```python
class ModelResponse(ChatCompletion)
```

Enhanced response wrapper for LLM text generation.

Structurally compatible with OpenAI ChatCompletion response format. All LLM provider
responses are normalized to this format by LiteLLM proxy, ensuring consistent
interface across providers (OpenAI, Anthropic, Google, Grok, etc.).

Additional Attributes:
headers: HTTP response headers including cost information. Only populated
when using our client; will be empty dict if deserializing from JSON.
model_options: Configuration used for this generation.

Key Properties:
content: Quick access to generated text content.
usage: Token usage statistics (inherited).
model: Model identifier used (inherited).
id: Unique response ID (inherited).

**Example**:

  >>> from ai_pipeline_core.llm import generate
  >>> response = await generate("gpt-5", messages="Hello")
  >>> print(response.content)  # Generated text
  >>> print(response.usage.total_tokens)  # Token count
  >>> print(response.headers.get("x-litellm-response-cost"))  # Cost

**Notes**:

  This class maintains full compatibility with ChatCompletion
  while adding pipeline-specific functionality.

#### ModelResponse.content

```python
@property
def content(self) -> str
```

Get the generated text content.

Convenience property for accessing the first choice's message
content. Returns empty string if no content available.

**Returns**:

  Generated text from the first choice, or empty string.

**Example**:

  >>> response = await generate("gpt-5", messages="Hello")
  >>> text = response.content  # Direct access to generated text

### StructuredModelResponse

```python
class StructuredModelResponse(ModelResponse, Generic[T])
```

Response wrapper for structured/typed LLM output.

Structurally compatible with OpenAI ChatCompletion response format. Extends ModelResponse
with type-safe access to parsed Pydantic model instances.

Type Parameter:
T: The Pydantic model type for the structured output.

Additional Features:
- Type-safe access to parsed Pydantic model
- Automatically parses structured JSON output from model response
- All features of ModelResponse (cost, metadata, etc.)

**Example**:

  >>> from pydantic import BaseModel
  >>> from ai_pipeline_core.llm import generate_structured
  >>>
  >>> class Analysis(BaseModel):
  ...     sentiment: float
  ...     summary: str
  >>>
  >>> response = await generate_structured(
  ...     "gpt-5",
  ...     response_format=Analysis,
  ...     messages="Analyze: ..."
  ... )
  >>>
  >>> # Type-safe access
  >>> analysis: Analysis = response.parsed
  >>> print(f"Sentiment: {analysis.sentiment}")
  >>>
  >>> # Still have access to metadata
  >>> print(f"Tokens used: {response.usage.total_tokens}")

**Notes**:

  The parsed property provides type-safe access to the
  validated Pydantic model instance.

#### StructuredModelResponse.parsed

```python
@property
def parsed(self) -> T
```

Get the parsed Pydantic model instance.

Provides type-safe access to the structured output that was
generated according to the specified schema.

**Returns**:

  Validated instance of the Pydantic model type T.

**Raises**:

- `ValueError` - If no parsed content is available (should
  not happen in normal operation).

**Example**:

  >>> class UserInfo(BaseModel):
  ...     name: str
  ...     age: int
  >>>
  >>> response: StructuredModelResponse[UserInfo] = ...
  >>> user = response.parsed  # Type is UserInfo
  >>> print(f"{user.name} is {user.age} years old")

  Type Safety:
  The return type matches the type parameter T, providing
  full IDE support and type checking.

**Notes**:

  This property should always return a value for properly
  generated structured responses. ValueError indicates an
  internal error.


## ai_pipeline_core.llm.ai_messages

AI message handling for LLM interactions.

Provides AIMessages container for managing conversations with mixed content types
including text, documents, and model responses.

### AIMessageType

```python
AIMessageType = str | Document | ModelResponse
```

Type for messages in AIMessages container.

Represents the allowed types for conversation messages:
- str: Plain text messages
- Document: Structured document content
- ModelResponse: LLM generation responses

### AIMessages

```python
class AIMessages(list[AIMessageType])
```

Container for AI conversation messages supporting mixed types.

This class extends list to manage conversation messages between user
and AI, supporting text, Document objects, and ModelResponse instances.
Messages are converted to OpenAI-compatible format for LLM interactions.

Conversion Rules:
- str: Becomes {"role": "user", "content": text}
- Document: Becomes {"role": "user", "content": structured_content}
- ModelResponse: Becomes {"role": "assistant", "content": response.content}

**Example**:

  >>> from ai_pipeline_core.llm import generate
  >>> messages = AIMessages()
  >>> messages.append("What is the capital of France?")
  >>> response = await generate("gpt-5", messages=messages)
  >>> messages.append(response)  # Add the actual response
  >>> prompt = messages.to_prompt()  # Convert to OpenAI format

#### AIMessages.get_last_message

```python
def get_last_message(self) -> AIMessageType
```

Get the last message in the conversation.

**Returns**:

  The last message in the conversation, which can be a string,
  Document, or ModelResponse.

#### AIMessages.get_last_message_as_str

```python
def get_last_message_as_str(self) -> str
```

Get the last message as a string, raising if not a string.

**Returns**:

  The last message as a string.

**Raises**:

- `ValueError` - If the last message is not a string.

#### AIMessages.to_prompt

```python
def to_prompt(self) -> list[ChatCompletionMessageParam]
```

Convert AIMessages to OpenAI-compatible format.

Transforms the message list into the format expected by OpenAI API.
Each message type is converted according to its role and content.

**Returns**:

  List of ChatCompletionMessageParam dicts with 'role' and 'content' keys.
  Ready to be passed to generate() or OpenAI API directly.

**Raises**:

- `ValueError` - If message type is not supported.

**Example**:

  >>> messages = AIMessages(["Hello", response, "Follow up"])
  >>> prompt = messages.to_prompt()
  >>> # Result: [
  >>> #   {"role": "user", "content": "Hello"},
  >>> #   {"role": "assistant", "content": "..."},
  >>> #   {"role": "user", "content": "Follow up"}
  >>> # ]


## ai_pipeline_core.llm.model_types

### ModelName

```python
ModelName: TypeAlias = Literal[
    # Core models
    "gemini-2.5-pro",
    "gpt-5",
    "grok-4",
    # Small models
    "gemini-2.5-flash",
    "gpt-5-mini",
    "grok-3-mini",
    # Search models
    "gemini-2.5-flash-search",
    "sonar-pro-search",
    "gpt-4o-search",
    "grok-3-mini-search",
]
```

Type-safe model name identifiers.

Provides compile-time validation and IDE autocompletion for supported
language model names. Used throughout the library to prevent typos
and ensure only valid models are referenced.

Model categories:
Core models (gemini-2.5-pro, gpt-5, grok-4):
High-capability models for complex tasks requiring deep reasoning,
nuanced understanding, or creative generation.

Small models (gemini-2.5-flash, gpt-5-mini, grok-3-mini):
Efficient models optimized for speed and cost, suitable for
simpler tasks or high-volume processing.

Search models (*-search suffix):
Models with integrated web search capabilities for retrieving
and synthesizing current information.

Extending with custom models:
The generate functions accept any string, not just ModelName literals.
To add custom models for type safety:
1. Create a new type alias: CustomModel = Literal["my-model"]
2. Use Union: model: ModelName | CustomModel = "my-model"
3. Or simply use strings: model = "any-model-via-litellm"

**Example**:

  >>> from ai_pipeline_core.llm import generate, ModelName
  >>>
  >>> # Type-safe model selection
  >>> model: ModelName = "gpt-5"  # IDE autocomplete works
  >>> response = await generate(model, messages="Hello")
  >>>
  >>> # Also accepts string for custom models
  >>> response = await generate("custom-model-v2", messages="Hello")
  >>>
  >>> # Custom type safety
  >>> from typing import Literal
  >>> MyModel = Literal["company-llm-v1"]
  >>> model: ModelName | MyModel = "company-llm-v1"

**Notes**:

  While the type alias provides suggestions for common models,
  the generate functions also accept string literals to support
  custom or newer models accessed via LiteLLM proxy.

**See Also**:

  - ai_pipeline_core.llm.generate: Main generation function
  - ai_pipeline_core.llm.ModelOptions: Model configuration


## ai_pipeline_core.llm.model_options

Configuration options for LLM generation.

Provides the ModelOptions class for configuring model behavior,
retry logic, and advanced features like web search and reasoning.

### ModelOptions

```python
class ModelOptions(BaseModel)
```

Configuration options for LLM generation requests.

ModelOptions encapsulates all configuration parameters for model
generation, including model behavior settings, retry logic, and
advanced features. All fields are optional with sensible defaults.

**Attributes**:

- `temperature` - Controls randomness in generation (0.0-2.0).
  Lower values = more deterministic, higher = more creative.
  If None, the parameter is omitted from the API call,
  causing the provider to use its own default (often 1.0).

- `system_prompt` - System-level instructions for the model.
  Sets the model's behavior and persona.

- `search_context_size` - Web search result depth for search-enabled models.
  Literal["low", "medium", "high"] | None
- `"low"` - Minimal context (~1-2 results)
- `"medium"` - Moderate context (~3-5 results)
- `"high"` - Extensive context (~6+ results)

- `reasoning_effort` - Reasoning intensity for models with explicit reasoning.
  Literal["low", "medium", "high"] | None
- `"low"` - Quick reasoning
- `"medium"` - Balanced reasoning
- `"high"` - Deep, thorough reasoning

- `retries` - Number of retry attempts on failure (default: 3).

- `retry_delay_seconds` - Seconds to wait between retries (default: 10).

- `timeout` - Maximum seconds to wait for response (default: 300).

- `service_tier` - OpenAI API tier selection for performance/cost trade-offs.
- `"auto"` - Let API choose
- `"default"` - Standard tier
- `"flex"` - Flexible (cheaper, may be slower)
- `"scale"` - Scaled performance
- `"priority"` - Priority processing
- `Note` - Only OpenAI models support this. Other providers
  (Anthropic, Google, Grok) silently ignore this parameter.

- `max_completion_tokens` - Maximum tokens to generate.
  None uses model default.

- `response_format` - Pydantic model class for structured output.
  Pass a Pydantic model; the client converts it to JSON Schema.
  Set automatically by generate_structured(). Provider support varies.

**Example**:

  >>> # Basic configuration
  >>> options = ModelOptions(
  ...     temperature=0.7,
  ...     max_completion_tokens=1000
  ... )
  >>>
  >>> # With system prompt
  >>> options = ModelOptions(
  ...     system_prompt="You are a helpful coding assistant",
  ...     temperature=0.3  # Lower for code generation
  ... )
  >>>
  >>> # For search-enabled models
  >>> options = ModelOptions(
  ...     search_context_size="high",  # Get more search results
  ...     max_completion_tokens=2000
  ... )
  >>>
  >>> # For reasoning models (O1 series)
  >>> options = ModelOptions(
  ...     reasoning_effort="high",  # Deep reasoning
  ...     timeout=600  # More time for complex reasoning
  ... )

**Notes**:

  - Not all options apply to all models
  - search_context_size only works with search models
  - reasoning_effort only works with O1-style models
  - response_format is set internally by generate_structured()


## ai_pipeline_core.llm.client

LLM client implementation for AI model interactions.

This module provides the core functionality for interacting with language models
through a unified interface. It handles retries, caching, structured outputs,
and integration with various LLM providers via LiteLLM.

Key functions:
- generate(): Text generation with optional context caching
- generate_structured(): Type-safe structured output generation

### generate

```python
@trace(ignore_inputs=["context"])
async def generate(model: ModelName | str, *, context: AIMessages | None = None, messages: AIMessages | str, options: ModelOptions | None = None) -> ModelResponse
```

Generate text response from a language model.

Main entry point for LLM text generation with smart context caching.
The context/messages split enables efficient token usage by caching
expensive static content separately from dynamic queries.

**Arguments**:

- `model` - Model to use (e.g., "gpt-5", "gemini-2.5-pro", "grok-4").
  Can be ModelName literal or any string for custom models.
- `context` - Static context to cache (documents, examples, instructions).
  Defaults to None (empty context). Cached for 120 seconds.
- `messages` - Dynamic messages/queries. Can be a single string or AIMessages.
  Converted to AIMessages if string.
- `options` - Model configuration (temperature, retries, timeout, etc.).
  Defaults to None (uses ModelOptions() with standard settings).

**Returns**:

  ModelResponse containing:
  - Generated text content
  - Usage statistics
  - Cost information (if available)
  - Model metadata

**Raises**:

- `ValueError` - If model is empty or messages are invalid.
- `LLMError` - If generation fails after all retries.

**Example**:

  >>> # Simple generation
  >>> response = await generate(
  ...     model="gpt-5",
  ...     messages="Explain quantum computing"
  ... )
  >>> print(response.content)

  >>> # With context caching
  >>> context = AIMessages([large_document])
  >>> response = await generate(
  ...     model="gemini-2.5-pro",
  ...     context=context,  # Cached for efficiency
  ...     messages="Summarize the key points",
  ...     options=ModelOptions(temperature=0.7)
  ... )

  >>> # Multi-turn conversation
  >>> messages = AIMessages([
  ...     "What is Python?",
  ...     previous_response,
  ...     "Can you give an example?"
  ... ])
  >>> response = await generate("gpt-5", messages=messages)

  Performance:
  - Context caching saves ~50-90% tokens on repeated calls
  - First call: full token cost
  - Subsequent calls (within 120s): only messages tokens
  - Default retry delay is 10s (configurable via ModelOptions.retry_delay_seconds)

  Caching:
  Context is cached by the LLM provider with a fixed 120-second TTL.
  The cache is based on the content hash and reduces token usage significantly.

**Notes**:

  - Context argument is ignored by the tracer to avoid recording large data
  - All models are accessed via LiteLLM proxy
  - Automatic retry with configurable delay between attempts
  - Cost tracking via response headers

**See Also**:

  - generate_structured: For typed/structured output
  - AIMessages: Message container with document support
  - ModelOptions: Configuration options

### generate_structured

```python
@trace(ignore_inputs=["context"])
async def generate_structured(model: ModelName | str, response_format: type[T], *, context: AIMessages | None = None, messages: AIMessages | str, options: ModelOptions | None = None) -> StructuredModelResponse[T]
```

Generate structured output conforming to a Pydantic model.

Type-safe generation that returns validated Pydantic model instances.
Uses OpenAI's structured output feature for guaranteed schema compliance.

**Arguments**:

- `model` - Model to use (must support structured output).
- `response_format` - Pydantic model class defining the output schema.
  The model will generate JSON matching this schema.
- `context` - Static context to cache (documents, schemas, examples).
  Defaults to None (empty AIMessages).
- `messages` - Dynamic prompts/queries. String or AIMessages.
- `options` - Model configuration. response_format is set automatically.

**Returns**:

  StructuredModelResponse[T] containing:
  - parsed: Validated instance of response_format class
  - All fields from regular ModelResponse (content, usage, etc.)

**Raises**:

- `TypeError` - If response_format is not a Pydantic model class.
- `ValueError` - If model doesn't support structured output or no parsed content returned.
- `LLMError` - If generation fails after retries.
- `ValidationError` - If response cannot be parsed into response_format.

**Example**:

  >>> from pydantic import BaseModel, Field
  >>>
  >>> class Analysis(BaseModel):
  ...     summary: str = Field(description="Brief summary")
  ...     sentiment: float = Field(ge=-1, le=1)
  ...     key_points: list[str] = Field(max_length=5)
  >>>
  >>> response = await generate_structured(
  ...     model="gpt-5",
  ...     response_format=Analysis,
  ...     messages="Analyze this product review: ..."
  ... )
  >>>
  >>> analysis = response.parsed  # Type: Analysis
  >>> print(f"Sentiment: {analysis.sentiment}")
  >>> for point in analysis.key_points:
  ...     print(f"- {point}")

  Supported models:
  Support varies by provider and model. Generally includes:
  - OpenAI: GPT-4 and newer models
  - Anthropic: Claude 3+ models
  - Google: Gemini Pro models
  Check provider documentation for specific model support.

  Performance:
  - Structured output may use more tokens than free text
  - Complex schemas increase generation time
  - Validation overhead is minimal (Pydantic is fast)

**Notes**:

  - Pydantic model is converted to JSON Schema for the API
  - The model generates JSON matching the schema
  - Validation happens automatically via Pydantic
  - Use Field() descriptions to guide generation

**See Also**:

  - generate: For unstructured text generation
  - ModelOptions: Configuration including response_format
  - StructuredModelResponse: Response wrapper with .parsed property


## ai_pipeline_core.prompt_manager

Jinja2-based prompt template management system.

This module provides the PromptManager class for loading and rendering
Jinja2 templates used as prompts for language models. It implements a
smart search strategy that looks for templates in both local and shared
directories.

Search strategy:
1. Local directory (same as calling module)
2. Local 'prompts' subdirectory
3. Parent 'prompts' directories (up to package boundary)

Key features:
- Automatic template discovery
- Jinja2 template rendering with context
- Smart path resolution (.jinja2/.jinja extension handling)
- Clear error messages for missing templates

**Example**:

  >>> from ai_pipeline_core.prompt_manager import PromptManager
  >>>
  >>> # In your module file
  >>> pm = PromptManager(__file__)
  >>>
  >>> # Render a template
  >>> prompt = pm.get(
  ...     "analyze.jinja2",
  ...     document=doc,
  ...     instructions="Extract key points"
  ... )

  Template organization:
  project/
  ├── my_module.py        # Can use local templates
  ├── analyze.jinja2      # Local template (same directory)
  └── prompts/           # Shared prompts directory
  ├── summarize.jinja2
  └── extract.jinja2

**Notes**:

  Templates should use .jinja2 or .jinja extension.
  The extension can be omitted when calling get().

### PromptManager

```python
class PromptManager
```

Manages Jinja2 prompt templates with smart path resolution.

PromptManager provides a convenient interface for loading and rendering
Jinja2 templates used as prompts for LLMs. It automatically searches for
templates in multiple locations, supporting both local (module-specific)
and shared (project-wide) templates.

Search hierarchy:
1. Same directory as the calling module (for local templates)
2. 'prompts' subdirectory in the calling module's directory
3. 'prompts' directories in parent packages (up to package boundary)

**Attributes**:

- `search_paths` - List of directories where templates are searched.
- `env` - Jinja2 Environment configured for prompt rendering.

**Example**:

  >>> # In flow/my_flow.py
  >>> pm = PromptManager(__file__)
  >>>
  >>> # Uses flow/prompts/analyze.jinja2 if it exists,
  >>> # otherwise searches parent directories
  >>> prompt = pm.get("analyze", context=data)
  >>>
  >>> # Can also use templates in same directory as module
  >>> prompt = pm.get("local_template.jinja2")

  Template format:
  Templates use standard Jinja2 syntax:
    ```jinja2
    Analyze the following document:
    {{ document.name }}

    {% if instructions %}
    Instructions: {{ instructions }}
    {% endif %}
    ```

**Notes**:

  - Autoescape is disabled for prompts (raw text output)
  - Whitespace control is enabled (trim_blocks, lstrip_blocks)

  Template Inheritance:
  Templates support standard Jinja2 inheritance. Templates are searched
  in order of search_paths, so templates in earlier paths override later ones.
  Precedence (first match wins):
  1. Same directory as module
  2. Module's prompts/ subdirectory
  3. Parent prompts/ directories (nearest to farthest)
  - Templates are cached by Jinja2 for performance

#### PromptManager.__init__

```python
def __init__(self, current_file: str, prompts_dir: str = "prompts")
```

Initialize PromptManager with smart template discovery.

Sets up the Jinja2 environment with a FileSystemLoader that searches
multiple directories for templates. The search starts from the calling
module's location and extends to parent package directories.

**Arguments**:

- `current_file` - The __file__ path of the calling module. Must be
  a valid file path (not __name__). Used as the
  starting point for template discovery.
- `prompts_dir` - Name of the prompts subdirectory to search for
  in each package level. Defaults to "prompts".

**Raises**:

- `PromptError` - If current_file is not a valid file path (e.g.,
  if __name__ was passed instead of __file__).

**Notes**:

  Search behavior - Given a module at /project/flows/my_flow.py:
  1. /project/flows/ (local templates)
  2. /project/flows/prompts/ (if exists)
  3. /project/prompts/ (if /project has __init__.py)

  Search stops when no __init__.py is found (package boundary).

**Example**:

  >>> # Correct usage
  >>> pm = PromptManager(__file__)
  >>>
  >>> # Custom prompts directory name
  >>> pm = PromptManager(__file__, prompts_dir="templates")
  >>>
  >>> # Common mistake (will raise PromptError)
  >>> pm = PromptManager(__name__)  # Wrong!

**Notes**:

  The search is limited to 4 parent levels to prevent
  excessive filesystem traversal.

#### PromptManager.get

```python
def get(self, prompt_path: str, **kwargs: Any) -> str
```

Load and render a Jinja2 template with the given context.

Searches for the template in all configured search paths and renders
it with the provided context variables. Automatically tries adding
.jinja2 or .jinja extensions if the file is not found.

**Arguments**:

- `prompt_path` - Path to the template file, relative to any search
  directory. Can be a simple filename ("analyze")
  or include subdirectories ("tasks/summarize").
  Extensions (.jinja2, .jinja) are optional.
- `**kwargs` - Context variables passed to the template. These become
  available as variables within the Jinja2 template.

**Returns**:

  The rendered template as a string, ready to be sent to an LLM.

**Raises**:

- `PromptNotFoundError` - If the template file cannot be found in
  any search path.
- `PromptRenderError` - If the template contains errors or if
  rendering fails (e.g., missing variables,
  syntax errors).

**Notes**:

  Template resolution - Given prompt_path="analyze":
  1. Try "analyze" as-is
  2. Try "analyze.jinja2"
  3. Try "analyze.jinja"

  The first matching file is used.

**Example**:

  >>> pm = PromptManager(__file__)
  >>>
  >>> # Simple rendering
  >>> prompt = pm.get("summarize", text="Long document...")
  >>>
  >>> # With complex context
  >>> prompt = pm.get(
  ...     "analyze",
  ...     document=doc,
  ...     max_length=500,
  ...     style="technical",
  ...     options={"include_metadata": True}
  ... )
  >>>
  >>> # Nested template path
  >>> prompt = pm.get("flows/extraction/extract_entities")

  Template example:
    ```jinja2
    Summarize the following text in {{ max_length }} words:

    {{ text }}

    {% if style %}
    Style: {{ style }}
    {% endif %}
    ```

**Notes**:

  All Jinja2 features are available: loops, conditionals,
  filters, macros, inheritance, etc.


## ai_pipeline_core.tracing

Tracing utilities that integrate Laminar (``lmnr``) with our code-base.

This module centralises:
• ``TraceInfo`` - a small helper object for propagating contextual metadata.
• ``trace`` decorator - augments a callable with Laminar tracing, automatic
``observe`` instrumentation, and optional support for test runs.

### TraceLevel

```python
TraceLevel = Literal["always", "debug", "off"]
```

Control level for tracing activation.

Values:
- "always": Always trace (default, production mode)
- "debug": Only trace when LMNR_DEBUG == "true"
- "off": Disable tracing completely

### trace

```python
def trace(func: Callable[P, R] | None = None, *, level: TraceLevel = "always", name: str | None = None, session_id: str | None = None, user_id: str | None = None, metadata: dict[str, Any] | None = None, tags: list[str] | None = None, span_type: str | None = None, ignore_input: bool = False, ignore_output: bool = False, ignore_inputs: list[str] | None = None, input_formatter: Callable[..., str] | None = None, output_formatter: Callable[..., str] | None = None, ignore_exceptions: bool = False, preserve_global_context: bool = True) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]
```

Add Laminar observability tracing to any function.

The trace decorator integrates functions with Laminar (LMNR) for
distributed tracing, performance monitoring, and debugging. It
automatically handles both sync and async functions, propagates
trace context, and provides fine-grained control over what gets traced.

**Arguments**:

- `func` - Function to trace (when used without parentheses: @trace).

- `level` - Controls when tracing is active:
  - "always": Always trace (default, production mode)
  - "debug": Only trace when LMNR_DEBUG == "true"
  - "off": Disable tracing completely

- `name` - Custom span name in traces (defaults to function.__name__).
  Use descriptive names for better trace readability.

- `session_id` - Override session ID for this function's traces.
  Typically propagated via TraceInfo instead.

- `user_id` - Override user ID for this function's traces.
  Typically propagated via TraceInfo instead.

- `metadata` - Additional key-value metadata attached to spans.
  Searchable in LMNR dashboard. Merged with TraceInfo metadata.

- `tags` - List of tags for categorizing spans (e.g., ["api", "critical"]).
  Merged with TraceInfo tags.

- `span_type` - Semantic type of the span (e.g., "LLM", "CHAIN", "TOOL").
  Affects visualization in LMNR dashboard.

- `ignore_input` - Don't record function inputs in trace (privacy/size).

- `ignore_output` - Don't record function output in trace (privacy/size).

- `ignore_inputs` - List of parameter names to exclude from trace.
  Useful for sensitive data like API keys.

- `input_formatter` - Custom function to format inputs for tracing.
  Receives all function args, returns display string.

- `output_formatter` - Custom function to format output for tracing.
  Receives function result, returns display string.

- `ignore_exceptions` - Don't record exceptions in traces (default False).

- `preserve_global_context` - Maintain Laminar's global context across
  calls (default True). Set False for isolated traces.

**Returns**:

  Decorated function with same signature but added tracing.

  TraceInfo propagation:
  If the decorated function has a 'trace_info' parameter, the decorator
  automatically creates or propagates a TraceInfo instance, ensuring
  consistent session/user tracking across the call chain.

**Example**:

  >>> # Simple usage
  >>> @trace
  >>> async def process_document(doc):
  ...     return await analyze(doc)
  >>>
  >>> # With configuration
  >>> @trace(
  ...     name="DocumentProcessor",
  ...     span_type="TOOL",
  ...     tags=["production"],
  ...     ignore_inputs=["api_key"]
  >>> )
  >>> async def process(doc, api_key, trace_info: TraceInfo):
  ...     # trace_info is automatically injected if not provided
  ...     return await call_api(doc, api_key)
  >>>
  >>> # Conditional tracing
  >>> @trace(level="debug")  # Only traces in debug mode
  >>> def expensive_operation():
  ...     pass
  >>>
  >>> # Custom formatting
  >>> @trace(
  ...     input_formatter=lambda doc: f"Document: {doc.id}",
  ...     output_formatter=lambda res: f"Results: {len(res)} items"
  >>> )
  >>> def analyze(doc):
  ...     return results

  Environment variables:
  - LMNR_DEBUG: Set to "true" to enable debug-level traces
  - LMNR_SESSION_ID: Default session ID if not in TraceInfo
  - LMNR_USER_ID: Default user ID if not in TraceInfo
  - LMNR_PROJECT_API_KEY: Required for trace submission

  Performance:
  - Tracing overhead is minimal (~1-2ms per call)
  - When level="off", decorator returns original function unchanged
  - Large inputs/outputs can be excluded with ignore_* parameters

**Notes**:

  - Automatically initializes Laminar on first use
  - Works with both sync and async functions
  - Preserves function signature and metadata
  - Thread-safe and async-safe

**See Also**:

  - TraceInfo: Container for trace metadata
  - pipeline_task: Task decorator with built-in tracing
  - pipeline_flow: Flow decorator with built-in tracing


## ai_pipeline_core.flow.options

Flow options configuration for pipeline execution.

Provides base configuration settings for AI pipeline flows,
including model selection and runtime parameters.

### FlowOptions

```python
class FlowOptions(BaseSettings)
```

Base configuration settings for AI pipeline flows.

FlowOptions provides runtime configuration for pipeline flows,
including model selection and other parameters. It uses pydantic-settings
to support environment variable overrides and is immutable (frozen) by default.

This class is designed to be subclassed for flow-specific configuration:

**Example**:

  >>> class MyFlowOptions(FlowOptions):
  ...     temperature: float = Field(0.7, ge=0, le=2)
  ...     batch_size: int = Field(10, gt=0)
  ...     custom_param: str = "default"

  >>> # Use in CLI with run_cli:
  >>> run_cli(
  ...     flows=[my_flow],
  ...     options_cls=MyFlowOptions  # Will parse CLI args
  ... )

  >>> # Or create programmatically:
  >>> options = MyFlowOptions(
  ...     core_model="gemini-2.5-pro",
  ...     temperature=0.9
  ... )

**Attributes**:

- `core_model` - Primary LLM for complex tasks (default: gpt-5)
- `small_model` - Fast model for simple tasks (default: gpt-5-mini)

  Configuration:
  - Frozen (immutable) after creation
  - Extra fields ignored (not strict)
  - Can be populated from environment variables
  - Used by simple_runner.cli for command-line parsing

**Notes**:

  The base class provides model selection. Subclasses should
  add flow-specific parameters with appropriate validation.


## ai_pipeline_core.flow.config

Flow configuration system for type-safe pipeline definitions.

This module provides the FlowConfig abstract base class that enforces
type safety for flow inputs and outputs in the pipeline system.

### FlowConfig

```python
class FlowConfig(ABC)
```

Abstract base class for type-safe flow configuration.

FlowConfig defines the contract for flow inputs and outputs, ensuring
type safety and preventing circular dependencies in pipeline flows.
Each flow must have a corresponding FlowConfig subclass that specifies
its input document types and output document type.

Class Variables:
INPUT_DOCUMENT_TYPES: List of FlowDocument types this flow accepts
OUTPUT_DOCUMENT_TYPE: Single FlowDocument type this flow produces

Validation Rules:
- INPUT_DOCUMENT_TYPES and OUTPUT_DOCUMENT_TYPE must be defined
- OUTPUT_DOCUMENT_TYPE cannot be in INPUT_DOCUMENT_TYPES (prevents cycles)
- Field names must be exact (common typos are detected)

**Example**:

  >>> class ProcessingFlowConfig(FlowConfig):
  ...     INPUT_DOCUMENT_TYPES = [RawDataDocument, ConfigDocument]
  ...     OUTPUT_DOCUMENT_TYPE = ProcessedDataDocument

  >>> # Use with flow:
  >>> if ProcessingFlowConfig.has_input_documents(docs):
  ...     input_docs = ProcessingFlowConfig.get_input_documents(docs)
  ...     output = await process_flow(input_docs)
  ...     ProcessingFlowConfig.validate_output_documents(output)

**Notes**:

  - Validation happens at class definition time
  - Helps catch configuration errors early
  - Used by simple_runner to manage document flow


## ai_pipeline_core.pipeline

Pipeline decorators with Prefect integration and tracing.

Wrappers around Prefect's @task and @flow that add Laminar tracing
and enforce async-only execution for consistency.

### pipeline_task

```python
def pipeline_task(__fn: Callable[..., Coroutine[Any, Any, R_co]] | None = None, *, trace_level: TraceLevel = "always", trace_ignore_input: bool = False, trace_ignore_output: bool = False, trace_ignore_inputs: list[str] | None = None, trace_input_formatter: Callable[..., str] | None = None, trace_output_formatter: Callable[..., str] | None = None, name: str | None = None, description: str | None = None, tags: Iterable[str] | None = None, version: str | None = None, cache_policy: CachePolicy | type[NotSet] = NotSet, cache_key_fn: Callable[[TaskRunContext, dict[str, Any]], str | None] | None = None, cache_expiration: datetime.timedelta | None = None, task_run_name: TaskRunNameValueOrCallable | None = None, retries: int | None = None, retry_delay_seconds: int | float | list[float] | Callable[[int], list[float]] | None = None, retry_jitter_factor: float | None = None, persist_result: bool | None = None, result_storage: ResultStorage | str | None = None, result_serializer: ResultSerializer | str | None = None, result_storage_key: str | None = None, cache_result_in_memory: bool = True, timeout_seconds: int | float | None = None, log_prints: bool | None = False, refresh_cache: bool | None = None, on_completion: list[StateHookCallable] | None = None, on_failure: list[StateHookCallable] | None = None, retry_condition_fn: RetryConditionCallable | None = None, viz_return_value: bool | None = None, asset_deps: list[str | Asset] | None = None) -> _TaskLike[R_co] | Callable[[Callable[..., Coroutine[Any, Any, R_co]]], _TaskLike[R_co]]
```

Decorate an async function as a traced Prefect task.

Wraps an async function with both Prefect task functionality and
LMNR tracing. The function MUST be async (declared with 'async def').

**Arguments**:

- `__fn` - Function to decorate (when used without parentheses).

  Tracing parameters:
- `trace_level` - When to trace ("always", "debug", "off").
  - "always": Always trace (default)
  - "debug": Only trace when LMNR_DEBUG="true"
  - "off": Disable tracing
- `trace_ignore_input` - Don't trace input arguments.
- `trace_ignore_output` - Don't trace return value.
- `trace_ignore_inputs` - List of parameter names to exclude from tracing.
- `trace_input_formatter` - Custom formatter for input tracing.
- `trace_output_formatter` - Custom formatter for output tracing.

  Prefect task parameters:
- `name` - Task name (defaults to function name).
- `description` - Human-readable task description.
- `tags` - Tags for organization and filtering.
- `version` - Task version string.
- `cache_policy` - Caching policy for task results.
- `cache_key_fn` - Custom cache key generation.
- `cache_expiration` - How long to cache results.
- `task_run_name` - Dynamic or static run name.
- `retries` - Number of retry attempts (default 0).
- `retry_delay_seconds` - Delay between retries.
- `retry_jitter_factor` - Random jitter for retry delays.
- `persist_result` - Whether to persist results.
- `result_storage` - Where to store results.
- `result_serializer` - How to serialize results.
- `result_storage_key` - Custom storage key.
- `cache_result_in_memory` - Keep results in memory.
- `timeout_seconds` - Task execution timeout.
- `log_prints` - Capture print() statements.
- `refresh_cache` - Force cache refresh.
- `on_completion` - Hooks for successful completion.
- `on_failure` - Hooks for task failure.
- `retry_condition_fn` - Custom retry condition.
- `viz_return_value` - Include return value in visualization.
- `asset_deps` - Upstream asset dependencies.

**Returns**:

  Decorated task callable that is awaitable and has Prefect
  task methods (submit, map, etc.).

**Example**:

  >>> @pipeline_task
  >>> async def process_document(doc: Document) -> Document:
  ...     result = await analyze(doc)
  ...     return result
  >>>
  >>> # With parameters
  >>> @pipeline_task(retries=3, timeout_seconds=30)
  >>> async def fetch_data(url: str) -> dict:
  ...     async with httpx.AsyncClient() as client:
  ...         return (await client.get(url)).json()

  Performance:
  - Task decoration overhead: ~1-2ms
  - Tracing overhead: ~1-2ms per call
  - Prefect state tracking: ~5-10ms

**Notes**:

  Tasks are automatically traced with LMNR and appear in
  both Prefect and LMNR dashboards.

**See Also**:

  - pipeline_flow: For flow-level decoration
  - trace: Lower-level tracing decorator
  - prefect.task: Standard Prefect task (no tracing)

### pipeline_flow

```python
def pipeline_flow(__fn: _DocumentsFlowCallable[FO_contra] | None = None, *, trace_level: TraceLevel = "always", trace_ignore_input: bool = False, trace_ignore_output: bool = False, trace_ignore_inputs: list[str] | None = None, trace_input_formatter: Callable[..., str] | None = None, trace_output_formatter: Callable[..., str] | None = None, name: str | None = None, version: str | None = None, flow_run_name: Union[Callable[[], str], str] | None = None, retries: int | None = None, retry_delay_seconds: int | float | None = None, task_runner: TaskRunner[PrefectFuture[Any]] | None = None, description: str | None = None, timeout_seconds: int | float | None = None, validate_parameters: bool = True, persist_result: bool | None = None, result_storage: ResultStorage | str | None = None, result_serializer: ResultSerializer | str | None = None, cache_result_in_memory: bool = True, log_prints: bool | None = None, on_completion: list[FlowStateHook[Any, Any]] | None = None, on_failure: list[FlowStateHook[Any, Any]] | None = None, on_cancellation: list[FlowStateHook[Any, Any]] | None = None, on_crashed: list[FlowStateHook[Any, Any]] | None = None, on_running: list[FlowStateHook[Any, Any]] | None = None) -> _FlowLike[FO_contra] | Callable[[_DocumentsFlowCallable[FO_contra]], _FlowLike[FO_contra]]
```

Decorate an async flow for document processing.

Wraps an async function as a Prefect flow with tracing and type safety.
The decorated function MUST be async and follow the required signature.

Required function signature:
async def flow_fn(
project_name: str,         # Project/pipeline identifier
documents: DocumentList,   # Input documents to process
flow_options: FlowOptions, # Configuration (or subclass)
*args,                     # Additional positional args
**kwargs                   # Additional keyword args
) -> DocumentList             # Must return DocumentList

**Arguments**:

- `__fn` - Function to decorate (when used without parentheses).

  Tracing parameters:
- `trace_level` - When to trace ("always", "debug", "off").
  - "always": Always trace (default)
  - "debug": Only trace when LMNR_DEBUG="true"
  - "off": Disable tracing
- `trace_ignore_input` - Don't trace input arguments.
- `trace_ignore_output` - Don't trace return value.
- `trace_ignore_inputs` - Parameter names to exclude from tracing.
- `trace_input_formatter` - Custom input formatter.
- `trace_output_formatter` - Custom output formatter.

  Prefect flow parameters:
- `name` - Flow name (defaults to function name).
- `version` - Flow version identifier.
- `flow_run_name` - Static or dynamic run name.
- `retries` - Number of flow retry attempts (default 0).
- `retry_delay_seconds` - Delay between flow retries.
- `task_runner` - Task execution strategy (sequential/concurrent).
- `description` - Human-readable flow description.
- `timeout_seconds` - Flow execution timeout.
- `validate_parameters` - Validate input parameters.
- `persist_result` - Persist flow results.
- `result_storage` - Where to store results.
- `result_serializer` - How to serialize results.
- `cache_result_in_memory` - Keep results in memory.
- `log_prints` - Capture print() statements.
- `on_completion` - Hooks for successful completion.
- `on_failure` - Hooks for flow failure.
- `on_cancellation` - Hooks for flow cancellation.
- `on_crashed` - Hooks for flow crashes.
- `on_running` - Hooks for flow start.

**Returns**:

  Decorated flow callable that maintains Prefect flow interface
  while enforcing document processing conventions.

**Example**:

  >>> from ai_pipeline_core.flow.options import FlowOptions
  >>>
  >>> @pipeline_flow
  >>> async def analyze_documents(
  ...     project_name: str,
  ...     documents: DocumentList,
  ...     flow_options: FlowOptions
  >>> ) -> DocumentList:
  ...     # Process each document
  ...     results = []
  ...     for doc in documents:
  ...         result = await process(doc)
  ...         results.append(result)
  ...     return DocumentList(results)
  >>>
  >>> # With custom options subclass
  >>> class MyOptions(FlowOptions):
  ...     model: str = "gpt-5"
  >>>
  >>> @pipeline_flow(retries=2, timeout_seconds=300)
  >>> async def custom_flow(
  ...     project_name: str,
  ...     documents: DocumentList,
  ...     flow_options: MyOptions,  # Subclass works
  ...     extra_param: str = "default"
  >>> ) -> DocumentList:
  ...     # Flow implementation
  ...     return documents

**Notes**:

  - Flow is wrapped with both Prefect and LMNR tracing
  - Return type is validated at runtime
  - FlowOptions can be subclassed for custom configuration
  - All Prefect flow methods (.serve(), .deploy()) are available

**See Also**:

  - pipeline_task: For task-level decoration
  - FlowConfig: Type-safe flow configuration
  - FlowOptions: Base class for flow options
  - simple_runner.run_pipeline: Execute flows locally


## ai_pipeline_core.logging.logging_config

Centralized logging configuration for AI Pipeline Core.

Provides logging configuration management that integrates with Prefect's logging system.

### LoggingConfig

```python
class LoggingConfig
```

Manages logging configuration for the pipeline.

Provides centralized logging configuration with Prefect integration.

Configuration precedence:
1. Explicit config_path parameter
2. AI_PIPELINE_LOGGING_CONFIG environment variable
3. PREFECT_LOGGING_SETTINGS_PATH environment variable
4. Default configuration

**Example**:

  >>> config = LoggingConfig()
  >>> config.apply()

### setup_logging

```python
def setup_logging(config_path: Optional[Path] = None, level: Optional[str] = None)
```

Setup logging for the AI Pipeline Core library.

Initializes logging configuration for the pipeline system.

**Arguments**:

- `config_path` - Optional path to YAML logging configuration file.
- `level` - Optional log level override (INFO, DEBUG, WARNING, etc.).

**Example**:

  >>> setup_logging()
  >>> setup_logging(level="DEBUG")

### get_pipeline_logger

```python
def get_pipeline_logger(name: str)
```

Get a logger for pipeline components.

Returns a Prefect-integrated logger with proper configuration.

**Arguments**:

- `name` - Logger name, typically __name__.

**Returns**:

  Prefect logger instance.

**Example**:

  >>> logger = get_pipeline_logger(__name__)
  >>> logger.info("Module initialized")


## ai_pipeline_core.logging

Logging infrastructure for AI Pipeline Core.

Provides a Prefect-integrated logging facade for unified logging across pipelines.
Prefer get_pipeline_logger instead of logging.getLogger to ensure proper integration.

**Example**:

  >>> from ai_pipeline_core.logging import get_pipeline_logger
  >>> logger = get_pipeline_logger(__name__)
  >>> logger.info("Processing started")


## ai_pipeline_core.logging.logging_mixin

Logging mixin for consistent logging across components using Prefect logging.

### LoggerMixin

```python
class LoggerMixin
```

Mixin class that provides consistent logging functionality using Prefect's logging system.

Automatically uses appropriate logger based on context:
- get_run_logger() when in flow/task context
- get_logger() when outside flow/task context

### StructuredLoggerMixin

```python
class StructuredLoggerMixin(LoggerMixin)
```

Extended mixin for structured logging with Prefect.


## ai_pipeline_core.settings

Core configuration settings for pipeline operations.

This module provides centralized configuration management for AI Pipeline Core,
handling all external service credentials and endpoints. Settings are loaded
from environment variables with .env file support via pydantic-settings.

Environment variables:
OPENAI_BASE_URL: LiteLLM proxy endpoint (e.g., http://localhost:4000)
OPENAI_API_KEY: API key for LiteLLM proxy authentication
PREFECT_API_URL: Prefect server endpoint for flow orchestration
PREFECT_API_KEY: Prefect API authentication key
LMNR_PROJECT_API_KEY: Laminar project key for observability

Configuration precedence:
1. Environment variables (highest priority)
2. .env file in current directory
3. Default values (empty strings)

**Example**:

  >>> from ai_pipeline_core.settings import settings
  >>>
  >>> # Access configuration
  >>> print(settings.openai_base_url)
  >>> print(settings.prefect_api_url)
  >>>
  >>> # Settings are frozen after initialization
  >>> settings.openai_api_key = "new_key"  # Raises error

  .env file format:
  OPENAI_BASE_URL=http://localhost:4000
  OPENAI_API_KEY=sk-1234567890
  PREFECT_API_URL=http://localhost:4200/api
  PREFECT_API_KEY=pnu_abc123
  LMNR_PROJECT_API_KEY=lmnr_proj_xyz

**Notes**:

  Settings are loaded once at module import and frozen. There is no
  built-in reload mechanism - the process must be restarted to pick up
  changes to environment variables or .env file. This is by design to
  ensure consistency during execution.

### settings

```python
settings = Settings()
```

Global settings instance for the entire application.

This singleton instance is created at module import and provides
configuration to all pipeline components. Access this instance
rather than creating new Settings objects.

**Example**:

  >>> from ai_pipeline_core.settings import settings
  >>> print(f"Using LLM proxy at {settings.openai_base_url}")

### Settings

```python
class Settings(BaseSettings)
```

Core configuration for AI Pipeline external services.

Settings provides type-safe configuration management with automatic
loading from environment variables and .env files. All settings are
immutable after initialization.

**Attributes**:

- `openai_base_url` - LiteLLM proxy URL for OpenAI-compatible API.
  Required for all LLM operations. Usually
  http://localhost:4000 for local development.

- `openai_api_key` - Authentication key for LiteLLM proxy. Required
  for LLM operations. Format depends on proxy config.

- `prefect_api_url` - Prefect server API endpoint. Required for flow
  deployment and remote execution. Leave empty for
  local-only execution.

- `prefect_api_key` - Prefect API authentication key. Required only
  when connecting to Prefect Cloud or secured server.

- `lmnr_project_api_key` - Laminar (LMNR) project API key for tracing
  and observability. Optional but recommended
  for production monitoring.

  Configuration sources:
  - Environment variables (OPENAI_BASE_URL, etc.)
  - .env file in current directory
  - Default empty strings if not configured

**Example**:

  >>> # Typically accessed via module-level instance
  >>> from ai_pipeline_core.settings import settings
  >>>
  >>> if not settings.openai_base_url:
  ...     raise ValueError("OPENAI_BASE_URL must be configured")
  >>>
  >>> # Settings are frozen (immutable)
  >>> print(settings.model_dump())  # View all settings

**Notes**:

  Empty strings are used as defaults to allow optional services.
  Check for empty values before using service-specific settings.


## ai_pipeline_core.documents.task_document

Task-specific document base class for temporary pipeline data.

This module provides the TaskDocument abstract base class for documents
that exist only during Prefect task execution and are not persisted.

### TaskDocument

```python
class TaskDocument(Document)
```

Abstract base class for temporary documents within task execution.

TaskDocument is used for intermediate data that exists only during
the execution of a Prefect task and is not persisted to disk. These
documents are ideal for temporary processing results, transformations,
and data that doesn't need to survive beyond the current task.

Key characteristics:
- Not persisted to file system
- Exists only during task execution
- Garbage collected after task completes
- Used for intermediate processing results
- More memory-efficient for temporary data

Creating TaskDocuments:
**Use the `create` classmethod** for most use cases. It handles automatic
conversion of various content types. Only use __init__ when you have bytes.

>>> from enum import StrEnum
>>>
>>> # Simple task document:
>>> class TempDoc(TaskDocument):
...     pass
>>>
>>> # With restricted files:
>>> class CacheDoc(TaskDocument):
...     class FILES(StrEnum):
...         CACHE = "cache.json"
...         INDEX = "index.dat"
>>>
>>> # RECOMMENDED - automatic conversion:
>>> doc = TempDoc.create(name="temp.json", content={"status": "processing"})
>>> doc = CacheDoc.create(name="cache.json", content={"data": [1, 2, 3]})

Use Cases:
- Intermediate transformation results
- Temporary buffers during processing
- Task-local cache data
- Processing status documents

**Notes**:

  - Cannot instantiate TaskDocument directly - must subclass
  - Not saved by simple_runner utilities
  - Reduces I/O overhead for temporary data
  - No additional abstract methods to implement

**See Also**:

- `FlowDocument` - For documents that persist across flow runs
- `TemporaryDocument` - Alternative for non-persistent documents


## ai_pipeline_core.documents.document

Document abstraction layer for AI pipeline flows.

This module provides the core document abstraction for working with various types of data
in AI pipelines. Documents are immutable Pydantic models that wrap binary content with metadata.

### Document

```python
class Document(BaseModel, ABC)
```

Abstract base class for all documents in the AI Pipeline Core system.

Document is the fundamental data abstraction for all content flowing through
pipelines. It provides automatic encoding, MIME type detection, serialization,
and validation. All documents must be subclassed from FlowDocument, TaskDocument,
or TemporaryDocument based on their persistence requirements.

Key features:
- Immutable by default (frozen Pydantic model)
- Automatic MIME type detection
- Content size validation
- SHA256 hashing for deduplication
- Support for text, JSON, YAML, PDF, and image formats
- Conversion utilities between different formats

Class Variables:
MAX_CONTENT_SIZE: Maximum allowed content size in bytes (default 25MB)
DESCRIPTION_EXTENSION: File extension for description files (.description.md)
MARKDOWN_LIST_SEPARATOR: Separator for markdown list items

**Attributes**:

- `name` - Document filename (validated for security)
- `description` - Optional human-readable description
- `content` - Raw document content as bytes

  Creating Documents:
  **Use the `create` classmethod** for most use cases. It accepts various
  content types (str, dict, list, BaseModel) and converts them automatically.
  Only use __init__ directly when you already have bytes content.

  >>> # RECOMMENDED: Use create for automatic conversion
  >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
  >>>
  >>> # Direct constructor: Only for bytes
  >>> doc = MyDocument(name="data.bin", content=b"\x00\x01\x02")

**Warnings**:

  - Document subclasses should NOT start with 'Test' prefix (pytest conflict)
  - Cannot instantiate Document directly - use FlowDocument or TaskDocument
  - Cannot add custom fields - only name, description, content are allowed

  Metadata Attachment Patterns:
  Since custom fields are not allowed, use these patterns for metadata:
  1. Use the 'description' field for human-readable metadata
  2. Embed metadata in content (e.g., JSON with data + metadata fields)
  3. Create a separate MetadataDocument type to accompany data documents
  4. Use document naming conventions (e.g., "data_v2_2024.json")
  5. Store metadata in flow_options or pass through TraceInfo

**Example**:

  >>> from enum import StrEnum
  >>>
  >>> # Simple document:
  >>> class MyDocument(FlowDocument):
  ...     pass
  >>>
  >>> # Document with file restrictions:
  >>> class ConfigDocument(FlowDocument):
  ...     class FILES(StrEnum):
  ...         CONFIG = "config.yaml"
  ...         SETTINGS = "settings.json"
  >>>
  >>> # RECOMMENDED: Use create for automatic conversion
  >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
  >>> print(doc.is_text)  # True
  >>> data = doc.as_json()  # {'key': 'value'}

#### Document.MAX_CONTENT_SIZE

```python
MAX_CONTENT_SIZE: ClassVar[int] = 25 * 1024 * 1024
```

Maximum allowed content size in bytes (default 25MB).

#### Document.DESCRIPTION_EXTENSION

```python
DESCRIPTION_EXTENSION: ClassVar[str] = ".description.md"
```

File extension for description files.

#### Document.MARKDOWN_LIST_SEPARATOR

```python
MARKDOWN_LIST_SEPARATOR: ClassVar[str] = "\n\n---\n\n"
```

Separator for markdown list items.

#### Document.create

```python
@classmethod
def create(cls, *, name: str, content: ContentInput, description: str | None = None) -> Self
```

Create a Document with automatic content type conversion (recommended).

This is the **recommended way to create documents**. It accepts various
content types and automatically converts them to bytes based on the file
extension. Use the `parse` method to reverse this conversion.

**Arguments**:

- `name` - Document filename (required, keyword-only).
  Extension determines serialization:
  - .json → JSON serialization
  - .yaml/.yml → YAML serialization
  - .md → Markdown list joining (for list[str])
  - Others → UTF-8 encoding (for str)
- `content` - Document content in various formats (required, keyword-only):
  - bytes: Used directly without conversion
  - str: Encoded to UTF-8 bytes
  - dict[str, Any]: Serialized to JSON (.json) or YAML (.yaml/.yml)
  - list[str]: Joined with separator for .md, else JSON/YAML
  - list[BaseModel]: Serialized to JSON or YAML based on extension
  - BaseModel: Serialized to JSON or YAML based on extension
- `description` - Optional human-readable description (keyword-only)

**Returns**:

  New Document instance with content converted to bytes

**Raises**:

- `ValueError` - If content type is not supported for the file extension
- `DocumentNameError` - If filename violates validation rules
- `DocumentSizeError` - If content exceeds MAX_CONTENT_SIZE

**Notes**:

  All conversions are reversible using the `parse` method.
  For example: MyDocument.create(name="data.json", content={"key": "value"}).parse(dict)
  returns the original dictionary {"key": "value"}.

**Example**:

  >>> # String content
  >>> doc = MyDocument.create(name="test.txt", content="Hello World")
  >>> doc.content  # b'Hello World'
  >>> doc.parse(str)  # "Hello World"

  >>> # Dictionary to JSON
  >>> doc = MyDocument.create(name="config.json", content={"key": "value"})
  >>> doc.content  # b'{"key": "value", ...}'
  >>> doc.parse(dict)  # {"key": "value"}

  >>> # Pydantic model to YAML
  >>> from pydantic import BaseModel
  >>> class Config(BaseModel):
  ...     host: str
  ...     port: int
  >>> config = Config(host="localhost", port=8080)
  >>> doc = MyDocument.create(name="config.yaml", content=config)
  >>> doc.parse(Config)  # Returns Config instance

  >>> # List to Markdown
  >>> items = ["Section 1", "Section 2"]
  >>> doc = MyDocument.create(name="sections.md", content=items)
  >>> doc.parse(list)  # ["Section 1", "Section 2"]

#### Document.__init__

```python
def __init__(self, *, name: str, content: bytes, description: str | None = None) -> None
```

Initialize a Document instance with raw bytes content.

Important:
**Most users should use the `create` classmethod instead of __init__.**
The create method provides automatic content conversion for various types
(str, dict, list, Pydantic models) while __init__ only accepts bytes.

This constructor accepts only bytes content for type safety. It prevents
direct instantiation of the abstract Document class.

**Arguments**:

- `name` - Document filename (required, keyword-only)
- `content` - Document content as raw bytes (required, keyword-only)
- `description` - Optional human-readable description (keyword-only)

**Raises**:

- `TypeError` - If attempting to instantiate Document directly.

**Example**:

  >>> # Direct constructor - only for bytes content:
  >>> doc = MyDocument(name="test.txt", content=b"Hello World")
  >>> doc.content  # b'Hello World'

  >>> # RECOMMENDED: Use create for automatic conversion:
  >>> doc = MyDocument.create(name="text.txt", content="Hello World")
  >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
  >>> doc = MyDocument.create(name="config.yaml", content=my_model)
  >>> doc = MyDocument.create(name="items.md", content=["item1", "item2"])

**See Also**:

- `create` - Recommended factory method with automatic type conversion
- `parse` - Method to reverse the conversion done by create

#### Document.base_type

```python
@final
@property
def base_type(self) -> Literal["flow", "task", "temporary"]
```

Get the document's base type.

Property alias for get_base_type() providing a cleaner API.
This property cannot be overridden by subclasses.

**Returns**:

  The document's base type: "flow", "task", or "temporary".

#### Document.is_flow

```python
@final
@property
def is_flow(self) -> bool
```

Check if this is a flow document.

Flow documents persist across Prefect flow runs and are saved
to the file system between pipeline steps.

**Returns**:

  True if this is a FlowDocument subclass, False otherwise.

#### Document.is_task

```python
@final
@property
def is_task(self) -> bool
```

Check if this is a task document.

Task documents are temporary within Prefect task execution
and are not persisted between pipeline steps.

**Returns**:

  True if this is a TaskDocument subclass, False otherwise.

#### Document.is_temporary

```python
@final
@property
def is_temporary(self) -> bool
```

Check if this is a temporary document.

Temporary documents are never persisted and exist only
during execution.

**Returns**:

  True if this is a TemporaryDocument, False otherwise.

#### Document.validate_file_name

```python
@classmethod
def validate_file_name(cls, name: str) -> None
```

Validate that a file name matches allowed patterns.

This method provides a hook for enforcing file naming conventions.
By default, it checks against the FILES enum if defined.
Subclasses can override for custom validation logic.

**Arguments**:

- `name` - The file name to validate.

**Raises**:

- `DocumentNameError` - If the name doesn't match allowed patterns.

**Notes**:

  - If FILES enum is defined, name must exactly match one of the values
  - If FILES is not defined, any name is allowed
  - Override in subclasses for regex patterns or other conventions

**See Also**:

  - get_expected_files: Returns list of allowed file names
  - validate_name: Pydantic validator for security checks

#### Document.id

```python
@final
@property
def id(self) -> str
```

Get a short unique identifier for the document.

Returns the first 6 characters of the base32-encoded SHA256 hash,
providing a compact identifier suitable for logging
and display purposes.

**Returns**:

  6-character base32-encoded string (uppercase, e.g., "A7B2C9").
  This is the first 6 chars of the full base32 SHA256, NOT hex.

  Collision Rate:
  With base32 encoding (5 bits per char), 6 chars = 30 bits.
  Expect collisions after ~32K documents (birthday paradox).
  For higher uniqueness requirements, use the full sha256 property.

**Notes**:

  While shorter than full SHA256, this provides
  reasonable uniqueness for most use cases.

#### Document.sha256

```python
@final
@cached_property
def sha256(self) -> str
```

Get the full SHA256 hash of the document content.

Computes and caches the SHA256 hash of the content,
encoded in base32 (uppercase). Used for content
deduplication and integrity verification.

**Returns**:

  Full SHA256 hash as base32-encoded uppercase string.

  Why Base32 Instead of Hex:
  - Base32 is case-insensitive (safer for filesystems)
  - More compact than hex (52 chars vs 64 chars for SHA-256)
  - Safe for URLs without encoding
  - Compatible with systems that have case-insensitive paths

**Notes**:

  This is computed once and cached for performance.
  The hash is deterministic based on content only.

#### Document.size

```python
@final
@property
def size(self) -> int
```

Get the size of the document content.

**Returns**:

  Size of content in bytes.

**Notes**:

  Useful for monitoring document sizes and
  ensuring they stay within limits.

#### Document.mime_type

```python
@property
def mime_type(self) -> str
```

Get the document's MIME type.

Primary property for accessing MIME type information.
Currently delegates to detected_mime_type but provides
a stable API for future enhancements.

**Returns**:

  MIME type string.

**Notes**:

  Prefer this over detected_mime_type for general use.

#### Document.is_text

```python
@property
def is_text(self) -> bool
```

Check if document contains text content.

**Returns**:

  True if MIME type indicates text content
  (text/*, application/json, application/yaml, etc.),
  False otherwise.

**Notes**:

  Used to determine if text property can be safely accessed.

#### Document.is_pdf

```python
@property
def is_pdf(self) -> bool
```

Check if document is a PDF file.

**Returns**:

  True if MIME type is application/pdf, False otherwise.

**Notes**:

  PDF documents require special handling and are
  supported by certain LLM models.

#### Document.is_image

```python
@property
def is_image(self) -> bool
```

Check if document is an image file.

**Returns**:

  True if MIME type starts with "image/", False otherwise.

**Notes**:

  Image documents are automatically encoded for
  vision-capable LLM models.

#### Document.text

```python
@property
def text(self) -> str
```

Get document content as UTF-8 text string.

Decodes the bytes content as UTF-8 text. Only available for
text-based documents (check is_text property first).

**Returns**:

  UTF-8 decoded string.

**Raises**:

- `ValueError` - If document is not text (is_text == False).

**Example**:

  >>> doc = MyDocument.create(name="data.txt", content="Hello ✨")
  >>> if doc.is_text:
  ...     print(doc.text)  # "Hello ✨"

  >>> # Binary document raises error:
  >>> binary_doc = MyDocument(name="image.png", content=png_bytes)
  >>> binary_doc.text  # Raises ValueError

#### Document.as_yaml

```python
def as_yaml(self) -> Any
```

Parse document content as YAML.

Parses the document's text content as YAML and returns Python objects.
Uses ruamel.yaml which is safe by default (no code execution).

**Returns**:

  Parsed YAML data: dict, list, str, int, float, bool, or None.

**Raises**:

- `ValueError` - If document is not text-based.
- `YAMLError` - If content is not valid YAML.

**Example**:

  >>> # From dict content
  >>> doc = MyDocument.create(name="config.yaml", content={
  ...     "server": {"host": "localhost", "port": 8080}
  ... })
  >>> doc.as_yaml()  # {'server': {'host': 'localhost', 'port': 8080}}

  >>> # From YAML string
  >>> doc2 = MyDocument(name="simple.yml", content=b"key: value\nitems:\n  - a\n  - b")
  >>> doc2.as_yaml()  # {'key': 'value', 'items': ['a', 'b']}

#### Document.as_json

```python
def as_json(self) -> Any
```

Parse document content as JSON.

Parses the document's text content as JSON and returns Python objects.
Document must contain valid JSON text.

**Returns**:

  Parsed JSON data: dict, list, str, int, float, bool, or None.

**Raises**:

- `ValueError` - If document is not text-based.
- `JSONDecodeError` - If content is not valid JSON.

**Example**:

  >>> # From dict content
  >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
  >>> doc.as_json()  # {'key': 'value'}

  >>> # From JSON string
  >>> doc2 = MyDocument(name="array.json", content=b'[1, 2, 3]')
  >>> doc2.as_json()  # [1, 2, 3]

  >>> # Invalid JSON
  >>> bad_doc = MyDocument(name="bad.json", content=b"not json")
  >>> bad_doc.as_json()  # Raises JSONDecodeError

#### Document.as_pydantic_model

```python
def as_pydantic_model(self, model_type: type[TModel] | type[list[TModel]]) -> TModel | list[TModel]
```

Parse document content as Pydantic model with validation.

Parses JSON or YAML content and validates it against a Pydantic model.
Automatically detects format based on MIME type. Supports both single
models and lists of models.

**Arguments**:

- `model_type` - Pydantic model class to validate against.
  Can be either:
  - type[Model] for single model
  - type[list[Model]] for list of models

**Returns**:

  Validated Pydantic model instance or list of instances.

**Raises**:

- `ValueError` - If document is not text or type mismatch.
- `ValidationError` - If data doesn't match model schema.
- `JSONDecodeError/YAMLError` - If content parsing fails.

**Example**:

  >>> from pydantic import BaseModel
  >>>
  >>> class User(BaseModel):
  ...     name: str
  ...     age: int
  >>>
  >>> # Single model
  >>> doc = MyDocument.create(name="user.json",
  ...     content={"name": "Alice", "age": 30})
  >>> user = doc.as_pydantic_model(User)
  >>> print(user.name)  # "Alice"
  >>>
  >>> # List of models
  >>> doc2 = MyDocument.create(name="users.json",
  ...     content=[{"name": "Bob", "age": 25}, {"name": "Eve", "age": 28}])
  >>> users = doc2.as_pydantic_model(list[User])
  >>> print(len(users))  # 2

#### Document.as_markdown_list

```python
def as_markdown_list(self) -> list[str]
```

Parse document as markdown-separated list of sections.

Splits text content using MARKDOWN_LIST_SEPARATOR ("\n\n---\n\n").
Designed for markdown documents with multiple sections.

**Returns**:

  List of string sections (preserves whitespace within sections).

**Raises**:

- `ValueError` - If document is not text-based.

**Example**:

  >>> # Using create with list
  >>> sections = ["# Chapter 1\nIntroduction", "# Chapter 2\nDetails"]
  >>> doc = MyDocument.create(name="book.md", content=sections)
  >>> doc.as_markdown_list()  # Returns original sections

  >>> # Manual creation with separator
  >>> content = "Part 1\n\n---\n\nPart 2\n\n---\n\nPart 3"
  >>> doc2 = MyDocument(name="parts.md", content=content.encode())
  >>> doc2.as_markdown_list()  # ['Part 1', 'Part 2', 'Part 3']

#### Document.parse

```python
def parse(self, type_: type[Any]) -> Any
```

Parse document content to original type (reverses create conversion).

This method reverses the automatic conversion performed by the `create`
classmethod. It intelligently parses the bytes content based on the
document's file extension and converts to the requested type.

Designed for roundtrip conversion:
>>> original = {"key": "value"}
>>> doc = MyDocument.create(name="data.json", content=original)
>>> restored = doc.parse(dict)
>>> assert restored == original  # True

**Arguments**:

- `type_` - Target type to parse content into. Supported types:
  - bytes: Returns raw content (no conversion)
  - str: Decodes UTF-8 text
  - dict: Parses JSON (.json) or YAML (.yaml/.yml)
  - list: Splits markdown (.md) or parses JSON/YAML
  - BaseModel subclasses: Validates JSON/YAML into model

**Returns**:

  Content parsed to the requested type.

**Raises**:

- `ValueError` - If type is unsupported or parsing fails.

  Extension Rules:
  - .json → JSON parsing for dict/list/BaseModel
  - .yaml/.yml → YAML parsing for dict/list/BaseModel
  - .md + list → Split by markdown separator
  - Any + str → UTF-8 decode
  - Any + bytes → Raw content

**Example**:

  >>> # String content
  >>> doc = MyDocument(name="test.txt", content=b"Hello")
  >>> doc.parse(str)
  'Hello'

  >>> # JSON content
  >>> doc = MyDocument.create(name="data.json", content={"key": "value"})
  >>> doc.parse(dict)  # Returns {'key': 'value'}

  >>> # Markdown list
  >>> items = ["Item 1", "Item 2"]
  >>> content = "\n\n---\n\n".join(items).encode()
  >>> doc = MyDocument(name="list.md", content=content)
  >>> doc.parse(list)
  ['Item 1', 'Item 2']


## ai_pipeline_core.documents

Document abstraction system for AI pipeline flows.

The documents package provides immutable, type-safe data structures for handling
various content types in AI pipelines, including text, images, PDFs, and other
binary data with automatic MIME type detection.


## ai_pipeline_core.documents.flow_document

Flow-specific document base class for persistent pipeline data.

This module provides the FlowDocument abstract base class for documents
that need to persist across Prefect flow runs and between pipeline steps.

### FlowDocument

```python
class FlowDocument(Document)
```

Abstract base class for documents that persist across flow runs.

FlowDocument is used for data that needs to be saved between pipeline
steps and across multiple flow executions. These documents are typically
written to the file system using the simple_runner utilities.

Key characteristics:
- Persisted to file system between pipeline steps
- Survives across multiple flow runs
- Used for flow inputs and outputs
- Saved in directories named after the document's canonical name

Creating FlowDocuments:
**Use the `create` classmethod** for most use cases. It handles automatic
conversion of various content types. Only use __init__ when you have bytes.

>>> from enum import StrEnum
>>>
>>> # Simple document with pass:
>>> class MyDoc(FlowDocument):
...     pass
>>>
>>> # Document with restricted file names:
>>> class ConfigDoc(FlowDocument):
...     class FILES(StrEnum):
...         CONFIG = "config.yaml"
...         SETTINGS = "settings.json"
>>>
>>> # RECOMMENDED - automatic conversion:
>>> doc = MyDoc.create(name="data.json", content={"key": "value"})
>>> doc = ConfigDoc.create(name="config.yaml", content={"host": "localhost"})

Persistence:
Documents are saved to: {output_dir}/{canonical_name}/{filename}
For example: output/my_doc/data.json

**Notes**:

  - Cannot instantiate FlowDocument directly - must subclass
  - Used with FlowConfig to define flow input/output types
  - No additional abstract methods to implement

**See Also**:

- `TaskDocument` - For temporary documents within task execution
- `TemporaryDocument` - For documents that are never persisted


## ai_pipeline_core.documents.temporary_document

Temporary document implementation for non-persistent data.

This module provides the TemporaryDocument class for documents that
are never persisted, regardless of context.

### TemporaryDocument

```python
class TemporaryDocument(Document)
```

Concrete document class for data that is never persisted.

TemporaryDocument is a final (non-subclassable) document type for
data that should never be saved to disk, regardless of whether it's
used in a flow or task context. Unlike FlowDocument and TaskDocument
which are abstract, TemporaryDocument can be instantiated directly.

Key characteristics:
- Never persisted to file system
- Can be instantiated directly (not abstract)
- Cannot be subclassed (marked as @final)
- Useful for transient data like API responses or intermediate calculations
- Ignored by simple_runner save operations

Creating TemporaryDocuments:
**Use the `create` classmethod** for most use cases. It handles automatic
conversion of various content types. Only use __init__ when you have bytes.

>>> # RECOMMENDED - automatic conversion:
>>> doc = TemporaryDocument.create(
...     name="api_response.json",
...     content={"status": "ok", "data": [1, 2, 3]}
... )
>>> doc = TemporaryDocument.create(
...     name="credentials.txt",
...     content="secret_token_xyz"
... )
>>>
>>> # Direct constructor - only for bytes:
>>> doc = TemporaryDocument(
...     name="binary.dat",
...     content=b"\x00\x01\x02"
... )
>>>
>>> doc.is_temporary  # Always True

Use Cases:
- API responses that shouldn't be cached
- Sensitive credentials or tokens
- Intermediate calculations
- Temporary transformations
- Data explicitly marked as non-persistent

**Notes**:

  - This is a final class and cannot be subclassed
  - Use when you explicitly want to prevent persistence
  - Useful for sensitive data that shouldn't be written to disk

**See Also**:

- `FlowDocument` - For documents that persist across flow runs
- `TaskDocument` - For documents temporary within task execution


## ai_pipeline_core.documents.document_list

Type-safe list container for Document objects.

### DocumentList

```python
class DocumentList(list[Document])
```

Type-safe container for Document objects.

Specialized list with validation and filtering for documents.

**Example**:

  >>> docs = DocumentList(validate_same_type=True)
  >>> docs.append(MyDocument(name="file.txt", content=b"data"))
  >>> doc = docs.get_by_name("file.txt")

#### DocumentList.__init__

```python
def __init__(self, documents: list[Document] | None = None, validate_same_type: bool = False, validate_duplicates: bool = False) -> None
```

Initialize DocumentList.

**Arguments**:

- `documents` - Initial list of documents.
- `validate_same_type` - Enforce same document type.
- `validate_duplicates` - Prevent duplicate filenames.

#### DocumentList.filter_by_type

```python
def filter_by_type(self, document_type: type[Document]) -> "DocumentList"
```

Filter documents by class type (including subclasses).

**Arguments**:

- `document_type` - The document class to filter for.

**Returns**:

  New DocumentList with filtered documents.

#### DocumentList.filter_by_types

```python
def filter_by_types(self, document_types: list[type[Document]]) -> "DocumentList"
```

Filter documents by multiple class types.

**Arguments**:

- `document_types` - List of document classes to filter for.

**Returns**:

  New DocumentList with filtered documents.

#### DocumentList.get_by_name

```python
def get_by_name(self, name: str) -> Document | None
```

Find a document by filename.

**Arguments**:

- `name` - The document filename to search for.

**Returns**:

  The first matching document, or None if not found.
