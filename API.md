# ai-pipeline-core API

* [ai\_pipeline\_core](#ai_pipeline_core)
* [ai\_pipeline\_core.exceptions](#ai_pipeline_core.exceptions)
  * [PipelineCoreError](#ai_pipeline_core.exceptions.PipelineCoreError)
  * [DocumentError](#ai_pipeline_core.exceptions.DocumentError)
  * [DocumentValidationError](#ai_pipeline_core.exceptions.DocumentValidationError)
  * [DocumentSizeError](#ai_pipeline_core.exceptions.DocumentSizeError)
  * [DocumentNameError](#ai_pipeline_core.exceptions.DocumentNameError)
  * [LLMError](#ai_pipeline_core.exceptions.LLMError)
  * [PromptError](#ai_pipeline_core.exceptions.PromptError)
  * [PromptRenderError](#ai_pipeline_core.exceptions.PromptRenderError)
  * [PromptNotFoundError](#ai_pipeline_core.exceptions.PromptNotFoundError)
  * [MimeTypeError](#ai_pipeline_core.exceptions.MimeTypeError)
* [ai\_pipeline\_core.llm.model\_response](#ai_pipeline_core.llm.model_response)
  * [ModelResponse](#ai_pipeline_core.llm.model_response.ModelResponse)
    * [content](#ai_pipeline_core.llm.model_response.ModelResponse.content)
  * [StructuredModelResponse](#ai_pipeline_core.llm.model_response.StructuredModelResponse)
    * [parsed](#ai_pipeline_core.llm.model_response.StructuredModelResponse.parsed)
* [ai\_pipeline\_core.llm.ai\_messages](#ai_pipeline_core.llm.ai_messages)
  * [AIMessageType](#ai_pipeline_core.llm.ai_messages.AIMessageType)
  * [AIMessages](#ai_pipeline_core.llm.ai_messages.AIMessages)
    * [get\_last\_message](#ai_pipeline_core.llm.ai_messages.AIMessages.get_last_message)
    * [get\_last\_message\_as\_str](#ai_pipeline_core.llm.ai_messages.AIMessages.get_last_message_as_str)
* [ai\_pipeline\_core.llm.model\_options](#ai_pipeline_core.llm.model_options)
  * [ModelOptions](#ai_pipeline_core.llm.model_options.ModelOptions)
* [ai\_pipeline\_core.llm.client](#ai_pipeline_core.llm.client)
  * [generate](#ai_pipeline_core.llm.client.generate)
  * [generate\_structured](#ai_pipeline_core.llm.client.generate_structured)
* [ai\_pipeline\_core.prompt\_manager](#ai_pipeline_core.prompt_manager)
  * [PromptManager](#ai_pipeline_core.prompt_manager.PromptManager)
    * [\_\_init\_\_](#ai_pipeline_core.prompt_manager.PromptManager.__init__)
    * [get](#ai_pipeline_core.prompt_manager.PromptManager.get)
* [ai\_pipeline\_core.tracing](#ai_pipeline_core.tracing)
  * [trace](#ai_pipeline_core.tracing.trace)
* [ai\_pipeline\_core.flow](#ai_pipeline_core.flow)
* [ai\_pipeline\_core.flow.options](#ai_pipeline_core.flow.options)
  * [FlowOptions](#ai_pipeline_core.flow.options.FlowOptions)
* [ai\_pipeline\_core.flow.config](#ai_pipeline_core.flow.config)
  * [FlowConfig](#ai_pipeline_core.flow.config.FlowConfig)
* [ai\_pipeline\_core.pipeline](#ai_pipeline_core.pipeline)
  * [pipeline\_task](#ai_pipeline_core.pipeline.pipeline_task)
  * [pipeline\_flow](#ai_pipeline_core.pipeline.pipeline_flow)
* [ai\_pipeline\_core.logging.logging\_config](#ai_pipeline_core.logging.logging_config)
  * [LoggingConfig](#ai_pipeline_core.logging.logging_config.LoggingConfig)
  * [setup\_logging](#ai_pipeline_core.logging.logging_config.setup_logging)
  * [get\_pipeline\_logger](#ai_pipeline_core.logging.logging_config.get_pipeline_logger)
* [ai\_pipeline\_core.logging](#ai_pipeline_core.logging)
* [ai\_pipeline\_core.logging.logging\_mixin](#ai_pipeline_core.logging.logging_mixin)
  * [LoggerMixin](#ai_pipeline_core.logging.logging_mixin.LoggerMixin)
  * [StructuredLoggerMixin](#ai_pipeline_core.logging.logging_mixin.StructuredLoggerMixin)
* [ai\_pipeline\_core.simple\_runner](#ai_pipeline_core.simple_runner)
* [ai\_pipeline\_core.settings](#ai_pipeline_core.settings)
  * [Settings](#ai_pipeline_core.settings.Settings)
  * [settings](#ai_pipeline_core.settings.settings)
* [ai\_pipeline\_core.prefect](#ai_pipeline_core.prefect)
* [ai\_pipeline\_core.documents.task\_document](#ai_pipeline_core.documents.task_document)
  * [TaskDocument](#ai_pipeline_core.documents.task_document.TaskDocument)
* [ai\_pipeline\_core.documents.document](#ai_pipeline_core.documents.document)
  * [Document](#ai_pipeline_core.documents.document.Document)
    * [MAX\_CONTENT\_SIZE](#ai_pipeline_core.documents.document.Document.MAX_CONTENT_SIZE)
    * [DESCRIPTION\_EXTENSION](#ai_pipeline_core.documents.document.Document.DESCRIPTION_EXTENSION)
    * [MARKDOWN\_LIST\_SEPARATOR](#ai_pipeline_core.documents.document.Document.MARKDOWN_LIST_SEPARATOR)
    * [\_\_init\_\_](#ai_pipeline_core.documents.document.Document.__init__)
    * [base\_type](#ai_pipeline_core.documents.document.Document.base_type)
    * [is\_flow](#ai_pipeline_core.documents.document.Document.is_flow)
    * [is\_task](#ai_pipeline_core.documents.document.Document.is_task)
    * [is\_temporary](#ai_pipeline_core.documents.document.Document.is_temporary)
    * [validate\_file\_name](#ai_pipeline_core.documents.document.Document.validate_file_name)
    * [id](#ai_pipeline_core.documents.document.Document.id)
    * [sha256](#ai_pipeline_core.documents.document.Document.sha256)
    * [size](#ai_pipeline_core.documents.document.Document.size)
    * [mime\_type](#ai_pipeline_core.documents.document.Document.mime_type)
    * [is\_text](#ai_pipeline_core.documents.document.Document.is_text)
    * [is\_pdf](#ai_pipeline_core.documents.document.Document.is_pdf)
    * [is\_image](#ai_pipeline_core.documents.document.Document.is_image)
    * [text](#ai_pipeline_core.documents.document.Document.text)
    * [as\_yaml](#ai_pipeline_core.documents.document.Document.as_yaml)
    * [as\_json](#ai_pipeline_core.documents.document.Document.as_json)
    * [as\_pydantic\_model](#ai_pipeline_core.documents.document.Document.as_pydantic_model)
    * [as\_markdown\_list](#ai_pipeline_core.documents.document.Document.as_markdown_list)
    * [parsed](#ai_pipeline_core.documents.document.Document.parsed)
* [ai\_pipeline\_core.documents](#ai_pipeline_core.documents)
* [ai\_pipeline\_core.documents.flow\_document](#ai_pipeline_core.documents.flow_document)
  * [FlowDocument](#ai_pipeline_core.documents.flow_document.FlowDocument)
* [ai\_pipeline\_core.documents.temporary\_document](#ai_pipeline_core.documents.temporary_document)
  * [TemporaryDocument](#ai_pipeline_core.documents.temporary_document.TemporaryDocument)
* [ai\_pipeline\_core.documents.document\_list](#ai_pipeline_core.documents.document_list)
  * [DocumentList](#ai_pipeline_core.documents.document_list.DocumentList)
    * [filter\_by\_type](#ai_pipeline_core.documents.document_list.DocumentList.filter_by_type)
    * [filter\_by\_types](#ai_pipeline_core.documents.document_list.DocumentList.filter_by_types)
    * [get\_by\_name](#ai_pipeline_core.documents.document_list.DocumentList.get_by_name)

<a id="ai_pipeline_core"></a>

# ai\_pipeline\_core

AI Pipeline Core - High-performance async library for AI pipelines.

@public

AI Pipeline Core provides a comprehensive framework for building production-ready
AI pipelines with strong typing, observability, and efficient LLM integration.
Built on Prefect for orchestration and LiteLLM for model access.

Key Features:
- **Async-first**: All I/O operations are async for maximum performance
- **Type safety**: Pydantic models and strong typing throughout
- **Document abstraction**: Unified handling of text, images, PDFs
- **LLM integration**: Smart context caching and structured outputs
- **Observability**: Built-in tracing with LMNR (Laminar)
- **Orchestration**: Prefect integration for flows and tasks
- **Simple runner**: Local execution without full orchestration

Core Components:
Documents: Type-safe document handling with automatic encoding
LLM: Unified interface for language models via LiteLLM proxy
Flow/Task: Pipeline orchestration with Prefect integration
Tracing: Distributed tracing and observability with LMNR
Logging: Unified logging with Prefect integration
Settings: Centralized configuration management

Quick Start:
>>> from ai_pipeline_core import (
...     pipeline_flow,
...     FlowDocument,
...     DocumentList,
...     FlowOptions,
...     ModelOptions,
...     llm
... )
>>>
>>> class InputDoc(FlowDocument):
...     '''Input document for analysis.'''
>>>
>>> @pipeline_flow
>>> async def analyze_flow(
...     project_name: str,
...     documents: DocumentList,
...     flow_options: FlowOptions
... ) -> DocumentList:
...     # Your pipeline logic here
...     response = await llm.generate(
...         model="gpt-5",
...         messages=documents[0].text
...     )
...     return DocumentList([...])

Environment Setup:
Required environment variables:
- OPENAI_BASE_URL: LiteLLM proxy endpoint
- OPENAI_API_KEY: API key for LiteLLM

Optional:
- PREFECT_API_URL: Prefect server for orchestration
- LMNR_PROJECT_API_KEY: Laminar project key for tracing

Documentation:
Full documentation: https://github.com/jxnl/ai-pipeline-core
Examples: See examples/ directory in repository

Version History:
- 0.1.10: Enhanced CLI error handling, improved documentation
- 0.1.9: Document class enhancements with type safety
- 0.1.8: Major refactoring and validation improvements
- 0.1.7: Pipeline decorators and simple runner module

Current Version: 0.1.10

<a id="ai_pipeline_core.exceptions"></a>

# ai\_pipeline\_core.exceptions

Exception hierarchy for AI Pipeline Core.

@public

This module defines the exception hierarchy used throughout the AI Pipeline Core library.
All exceptions inherit from PipelineCoreError, providing a consistent error handling interface.

<a id="ai_pipeline_core.exceptions.PipelineCoreError"></a>

## PipelineCoreError Objects

```python
class PipelineCoreError(Exception)
```

Base exception for all AI Pipeline Core errors.

@public

<a id="ai_pipeline_core.exceptions.DocumentError"></a>

## DocumentError Objects

```python
class DocumentError(PipelineCoreError)
```

Base exception for document-related errors.

@public

<a id="ai_pipeline_core.exceptions.DocumentValidationError"></a>

## DocumentValidationError Objects

```python
class DocumentValidationError(DocumentError)
```

Raised when document validation fails.

@public

<a id="ai_pipeline_core.exceptions.DocumentSizeError"></a>

## DocumentSizeError Objects

```python
class DocumentSizeError(DocumentValidationError)
```

Raised when document content exceeds MAX_CONTENT_SIZE limit.

@public

<a id="ai_pipeline_core.exceptions.DocumentNameError"></a>

## DocumentNameError Objects

```python
class DocumentNameError(DocumentValidationError)
```

Raised when document name contains invalid characters or patterns.

@public

<a id="ai_pipeline_core.exceptions.LLMError"></a>

## LLMError Objects

```python
class LLMError(PipelineCoreError)
```

Raised when LLM generation fails after all retries.

@public

<a id="ai_pipeline_core.exceptions.PromptError"></a>

## PromptError Objects

```python
class PromptError(PipelineCoreError)
```

Base exception for prompt template errors.

@public

<a id="ai_pipeline_core.exceptions.PromptRenderError"></a>

## PromptRenderError Objects

```python
class PromptRenderError(PromptError)
```

Raised when Jinja2 template rendering fails.

@public

<a id="ai_pipeline_core.exceptions.PromptNotFoundError"></a>

## PromptNotFoundError Objects

```python
class PromptNotFoundError(PromptError)
```

Raised when prompt template file is not found in search paths.

@public

<a id="ai_pipeline_core.exceptions.MimeTypeError"></a>

## MimeTypeError Objects

```python
class MimeTypeError(DocumentError)
```

Raised when MIME type detection or validation fails.

@public

<a id="ai_pipeline_core.llm.model_response"></a>

# ai\_pipeline\_core.llm.model\_response

Model response structures for LLM interactions.

@public

Provides enhanced response classes that wrap OpenAI API responses
with additional metadata, cost tracking, and structured output support.

<a id="ai_pipeline_core.llm.model_response.ModelResponse"></a>

## ModelResponse Objects

```python
class ModelResponse(ChatCompletion)
```

Enhanced response wrapper for LLM text generation.

@public

Binary compatible with OpenAI ChatCompletion response format. All LLM provider
responses are normalized to this format by LiteLLM proxy, ensuring consistent
interface across providers (OpenAI, Anthropic, Google, Grok, etc.).

Additional Attributes:
headers: HTTP response headers including cost information.
model_options: Configuration used for this generation.

Key Properties:
content: Quick access to generated text content.
usage: Token usage statistics (inherited).
model: Model identifier used (inherited).
id: Unique response ID (inherited).

**Example**:

  >>> response = await generate("gpt-5", messages="Hello")
  >>> print(response.content)  # Generated text
  >>> print(response.usage.total_tokens)  # Token count
  >>> print(response.headers.get("x-litellm-response-cost"))  # Cost
  

**Notes**:

  This class maintains full compatibility with ChatCompletion
  while adding pipeline-specific functionality.

<a id="ai_pipeline_core.llm.model_response.ModelResponse.content"></a>

#### content

```python
@property
def content() -> str
```

Get the generated text content.

@public

Convenience property for accessing the first choice's message
content. Returns empty string if no content available.

**Returns**:

  Generated text from the first choice, or empty string.
  

**Example**:

  >>> response = await generate("gpt-5", messages="Hello")
  >>> text = response.content  # Direct access to generated text

<a id="ai_pipeline_core.llm.model_response.StructuredModelResponse"></a>

## StructuredModelResponse Objects

```python
class StructuredModelResponse(ModelResponse, Generic[T])
```

Response wrapper for structured/typed LLM output.

@public

Binary compatible with OpenAI ChatCompletion response format. Extends ModelResponse
with type-safe access to parsed Pydantic model instances.

Type Parameter:
T: The Pydantic model type for the structured output.

Additional Features:
- Type-safe access to parsed Pydantic model
- Automatic extraction from ParsedChatCompletion
- All features of ModelResponse (cost, metadata, etc.)

**Example**:

  >>> from pydantic import BaseModel
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

<a id="ai_pipeline_core.llm.model_response.StructuredModelResponse.parsed"></a>

#### parsed

```python
@property
def parsed() -> T
```

Get the parsed Pydantic model instance.

@public

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

<a id="ai_pipeline_core.llm.ai_messages"></a>

# ai\_pipeline\_core.llm.ai\_messages

AI message handling for LLM interactions.

@public

Provides AIMessages container for managing conversations with mixed content types
including text, documents, and model responses.

<a id="ai_pipeline_core.llm.ai_messages.AIMessageType"></a>

#### AIMessageType

Type for messages in AIMessages container.

@public

Represents the allowed types for conversation messages:
- str: Plain text messages
- Document: Structured document content
- ModelResponse: LLM generation responses

<a id="ai_pipeline_core.llm.ai_messages.AIMessages"></a>

## AIMessages Objects

```python
class AIMessages(list[AIMessageType])
```

Container for AI conversation messages supporting mixed types.

@public

This class extends list to manage conversation messages between user
and AI, supporting text, Document objects, and ModelResponse instances.
Messages are converted to OpenAI-compatible format for LLM interactions.

**Example**:

  >>> messages = AIMessages()
  >>> messages.append("What is the capital of France?")
  >>> messages.append(ModelResponse(content="The capital of France is Paris."))
  >>> prompt = messages.to_prompt()

<a id="ai_pipeline_core.llm.ai_messages.AIMessages.get_last_message"></a>

#### get\_last\_message

```python
def get_last_message() -> AIMessageType
```

Get the last message in the conversation.

@public

**Returns**:

  The last message in the conversation, which can be a string,
  Document, or ModelResponse.

<a id="ai_pipeline_core.llm.ai_messages.AIMessages.get_last_message_as_str"></a>

#### get\_last\_message\_as\_str

```python
def get_last_message_as_str() -> str
```

Get the last message as a string, raising if not a string.

@public

**Returns**:

  The last message as a string.
  

**Raises**:

- `ValueError` - If the last message is not a string.

<a id="ai_pipeline_core.llm.model_options"></a>

# ai\_pipeline\_core.llm.model\_options

Configuration options for LLM generation.

@public

Provides the ModelOptions class for configuring model behavior,
retry logic, and advanced features like web search and reasoning.

<a id="ai_pipeline_core.llm.model_options.ModelOptions"></a>

## ModelOptions Objects

```python
class ModelOptions(BaseModel)
```

Configuration options for LLM generation requests.

@public

ModelOptions encapsulates all configuration parameters for model
generation, including model behavior settings, retry logic, and
advanced features. All fields are optional with sensible defaults.

**Attributes**:

- `temperature` - Controls randomness in generation (0.0-2.0).
  Lower values = more deterministic, higher = more creative.
  None uses model default (usually 1.0).
  
- `system_prompt` - System-level instructions for the model.
  Sets the model's behavior and persona.
  
- `search_context_size` - Web search result depth for search-enabled models.
- `"low"` - Minimal context (~1-2 results)
- `"medium"` - Moderate context (~3-5 results)
- `"high"` - Extensive context (~6+ results)
  
- `reasoning_effort` - Reasoning intensity for O1-style models.
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
  
- `response_format` - Pydantic model for structured output.
  Set automatically by generate_structured().
  

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

<a id="ai_pipeline_core.llm.client"></a>

# ai\_pipeline\_core.llm.client

LLM client implementation for AI model interactions.

@public

This module provides the core functionality for interacting with language models
through a unified interface. It handles retries, caching, structured outputs,
and integration with various LLM providers via LiteLLM.

Key functions:
- generate(): Text generation with optional context caching
- generate_structured(): Type-safe structured output generation

<a id="ai_pipeline_core.llm.client.generate"></a>

#### generate

```python
@trace(ignore_inputs=["context"])
async def generate(model: ModelName | str,
                   *,
                   context: AIMessages | None = None,
                   messages: AIMessages | str,
                   options: ModelOptions | None = None) -> ModelResponse
```

Generate text response from a language model.

@public

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
  - Retry adds 10s delay between attempts
  
  Caching Implementation:
  Context caching uses OpenAI's native cache_control with ephemeral TTL.
  The cache is managed by the LLM provider (not locally stored).
  Cache key is generated from context hash via get_prompt_cache_key().
  TTL is fixed at 120 seconds and not configurable (provider limitation).
  

**Notes**:

  - Context is traced but ignored in logs for brevity
  - All models are accessed via LiteLLM proxy
  - Automatic retry with fixed delay (default 10s between attempts)
  - Cost tracking via response headers
  

**See Also**:

  - generate_structured: For typed/structured output
  - AIMessages: Message container with document support
  - ModelOptions: Configuration options

<a id="ai_pipeline_core.llm.client.generate_structured"></a>

#### generate\_structured

```python
@trace(ignore_inputs=["context"])
async def generate_structured(
        model: ModelName | str,
        response_format: type[T],
        *,
        context: AIMessages | None = None,
        messages: AIMessages | str,
        options: ModelOptions | None = None) -> StructuredModelResponse[T]
```

Generate structured output conforming to a Pydantic model.

@public

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

- `ValueError` - If model doesn't support structured output or parsing fails.
- `LLMError` - If generation fails after retries.
- `TypeError` - If response cannot be converted to response_format.
  

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
  Most modern models support structured output:
  - OpenAI: gpt-4o-mini and above
  - Anthropic: All Claude 3+ models
  - Google: Gemini Pro models
  - Others via LiteLLM compatibility
  
  Performance:
  - Structured output may use more tokens than free text
  - Complex schemas increase generation time
  - Validation overhead is minimal (Pydantic is fast)
  

**Notes**:

  - response_format is automatically added to options
  - The model is instructed to follow the schema exactly
  - Validation happens automatically via Pydantic
  - Use Field() descriptions to guide generation
  

**See Also**:

  - generate: For unstructured text generation
  - ModelOptions: Configuration including response_format
  - StructuredModelResponse: Response wrapper with .parsed property

<a id="ai_pipeline_core.prompt_manager"></a>

# ai\_pipeline\_core.prompt\_manager

Jinja2-based prompt template management system.

@public

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

<a id="ai_pipeline_core.prompt_manager.PromptManager"></a>

## PromptManager Objects

```python
class PromptManager()
```

Manages Jinja2 prompt templates with smart path resolution.

@public

PromptManager provides a convenient interface for loading and rendering
Jinja2 templates used as prompts for LLMs. It automatically searches for
templates in multiple locations, supporting both local (module-specific)
and shared (project-wide) templates.

Search hierarchy:
1. Same directory as the calling module (for local templates)
2. 'prompts' subdirectory in the calling module's directory
3. 'prompts' directories in parent packages (up to 4 levels)

Search Stopping Rule:
The search traverses UP TO 4 parent levels OR until it hits a
directory without __init__.py (package boundary), whichever comes
first. This prevents searching outside the package structure.

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

<a id="ai_pipeline_core.prompt_manager.PromptManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__(current_dir: str, prompts_dir: str = "prompts")
```

Initialize PromptManager with smart template discovery.

@public

Sets up the Jinja2 environment with a FileSystemLoader that searches
multiple directories for templates. The search starts from the calling
module's location and extends to parent package directories.

**Arguments**:

- `current_dir` - The __file__ path of the calling module. Must be
  a valid file path (not __name__). Used as the
  starting point for template discovery.
- `prompts_dir` - Name of the prompts subdirectory to search for
  in each package level. Defaults to "prompts".
  

**Raises**:

- `PromptError` - If current_dir is not a valid file path (e.g.,
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

<a id="ai_pipeline_core.prompt_manager.PromptManager.get"></a>

#### get

```python
def get(prompt_path: str, **kwargs: Any) -> str
```

Load and render a Jinja2 template with the given context.

@public

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

<a id="ai_pipeline_core.tracing"></a>

# ai\_pipeline\_core.tracing

Tracing utilities that integrate Laminar (``lmnr``) with our code-base.

@public

This module centralises:
• ``TraceInfo`` - a small helper object for propagating contextual metadata.
• ``trace`` decorator - augments a callable with Laminar tracing, automatic
``observe`` instrumentation, and optional support for test runs.

<a id="ai_pipeline_core.tracing.trace"></a>

#### trace

```python
def trace(
    func: Callable[P, R] | None = None,
    *,
    level: TraceLevel = "always",
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    span_type: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    ignore_exceptions: bool = False,
    preserve_global_context: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]
```

Add Laminar observability tracing to any function.

@public

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

<a id="ai_pipeline_core.flow"></a>

# ai\_pipeline\_core.flow

Flow configuration and options for Prefect-based pipeline flows.

@public

This package provides type-safe flow configuration with input/output document type validation.

<a id="ai_pipeline_core.flow.options"></a>

# ai\_pipeline\_core.flow.options

Flow options configuration for pipeline execution.

@public

Provides base configuration settings for AI pipeline flows,
including model selection and runtime parameters.

<a id="ai_pipeline_core.flow.options.FlowOptions"></a>

## FlowOptions Objects

```python
class FlowOptions(BaseSettings)
```

Base configuration settings for AI pipeline flows.

@public

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

<a id="ai_pipeline_core.flow.config"></a>

# ai\_pipeline\_core.flow.config

Flow configuration system for type-safe pipeline definitions.

@public

This module provides the FlowConfig abstract base class that enforces
type safety for flow inputs and outputs in the pipeline system.

<a id="ai_pipeline_core.flow.config.FlowConfig"></a>

## FlowConfig Objects

```python
class FlowConfig(ABC)
```

Abstract base class for type-safe flow configuration.

@public

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

<a id="ai_pipeline_core.pipeline"></a>

# ai\_pipeline\_core.pipeline

Pipeline decorators for Prefect integration.

@public

Tiny wrappers around Prefect's public ``@task`` and ``@flow`` that add our
``trace`` decorator and **require async functions**.

Why this exists
---------------
Prefect tasks/flows are awaitable at runtime, but their public type stubs
don't declare that clearly. We therefore:

1) Return the **real Prefect objects** (so you keep every Prefect method).
2) Type them as small Protocols that say "this is awaitable and has common
   helpers like `.submit`/`.map`”.

This keeps Pyright happy without altering runtime behavior and avoids
leaking advanced typing constructs (like ``ParamSpec``) that confuse tools
that introspect callables (e.g., Pydantic).

Quick start
-----------
from ai_pipeline_core.pipeline import pipeline_task, pipeline_flow
from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.flow.options import FlowOptions

@pipeline_task
async def add(x: int, y: int) -> int:
    return x + y

@pipeline_flow
async def my_flow(project_name: str, docs: DocumentList, opts: FlowOptions) -> DocumentList:
    await add(1, 2)  # awaitable and typed
    return docs

Rules
-----
• Your decorated function **must** be ``async def``.
• ``@pipeline_flow`` functions must accept at least:
  (project_name: str, documents: DocumentList, flow_options: FlowOptions | subclass).
• Both wrappers return the same Prefect objects you'd get from Prefect directly.

<a id="ai_pipeline_core.pipeline.pipeline_task"></a>

#### pipeline\_task

```python
def pipeline_task(
    __fn: Callable[..., Coroutine[Any, Any, R_co]] | None = None,
    *,
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | None = None,
    version: str | None = None,
    cache_policy: CachePolicy | type[NotSet] = NotSet,
    cache_key_fn: Callable[[TaskRunContext, dict[str, Any]], str | None]
    | None = None,
    cache_expiration: datetime.timedelta | None = None,
    task_run_name: TaskRunNameValueOrCallable | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | list[float]
    | Callable[[int], list[float]] | None = None,
    retry_jitter_factor: float | None = None,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    result_storage_key: str | None = None,
    cache_result_in_memory: bool = True,
    timeout_seconds: int | float | None = None,
    log_prints: bool | None = False,
    refresh_cache: bool | None = None,
    on_completion: list[StateHookCallable] | None = None,
    on_failure: list[StateHookCallable] | None = None,
    retry_condition_fn: RetryConditionCallable | None = None,
    viz_return_value: bool | None = None,
    asset_deps: list[str | Asset] | None = None
) -> _TaskLike[R_co] | Callable[[Callable[..., Coroutine[Any, Any, R_co]]],
                                _TaskLike[R_co]]
```

Decorate an async function as a traced Prefect task.

@public

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

<a id="ai_pipeline_core.pipeline.pipeline_flow"></a>

#### pipeline\_flow

```python
def pipeline_flow(
    __fn: _DocumentsFlowCallable[FO_contra] | None = None,
    *,
    trace_level: TraceLevel = "always",
    trace_ignore_input: bool = False,
    trace_ignore_output: bool = False,
    trace_ignore_inputs: list[str] | None = None,
    trace_input_formatter: Callable[..., str] | None = None,
    trace_output_formatter: Callable[..., str] | None = None,
    name: str | None = None,
    version: str | None = None,
    flow_run_name: Union[Callable[[], str], str] | None = None,
    retries: int | None = None,
    retry_delay_seconds: int | float | None = None,
    task_runner: TaskRunner[PrefectFuture[Any]] | None = None,
    description: str | None = None,
    timeout_seconds: int | float | None = None,
    validate_parameters: bool = True,
    persist_result: bool | None = None,
    result_storage: ResultStorage | str | None = None,
    result_serializer: ResultSerializer | str | None = None,
    cache_result_in_memory: bool = True,
    log_prints: bool | None = None,
    on_completion: list[FlowStateHook[Any, Any]] | None = None,
    on_failure: list[FlowStateHook[Any, Any]] | None = None,
    on_cancellation: list[FlowStateHook[Any, Any]] | None = None,
    on_crashed: list[FlowStateHook[Any, Any]] | None = None,
    on_running: list[FlowStateHook[Any, Any]] | None = None
) -> _FlowLike[FO_contra] | Callable[[_DocumentsFlowCallable[FO_contra]],
                                     _FlowLike[FO_contra]]
```

Decorate an async flow for document processing.

@public

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

<a id="ai_pipeline_core.logging.logging_config"></a>

# ai\_pipeline\_core.logging.logging\_config

Centralized logging configuration for AI Pipeline Core.

@public

This module provides logging configuration management that integrates
with Prefect's logging system. It supports both YAML-based configuration
and programmatic setup with sensible defaults.

Key features:
- Prefect-integrated logging for flows and tasks
- YAML configuration file support
- Environment variable overrides
- Component-specific log levels
- Automatic logger creation with proper formatting

Usage:
    >>> from ai_pipeline_core.logging import get_pipeline_logger
    >>> logger = get_pipeline_logger(__name__)
    >>> logger.info("Processing started")

Environment variables:
    AI_PIPELINE_LOGGING_CONFIG: Path to custom logging.yml
    AI_PIPELINE_LOG_LEVEL: Default log level (INFO, DEBUG, etc.)
    PREFECT_LOGGING_LEVEL: Prefect's logging level
    PREFECT_LOGGING_SETTINGS_PATH: Alternative config path

<a id="ai_pipeline_core.logging.logging_config.LoggingConfig"></a>

## LoggingConfig Objects

```python
class LoggingConfig()
```

Manages logging configuration for the pipeline.

@public

LoggingConfig provides centralized management of logging settings,
supporting both file-based and programmatic configuration. It
integrates seamlessly with Prefect's logging system.

**Attributes**:

- `config_path` - Path to YAML configuration file.
- `_config` - Cached configuration dictionary.
  
  Configuration precedence:
  1. Explicit config_path parameter
  2. AI_PIPELINE_LOGGING_CONFIG environment variable
  3. PREFECT_LOGGING_SETTINGS_PATH environment variable
  4. Default configuration
  

**Example**:

  >>> # Use default configuration
  >>> config = LoggingConfig()
  >>> config.apply()
  >>>
  >>> # Use custom config file
  >>> config = LoggingConfig(Path("custom_logging.yml"))
  >>> config.apply()
  

**Notes**:

  Configuration is lazy-loaded and cached after first access.

<a id="ai_pipeline_core.logging.logging_config.setup_logging"></a>

#### setup\_logging

```python
def setup_logging(config_path: Optional[Path] = None,
                  level: Optional[str] = None)
```

Setup logging for the AI Pipeline Core library.

@public

Initializes and applies logging configuration for the entire
pipeline system. This is the main entry point for logging setup
and should be called early in application initialization.

**Arguments**:

- `config_path` - Optional path to YAML logging configuration file.
  If None, uses environment variables or defaults.
- `level` - Optional log level override (INFO, DEBUG, WARNING, etc.).
  This overrides any level set in configuration or environment.
  
  Global effects:
  - Creates global LoggingConfig instance
  - Configures Python logging system
  - Sets Prefect logging environment variables
  - Overrides component log levels if level is specified
  

**Example**:

  >>> # Use default configuration
  >>> setup_logging()
  >>>
  >>> # Use custom config file
  >>> setup_logging(Path("/etc/myapp/logging.yml"))
  >>>
  >>> # Override log level for debugging
  >>> setup_logging(level="DEBUG")
  >>>
  >>> # Both config file and level override
  >>> setup_logging(Path("custom.yml"), level="WARNING")
  

**Notes**:

  This function is idempotent but will reconfigure logging
  each time it's called. Usually called once at startup.

<a id="ai_pipeline_core.logging.logging_config.get_pipeline_logger"></a>

#### get\_pipeline\_logger

```python
def get_pipeline_logger(name: str)
```

Get a logger for pipeline components.

@public

Factory function that returns a Prefect-integrated logger with
proper configuration. Automatically initializes logging if not
already configured.

**Arguments**:

- `name` - Logger name, typically __name__ or a module path
  like "ai_pipeline_core.documents". Follows Python's
  hierarchical naming convention.
  

**Returns**:

  Prefect logger instance with:
  - Proper formatting based on configuration
  - Integration with Prefect's flow/task logging
  - Automatic level inheritance from parent loggers
  

**Example**:

  >>> # In a module
  >>> logger = get_pipeline_logger(__name__)
  >>> logger.info("Module initialized")
  >>>
  >>> # With specific name
  >>> logger = get_pipeline_logger("ai_pipeline_core.custom")
  >>> logger.debug("Debug information", extra={"key": "value"})
  >>>
  >>> # In a Prefect flow
  >>> @flow
  >>> async def my_flow():
  ...     logger = get_pipeline_logger("flows.my_flow")
  ...     logger.info("Flow started")  # Appears in Prefect UI
  

**Notes**:

  Always use this function instead of Python's logging.getLogger()
  to ensure proper Prefect integration and configuration.

<a id="ai_pipeline_core.logging"></a>

# ai\_pipeline\_core.logging

Logging infrastructure for AI Pipeline Core.

@public

This module provides unified, Prefect-integrated logging with structured
output support. It replaces Python's standard logging module with a
pipeline-aware logging system.

Key components:
get_pipeline_logger: Factory function for creating pipeline loggers
setup_logging: Initialize logging configuration from YAML
LoggerMixin: Base mixin for adding logging to classes
StructuredLoggerMixin: Mixin for structured/JSON logging
LoggingConfig: Configuration class for logging settings

**Example**:

  >>> from ai_pipeline_core.logging import get_pipeline_logger
  >>>
  >>> logger = get_pipeline_logger(__name__)
  >>> logger.info("Processing started")
  >>>
  >>> # For structured logging in classes
  >>> from ai_pipeline_core.logging import StructuredLoggerMixin
  >>>
  >>> class MyProcessor(StructuredLoggerMixin):
  ...     def process(self):
  ...         self.logger.info("Processing", extra={"items": 100})
  

**Notes**:

  Never import Python's logging module directly. Always use
  get_pipeline_logger() for consistent Prefect integration.

<a id="ai_pipeline_core.logging.logging_mixin"></a>

# ai\_pipeline\_core.logging.logging\_mixin

Logging mixin for consistent logging across components using Prefect logging.

@public

<a id="ai_pipeline_core.logging.logging_mixin.LoggerMixin"></a>

## LoggerMixin Objects

```python
class LoggerMixin()
```

Mixin class that provides consistent logging functionality using Prefect's logging system.

@public

Automatically uses appropriate logger based on context:
- get_run_logger() when in flow/task context
- get_logger() when outside flow/task context

<a id="ai_pipeline_core.logging.logging_mixin.StructuredLoggerMixin"></a>

## StructuredLoggerMixin Objects

```python
class StructuredLoggerMixin(LoggerMixin)
```

Extended mixin for structured logging with Prefect.

@public

<a id="ai_pipeline_core.simple_runner"></a>

# ai\_pipeline\_core.simple\_runner

Simple pipeline execution framework for local development.

@public

The simple_runner module provides utilities for running AI pipelines
locally without full Prefect orchestration. It's designed for rapid
prototyping, testing, and simple workflows that don't need distributed
execution or advanced scheduling.

Key features:
- Sequential flow execution without Prefect server
- Directory-based document loading and saving
- CLI interface for command-line execution
- Flow chaining with automatic document passing
- Test environment detection and configuration

Main components:
run_pipeline: Execute a single flow with documents
run_pipelines: Execute multiple flows in sequence
run_cli: Command-line interface for flow execution
load_documents_from_directory: Load documents from filesystem
save_documents_to_directory: Save results to filesystem
FlowSequence: Type for flow + config pairs
ConfigSequence: Type for config class + options pairs

**Example**:

  >>> from ai_pipeline_core.simple_runner import run_pipeline
  >>> from my_flows import AnalysisFlow, AnalysisConfig
  >>>
  >>> # Run single flow
  >>> results = await run_pipeline(
  ...     project_name="test_project",
  ...     flow=AnalysisFlow,
  ...     config_class=AnalysisConfig,
  ...     input_dir="./data",
  ...     output_dir="./results"
  ... )
  >>>
  >>> # Or use CLI
  >>> # python -m my_module --input-dir ./data --output-dir ./results
  

**Notes**:

  Simple runner is for local development. For production,
  use Prefect deployment with proper orchestration.

<a id="ai_pipeline_core.settings"></a>

# ai\_pipeline\_core.settings

Core configuration settings for pipeline operations.

@public

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

<a id="ai_pipeline_core.settings.Settings"></a>

## Settings Objects

```python
class Settings(BaseSettings)
```

Core configuration for AI Pipeline external services.

@public

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

<a id="ai_pipeline_core.settings.settings"></a>

#### settings

Global settings instance for the entire application.

@public

This singleton instance is created at module import and provides
configuration to all pipeline components. Access this instance
rather than creating new Settings objects.

**Example**:

  >>> from ai_pipeline_core.settings import settings
  >>> print(f"Using LLM proxy at {settings.openai_base_url}")

<a id="ai_pipeline_core.prefect"></a>

# ai\_pipeline\_core.prefect

Prefect core features for pipeline orchestration.

@public

This module provides clean re-exports of essential Prefect functionality.

IMPORTANT: You should NEVER use the `task` and `flow` decorators directly
unless it is 100% impossible to use `pipeline_task` and `pipeline_flow`.
The standard Prefect decorators are exported here only for extremely
limited edge cases where the pipeline decorators cannot be used.

Always prefer:
>>> from ai_pipeline_core import pipeline_task, pipeline_flow
>>>
>>> @pipeline_task
>>> async def my_task(...): ...
>>>
>>> @pipeline_flow
>>> async def my_flow(...): ...

The `task` and `flow` decorators should only be used when:
- You absolutely cannot convert to async (pipeline decorators require async)
- You have a very specific Prefect integration that conflicts with tracing
- You are writing test utilities or infrastructure code

Exported components:
task: Prefect task decorator (AVOID - use pipeline_task instead).
flow: Prefect flow decorator (AVOID - use pipeline_flow instead).
disable_run_logger: Context manager to suppress Prefect logging.
prefect_test_harness: Test harness for unit testing flows/tasks.

Testing utilities (use as fixtures):
The disable_run_logger and prefect_test_harness should be used as
pytest fixtures as shown in tests/conftest.py:

>>> @pytest.fixture(autouse=True, scope="session")
>>> def prefect_test_fixture():
...     with prefect_test_harness():
...         yield
>>>
>>> @pytest.fixture(autouse=True)
>>> def disable_prefect_logging():
...     with disable_run_logger():
...         yield

**Notes**:

  The pipeline_task and pipeline_flow decorators from
  ai_pipeline_core.pipeline provide async-only execution with
  integrated LMNR tracing and are the standard for this library.

<a id="ai_pipeline_core.documents.task_document"></a>

# ai\_pipeline\_core.documents.task\_document

Task-specific document base class for temporary pipeline data.

@public

This module provides the TaskDocument abstract base class for documents
that exist only during Prefect task execution and are not persisted.

<a id="ai_pipeline_core.documents.task_document.TaskDocument"></a>

## TaskDocument Objects

```python
class TaskDocument(Document)
```

Abstract base class for temporary documents within task execution.

@public

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

Usage:
Always subclass TaskDocument for temporary document types:

>>> class TempProcessingDoc(TaskDocument):
...     def get_type(self) -> str:
...         return "temp_processing"
>>> doc = TempProcessingDoc(name="temp.json", content=b'{}')

**Notes**:

  - Cannot instantiate TaskDocument directly - must subclass
  - Not saved by simple_runner utilities
  - Useful for data transformations within tasks
  - Reduces I/O overhead for temporary data
  

**See Also**:

- `FlowDocument` - For documents that persist across flow runs
- `TemporaryDocument` - Alternative for non-persistent documents

<a id="ai_pipeline_core.documents.document"></a>

# ai\_pipeline\_core.documents.document

Document abstraction layer for AI pipeline flows.

@public

This module provides the core document abstraction for working with various types of data
in AI pipelines. Documents are immutable Pydantic models that wrap binary content with metadata.

<a id="ai_pipeline_core.documents.document.Document"></a>

## Document Objects

```python
class Document(BaseModel, ABC)
```

Abstract base class for all documents in the AI Pipeline Core system.

@public

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

  >>> class MyDocument(FlowDocument):
  ...     def get_type(self) -> str:
  ...         return "my_doc"
  >>> doc = MyDocument(name="data.json", content=b'{"key": "value"}')
  >>> print(doc.is_text)  # True
  >>> data = doc.as_json()  # {'key': 'value'}

<a id="ai_pipeline_core.documents.document.Document.MAX_CONTENT_SIZE"></a>

#### MAX\_CONTENT\_SIZE

Maximum allowed content size in bytes (default 25MB).

@public

<a id="ai_pipeline_core.documents.document.Document.DESCRIPTION_EXTENSION"></a>

#### DESCRIPTION\_EXTENSION

File extension for description files.

@public

<a id="ai_pipeline_core.documents.document.Document.MARKDOWN_LIST_SEPARATOR"></a>

#### MARKDOWN\_LIST\_SEPARATOR

Separator for markdown list items.

@public

<a id="ai_pipeline_core.documents.document.Document.__init__"></a>

#### \_\_init\_\_

```python
def __init__(**data: Any) -> None
```

Initialize a Document instance.

@public

Prevents direct instantiation of the abstract Document class.
Content can be passed as either str or bytes (automatically converted to bytes).
Handles legacy signature where content was passed as description.

**Arguments**:

- `**data` - Keyword arguments for fields:
  - name (str): Document filename
  - description (str | None): Optional description (or content in legacy)
  - content (str | bytes | None): Content (automatically converted to bytes)
  

**Raises**:

- `TypeError` - If attempting to instantiate Document directly.
  

**Example**:

  >>> doc = MyDocument(name="test.txt", content="Hello World")
  >>> doc.content  # b'Hello World'
  >>> # Legacy: content as description
  >>> doc = MyDocument(name="test.txt", description=b"data", content=None)
  >>> doc.content  # b'data'

<a id="ai_pipeline_core.documents.document.Document.base_type"></a>

#### base\_type

```python
@final
@property
def base_type() -> Literal["flow", "task", "temporary"]
```

Get the document's base type.

@public

Property alias for get_base_type() providing a cleaner API.
This property cannot be overridden by subclasses.

**Returns**:

  The document's base type: "flow", "task", or "temporary".

<a id="ai_pipeline_core.documents.document.Document.is_flow"></a>

#### is\_flow

```python
@final
@property
def is_flow() -> bool
```

Check if this is a flow document.

@public

Flow documents persist across Prefect flow runs and are saved
to the file system between pipeline steps.

**Returns**:

  True if this is a FlowDocument subclass, False otherwise.

<a id="ai_pipeline_core.documents.document.Document.is_task"></a>

#### is\_task

```python
@final
@property
def is_task() -> bool
```

Check if this is a task document.

@public

Task documents are temporary within Prefect task execution
and are not persisted between pipeline steps.

**Returns**:

  True if this is a TaskDocument subclass, False otherwise.

<a id="ai_pipeline_core.documents.document.Document.is_temporary"></a>

#### is\_temporary

```python
@final
@property
def is_temporary() -> bool
```

Check if this is a temporary document.

@public

Temporary documents are never persisted and exist only
during execution.

**Returns**:

  True if this is a TemporaryDocument, False otherwise.

<a id="ai_pipeline_core.documents.document.Document.validate_file_name"></a>

#### validate\_file\_name

```python
@classmethod
def validate_file_name(cls, name: str) -> None
```

Validate that a file name matches allowed patterns.

@public

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

<a id="ai_pipeline_core.documents.document.Document.id"></a>

#### id

```python
@final
@property
def id() -> str
```

Get a short unique identifier for the document.

@public

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

<a id="ai_pipeline_core.documents.document.Document.sha256"></a>

#### sha256

```python
@final
@cached_property
def sha256() -> str
```

Get the full SHA256 hash of the document content.

@public

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

<a id="ai_pipeline_core.documents.document.Document.size"></a>

#### size

```python
@final
@property
def size() -> int
```

Get the size of the document content.

@public

**Returns**:

  Size of content in bytes.
  

**Notes**:

  Useful for monitoring document sizes and
  ensuring they stay within limits.

<a id="ai_pipeline_core.documents.document.Document.mime_type"></a>

#### mime\_type

```python
@property
def mime_type() -> str
```

Get the document's MIME type.

@public

Primary property for accessing MIME type information.
Currently delegates to detected_mime_type but provides
a stable API for future enhancements.

**Returns**:

  MIME type string.
  

**Notes**:

  Prefer this over detected_mime_type for general use.

<a id="ai_pipeline_core.documents.document.Document.is_text"></a>

#### is\_text

```python
@property
def is_text() -> bool
```

Check if document contains text content.

@public

**Returns**:

  True if MIME type indicates text content
  (text/*, application/json, application/yaml, etc.),
  False otherwise.
  

**Notes**:

  Used to determine if text property can be safely accessed.

<a id="ai_pipeline_core.documents.document.Document.is_pdf"></a>

#### is\_pdf

```python
@property
def is_pdf() -> bool
```

Check if document is a PDF file.

@public

**Returns**:

  True if MIME type is application/pdf, False otherwise.
  

**Notes**:

  PDF documents require special handling and are
  supported by certain LLM models.

<a id="ai_pipeline_core.documents.document.Document.is_image"></a>

#### is\_image

```python
@property
def is_image() -> bool
```

Check if document is an image file.

@public

**Returns**:

  True if MIME type starts with "image/", False otherwise.
  

**Notes**:

  Image documents are automatically encoded for
  vision-capable LLM models.

<a id="ai_pipeline_core.documents.document.Document.text"></a>

#### text

```python
@property
def text() -> str
```

Get document content as text string.

@public

Decodes the bytes content as UTF-8 text. Only works
for text-based documents.

**Returns**:

  UTF-8 decoded string.
  

**Raises**:

- `ValueError` - If document is not text-based (check is_text first).
  

**Example**:

  >>> doc = MyDocument(name="data.txt", content=b"Hello")
  >>> doc.text
  'Hello'

<a id="ai_pipeline_core.documents.document.Document.as_yaml"></a>

#### as\_yaml

```python
def as_yaml() -> Any
```

Parse document content as YAML.

@public

Converts text content to Python objects using YAML parser.
Document must be text-based.

**Returns**:

  Parsed YAML data (dict, list, or scalar).
  

**Notes**:

  Raises ValueError if document is not text (from text property).
  Uses ruamel.yaml which is safe by default (no arbitrary code execution).
  

**Example**:

  >>> doc = MyDocument(name="config.yaml", content=b"key: value")
  >>> doc.as_yaml()
- `{'key'` - 'value'}

<a id="ai_pipeline_core.documents.document.Document.as_json"></a>

#### as\_json

```python
def as_json() -> Any
```

Parse document content as JSON.

@public

Converts text content to Python objects using JSON parser.
Document must be text-based.

**Returns**:

  Parsed JSON data (dict, list, or scalar).
  

**Notes**:

  Raises ValueError if document is not text (from text property).
  Raises JSONDecodeError if content is not valid JSON (from json.loads()).
  

**Example**:

  >>> doc = MyDocument(name="data.json", content=b'{"key": "value"}')
  >>> doc.as_json()
- `{'key'` - 'value'}

<a id="ai_pipeline_core.documents.document.Document.as_pydantic_model"></a>

#### as\_pydantic\_model

```python
def as_pydantic_model(
        model_type: type[TModel] | type[list[TModel]]
) -> TModel | list[TModel]
```

Parse document content as a Pydantic model.

@public

Converts JSON or YAML content to validated Pydantic model instances.
Supports both single models and lists of models.

**Arguments**:

- `model_type` - The Pydantic model class or list[ModelClass] to parse into.
  

**Returns**:

  Validated Pydantic model instance or list of instances.
  

**Raises**:

- `ValueError` - If document is not text or data doesn't match expected type.
  

**Notes**:

  May raise ValidationError from Pydantic if data doesn't validate against the model.
  

**Example**:

  >>> class Config(BaseModel):
  ...     key: str
  >>> doc = MyDocument(name="config.json", content=b'{"key": "value"}')
  >>> config = doc.as_pydantic_model(Config)
  >>> config.key
  'value'

<a id="ai_pipeline_core.documents.document.Document.as_markdown_list"></a>

#### as\_markdown\_list

```python
def as_markdown_list() -> list[str]
```

Parse document as a markdown-separated list.

@public

Splits text content by the MARKDOWN_LIST_SEPARATOR
(default: "\n\n---\n\n") to extract list items.

**Returns**:

  List of string items.
  

**Notes**:

  Raises ValueError if document is not text (from text property).
  

**Example**:

  >>> content = "Item 1\n\n---\n\nItem 2"
  >>> doc = MyDocument(name="list.md", content=content.encode())
  >>> doc.as_markdown_list()
  ['Item 1', 'Item 2']

<a id="ai_pipeline_core.documents.document.Document.parsed"></a>

#### parsed

```python
def parsed(type_: type[Any]) -> Any
```

Parse document content based on file extension and requested type.

@public

Intelligently parses content based on the document's file extension
and converts to the requested type. This is designed to be the inverse
of document creation, so that Document(name=n, content=x).parsed(type(x)) == x.

**Arguments**:

- `type_` - Target type to parse content into. Supported types:
  - bytes: Returns raw content
  - str: Returns decoded text
  - dict: Parses JSON/YAML based on extension
  - list: For .md files, splits by markdown separator
  - BaseModel subclasses: Parses and validates JSON/YAML
  

**Returns**:

  Parsed content in the requested type.
  

**Raises**:

- `ValueError` - If the type is not supported or content cannot be parsed.
  

**Example**:

  >>> # String content
  >>> doc = MyDocument(name="test.txt", content="Hello")
  >>> doc.parsed(str)
  'Hello'
  
  >>> # JSON content
  >>> import json
  >>> data = {"key": "value"}
  >>> doc = MyDocument(name="data.json", content=json.dumps(data))
  >>> doc.parsed(dict)
- `{'key'` - 'value'}
  
  >>> # Markdown list
  >>> items = ["Item 1", "Item 2"]
  >>> content = "\n\n---\n\n".join(items)
  >>> doc = MyDocument(name="list.md", content=content)
  >>> doc.parsed(list)
  ['Item 1', 'Item 2']

<a id="ai_pipeline_core.documents"></a>

# ai\_pipeline\_core.documents

Document abstraction system for AI pipeline flows.

@public

The documents package provides immutable, type-safe data structures for handling
various content types in AI pipelines, including text, images, PDFs, and other
binary data with automatic MIME type detection.

<a id="ai_pipeline_core.documents.flow_document"></a>

# ai\_pipeline\_core.documents.flow\_document

Flow-specific document base class for persistent pipeline data.

@public

This module provides the FlowDocument abstract base class for documents
that need to persist across Prefect flow runs and between pipeline steps.

<a id="ai_pipeline_core.documents.flow_document.FlowDocument"></a>

## FlowDocument Objects

```python
class FlowDocument(Document)
```

Abstract base class for documents that persist across flow runs.

@public

FlowDocument is used for data that needs to be saved between pipeline
steps and across multiple flow executions. These documents are typically
written to the file system using the simple_runner utilities.

Key characteristics:
- Persisted to file system between pipeline steps
- Survives across multiple flow runs
- Used for flow inputs and outputs
- Saved in directories named after the document's canonical name

Usage:
Always subclass FlowDocument for your specific document types:

>>> class InputDataDocument(FlowDocument):
...     def get_type(self) -> str:
...         return "input_data"
>>> doc = InputDataDocument(name="data.json", content=b'{}')

**Notes**:

  - Cannot instantiate FlowDocument directly - must subclass
  - Documents are saved to {output_dir}/{canonical_name}/{filename}
  - Used with FlowConfig to define flow input/output types
  

**See Also**:

- `TaskDocument` - For temporary documents within task execution
- `TemporaryDocument` - For documents that are never persisted

<a id="ai_pipeline_core.documents.temporary_document"></a>

# ai\_pipeline\_core.documents.temporary\_document

Temporary document implementation for non-persistent data.

@public

This module provides the TemporaryDocument class for documents that
are never persisted, regardless of context.

<a id="ai_pipeline_core.documents.temporary_document.TemporaryDocument"></a>

## TemporaryDocument Objects

```python
@final
class TemporaryDocument(Document)
```

Concrete document class for data that is never persisted.

@public

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

Usage:
Can be instantiated directly without subclassing:

>>> doc = TemporaryDocument(
...     name="api_response.json",
...     content=b'{"status": "ok"}'
... )
>>> doc.is_temporary  # True

**Notes**:

  - This is a final class and cannot be subclassed
  - Use when you explicitly want to prevent persistence
  - Useful for sensitive data that shouldn't be written to disk
  - API responses, credentials, or intermediate calculations
  

**See Also**:

- `FlowDocument` - For documents that persist across flow runs
- `TaskDocument` - For documents temporary within task execution

<a id="ai_pipeline_core.documents.document_list"></a>

# ai\_pipeline\_core.documents.document\_list

Type-safe list container for Document objects with validation.

@public

This module provides the DocumentList class that ensures all items are valid Document
instances while maintaining list-like interface.

<a id="ai_pipeline_core.documents.document_list.DocumentList"></a>

## DocumentList Objects

```python
class DocumentList(list[Document])
```

Type-safe container for Document objects with validation.

@public

A specialized list that ensures document integrity and provides
convenient filtering and search operations. Used throughout the
pipeline system to pass documents between flows and tasks.

Key features:
- Optional duplicate filename validation
- Optional same-type validation (for flow outputs)
- Automatic validation on all modifications
- Filtering by document type
- Lookup by document name

**Attributes**:

- `_validate_same_type` - Whether to enforce all documents have same type
- `_validate_duplicates` - Whether to prevent duplicate filenames
  

**Example**:

  >>> docs = DocumentList(validate_same_type=True)
  >>> docs.append(MyDocument(name="file1.txt", content=b"data"))
  >>> docs.append(MyDocument(name="file2.txt", content=b"more"))
  >>> doc = docs.get_by_name("file1.txt")
  

**Notes**:

  Validation is performed automatically after every modification.
  For flow outputs, use validate_same_type=True to ensure consistency.

<a id="ai_pipeline_core.documents.document_list.DocumentList.filter_by_type"></a>

#### filter\_by\_type

```python
def filter_by_type(document_type: type[Document]) -> "DocumentList"
```

Filter documents by class type (including subclasses).

@public

Creates a new DocumentList containing documents that are
instances of the specified type (includes subclasses).

**Arguments**:

- `document_type` - The document class to filter for.
  

**Returns**:

  New DocumentList with filtered documents.
  

**Example**:

  >>> all_docs = DocumentList([flow_doc1, task_doc1, flow_doc2])
  >>> flow_docs = all_docs.filter_by_type(FlowDocument)
  >>> len(flow_docs)  # 2 (includes all FlowDocument subclasses)

<a id="ai_pipeline_core.documents.document_list.DocumentList.filter_by_types"></a>

#### filter\_by\_types

```python
def filter_by_types(document_types: list[type[Document]]) -> "DocumentList"
```

Filter documents by multiple class types.

@public

Creates a new DocumentList containing documents matching any
of the specified types.

**Arguments**:

- `document_types` - List of document classes to filter for.
  

**Returns**:

  New DocumentList with filtered documents.
  

**Example**:

  >>> docs = all_docs.filter_by_types([InputDoc, ConfigDoc])

<a id="ai_pipeline_core.documents.document_list.DocumentList.get_by_name"></a>

#### get\_by\_name

```python
def get_by_name(name: str) -> Document | None
```

Find a document by filename.

@public

Searches for the first document with the specified name.

**Arguments**:

- `name` - The document filename to search for.
  

**Returns**:

  The first matching document, or None if not found.
  

**Example**:

  >>> doc = docs.get_by_name("config.json")
  >>> if doc:
  ...     data = doc.as_json()

