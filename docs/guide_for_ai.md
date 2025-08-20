Of course. Here is the final, comprehensive guide for the `ai-pipeline-core` library, written as official documentation. It is detailed, contains extensive code examples, and does not reference the specific `research-pipeline` project.

---

# The Official `ai-pipeline-core` Developer's Guide

## Table of Contents

1.  **Introduction & Philosophy**
    *   What is `ai-pipeline-core`?
    *   Core Principles

2.  **Getting Started: Environment Setup**
    *   Installation
    *   Environment Variables (`.env`)
    *   **Crucial**: Setting up the LiteLLM Proxy

3.  **Core Concepts: A Deep Dive**
    *   **The Document System**: The Heart of Data Flow
        *   `Document`: The Abstract Base
        *   `FlowDocument`: Persistent State
        *   `TaskDocument`: Ephemeral Data
        *   `DocumentList`: The Type-Safe Container
    *   **The LLM Client**: Your Gateway to AI
        *   `generate()`: For Standard Text Generation
        *   `generate_structured()`: For Pydantic-backed Responses
        *   `AIMessages`: The Universal Input Formatter
        *   `ModelOptions`: Configuring AI Calls
        *   `ModelResponse` & `StructuredModelResponse`: Understanding Outputs
    *   **Flow Orchestration**: Building the Pipeline
        *   `FlowConfig`: The Contract for Your Flows
    *   **Observability & Utilities**
        *   `@trace`: Automatic Monitoring
        *   `PromptManager`: Organized & Reusable Prompts
        *   `settings`: Type-Safe Configuration
        *   `get_pipeline_logger`: Unified Logging

4.  **Practical Walkthrough: Building a Sentiment Analysis Flow**
    *   Step 1: Defining the Documents
    *   Step 2: Defining the `FlowConfig`
    *   Step 3: Creating the Prompt Template
    *   Step 4: Writing the Analysis Task
    *   Step 5: Composing the Prefect Flow
    *   Step 6: Running the Flow

5.  **Complete API Reference**
    *   Module: `ai_pipeline_core.documents`
    *   Module: `ai_pipeline_core.llm`
    *   Module: `ai_pipeline_core.flow`
    *   Module: `ai_pipeline_core.tracing`
    *   Module: `ai_pipeline_core.prompt_manager`
    *   Module: `ai_pipeline_core.settings`
    *   Module: `ai_pipeline_core.logging`

6.  **Best Practices & Advanced Patterns**
    *   Error Handling Strategies
    *   Creating Flexible Pipelines with Custom Options
    *   Testing Your Tasks and Flows
    *   DOs and DON'Ts

7.  **Troubleshooting & FAQ**

---

## 1. Introduction & Philosophy

### What is `ai-pipeline-core`?

`ai-pipeline-core` is a specialized Python library designed to be the foundational layer for building complex, production-grade AI data processing pipelines. It was created to solve common problems encountered in AI engineering: untyped data, blocking I/O, poor observability, and inconsistent patterns. It provides a set of robust, reusable components, allowing developers to focus exclusively on the unique business logic of their AI application.

### Core Principles

Every feature in `ai-pipeline-core` adheres to these principles:

1.  **100% Async Architecture**: All I/O operations (especially LLM calls) are asynchronous, built on `asyncio` and `httpx`. This is critical for building high-throughput systems that can perform many operations concurrently without being blocked.
2.  **Strong Typing & Validation**: Pydantic is used for all data structures. This eliminates runtime errors from malformed data and makes the codebase self-documenting. If the type-checker passes, the code is far more likely to be correct.
3.  **Minimalism & Abstraction**: The library provides the thinnest possible layer over powerful tools like Prefect, OpenAI-compatible APIs, and LMNR. Every line of code must justify its existence. There are no deep, complex abstractions to learn.
4.  **Production-Ready by Default**: Features like automatic retries, structured logging, and observability tracing are built-in, not afterthoughts.
5.  **Developer Experience for Experts**: The library is designed for experienced developers. It prioritizes clarity, consistency, and conciseness over verbose comments or boilerplate.

## 2. Getting Started: Environment Setup

Before you can build your first pipeline, you must set up your local environment to meet the expectations of `ai-pipeline-core`.

### Installation

Install the library directly from PyPI:
```bash
pip install ai-pipeline-core
```

### Environment Variables (`.env`)

Create a `.env` file in the root of your project or set enviromental variables.
If `.env` file exist it is loaded automatically by the `settings` object.

```dotenv
# .env

# [REQUIRED] LiteLLM Proxy Configuration
# This tells the library where to send all AI requests.
OPENAI_BASE_URL=http://localhost:4000/v1
OPENAI_API_KEY=your-litellm-proxy-key # Can be any string, used for auth if you configure it.

# [OPTIONAL] Observability
# Your project key from LMNR (lmnr.ai) for tracing.
LMNR_PROJECT_API_KEY=lmnr_...

# [OPTIONAL] Prefect Configuration
# Only needed if you are connecting to a Prefect server or cloud.
# Leave blank for the default local Prefect UI.
PREFECT_API_URL=
PREFECT_API_KEY=
```

### **Crucial**: Setting up the LiteLLM Proxy

`ai-pipeline-core` does not call OpenAI, Anthropic, or Google APIs directly. It speaks to a single, standardized **OpenAI-compatible API endpoint**. The recommended way to create this endpoint locally is with **LiteLLM**. LiteLLM acts as a translation layer, allowing you to use models from over 100 providers through a single, consistent API.

**Why is this required?** It decouples your code from specific model providers. You can switch from `gpt-5-mini` to `gemini-2.5-flash` just by changing a string, without altering any application code.

**Steps to run LiteLLM locally:**

1.  **Create a `litellm-config.yml` file** in your project root. This file contains your actual API keys for the model providers.

    ```yaml
    # litellm-config.yml

    model_list:
      - model_name: gpt-5-mini
        litellm_params:
          model: openai/gpt-4o-mini
          api_key: sk-proj-... # Your OpenAI API Key
      - model_name: gemini-2.5-flash
        litellm_params:
          model: gemini/flash-1.5
          api_key: your-google-api-key
      # Add any other models you want to use

    litellm_settings:
      drop_params: true # Helps prevent errors from unsupported parameters
    ```

2.  **Run LiteLLM using Docker (Recommended):**
    This is the easiest and most isolated way to run the proxy.

    ```bash
    docker run -d \
      --name litellm-proxy \
      -p 4000:4000 \
      -v $(pwd)/litellm-config.yml:/app/config.yaml \
      ghcr.io/berriai/litellm:main \
      --config /app/config.yaml \
      --port 4000 \
      --host 0.0.0.0
    ```
    Your `OPENAI_BASE_URL` should now point to `http://localhost:4000/v1`. The proxy is running and ready to receive requests from `ai-pipeline-core`.

---

## 3. Core Concepts: A Deep Dive

### **The Document System**: The Heart of Data Flow

#### `Document`: The Abstract Base
This is the abstract class that underpins all data objects. You will never instantiate it directly.

-   **Key Properties**:
    -   `name: str`: The filename of the document (e.g., `initial_whitepaper.md`).
    -   `content: bytes`: The raw, binary content of the document.
    -   `description: str | None`: An optional Markdown description.
    -   `id: str`: A unique, 6-character ID generated from a hash of the content. It's deterministic: the same content always produces the same ID.
    -   `sha256: str`: The full SHA256 hash of the content.
    -   `mime_type: str`: The automatically detected MIME type (e.g., `text/plain`, `application/pdf`, `image/png`).
    -   `is_text`, `is_pdf`, `is_image`: Boolean helpers for quick type checking.
-   **Key Methods**:
    -   `as_text() -> str`: Safely decodes the `content` as a UTF-8 string. Raises a `ValueError` if the document is binary.
    -   `as_json() -> Any`: Parses the text content as JSON.
    -   `as_yaml() -> Any`: Parses the text content as YAML.
    -   `as_pydantic_model(model_type: type[T]) -> T`: Parses the document content (JSON or YAML based on MIME type) and validates it as a Pydantic model instance.
    -   `as_markdown_list() -> list[str]`: Splits a Markdown document that uses `\n\n---\n\n` as a separator into a list of strings.

#### Creating Documents: The Smart `create` Method

The `Document.create()` class method is a powerful factory that intelligently handles multiple content types based on both the content type and file extension:

-   **Signature**: `create(name: str, description: str | None, content: bytes | str | BaseModel | list[str] | Any) -> Self`
-   **Smart Type Detection**:
    -   `bytes`: Used directly as document content
    -   `str`: Automatically encoded to UTF-8 bytes
    -   `list[str]` + `.md` extension: Creates a markdown list document using `create_as_markdown_list()`
    -   `dict/BaseModel` + `.json` extension: Serializes to formatted JSON using `create_as_json()`
    -   `dict/BaseModel` + `.yaml/.yml` extension: Serializes to formatted YAML using `create_as_yaml()`
    -   Other types: Raises `ValueError` with clear error message

**Usage Examples:**

```python
from pydantic import BaseModel
from ai_pipeline_core.documents import FlowDocument

class MyDocument(FlowDocument):
    pass

# Create from string - automatically encodes to UTF-8
doc1 = MyDocument.create("text.txt", "Simple text", "Hello, World!")

# Create from bytes - used directly
doc2 = MyDocument.create("binary.dat", None, b"\x00\x01\x02")

# Create JSON document from dict - automatically formatted
data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
doc3 = MyDocument.create("data.json", "User data", data)
# The content will be pretty-printed JSON with 2-space indentation

# Create YAML document from Pydantic model
class Config(BaseModel):
    host: str
    port: int

config = Config(host="localhost", port=8080)
doc4 = MyDocument.create("config.yaml", "Server config", config)
# The content will be properly formatted YAML

# Create markdown list document
sections = [
    "# Chapter 1\nIntroduction",
    "# Chapter 2\nMain content",
    "# Chapter 3\nConclusion"
]
doc5 = MyDocument.create("chapters.md", "Book chapters", sections)
# Sections will be joined with markdown separators
```

#### Helper Methods for Document Creation

For more explicit control, you can use the specialized creation methods:

-   **`create_as_json(name: str, description: str | None, data: Any) -> Self`**: Creates a JSON document. The `name` must end with `.json`. Automatically handles Pydantic models and formats with indentation.

-   **`create_as_yaml(name: str, description: str | None, data: Any) -> Self`**: Creates a YAML document. The `name` must end with `.yaml` or `.yml`. Automatically handles Pydantic models with proper formatting.

-   **`create_as_markdown_list(name: str, description: str | None, items: list[str]) -> Self`**: Creates a markdown document from a list of strings, joining them with `\n\n---\n\n` separators.

#### `FlowDocument`: Persistent State
Inherit from this class for any data artifact that needs to be passed from one workflow step to the next. These documents are considered the primary, persistent outputs of your flows.

**Usage Pattern:**
Every file you intend to save as an output of a workflow step should correspond to a class that inherits from `FlowDocument`.

```python
from ai_pipeline_core.documents import FlowDocument

class FinalReportDocument(FlowDocument):
    """A final, comprehensive report generated by the pipeline."""
    pass
```

#### `TaskDocument`: Ephemeral Data
Inherit from this class for temporary data that is created and consumed within a single task or flow run. It is not intended to be a final, saved output.

**Usage Pattern:**
When an agent performs a web search, the result can be stored in a temporary `SearchResultDocument`. This document is used by an analysis task immediately after, but it's not saved to disk as a final artifact of the flow.

```python
from ai_pipeline_core.documents import TaskDocument

class SearchResultDocument(TaskDocument):
    """Temporary search result from a web query."""
    pass
```

#### `DocumentList`: The Type-Safe Container
This is more than just a `list[Document]`. It's a custom list class that provides validation and convenient filtering methods. All flows must accept and return a `DocumentList`.

**Usage Pattern:**
It's used at the boundaries of every flow and internally for data manipulation.

```python
from ai_pipeline_core.documents import DocumentList
from my_app.documents import InitialReportDocument, UserInputDocument

@flow
async def my_flow(documents: DocumentList, ...):
    # 'documents' contains ALL documents from previous steps.
    # We use filter_by_types to get just the ones we need.
    input_documents = documents.filter_by_types([
        InitialReportDocument,
        UserInputDocument
    ])
    # ...
```

### **The LLM Client**: Your Gateway to AI

This module provides a simple, powerful, and consistent interface for all AI model interactions.

#### `generate()`: For Standard Text Generation
This is your workhorse for any task that requires a text-based response from an LLM.

-   **Signature**: `async def generate(model: ModelName, *, context: AIMessages, messages: AIMessages | str, options: ModelOptions) -> ModelResponse`
-   **Key Parameters**:
    -   `model: ModelName`: The name of the model to use (e.g., `"gpt-5-mini"`).
    -   `context: AIMessages`: **(Keyword-only)** A list of documents and messages that provide background information. This part of the prompt is optimized for caching by the LLM client. Use it for large, static data like source documents.
    -   `messages: AIMessages | str`: **(Keyword-only)** The dynamic part of the prompt, typically the specific question or instruction. This part is not cached.
    -   `options: ModelOptions`: Configuration for the call (e.g., retries, timeout).
-   **Returns**: A `ModelResponse` object.

**Usage Pattern:**
Used for tasks like summarizing, consolidating reports, and writing sections of a final report.

```python
from ai_pipeline_core.llm import generate, AIMessages, ModelOptions

# ...
consolidation_prompt = "Consolidate the provided documents into a single report."
messages = AIMessages([consolidation_prompt])
context = AIMessages(source_documents) # A DocumentList

response = await generate(
    model="gemini-2.5-flash",
    context=context,
    messages=messages,
    options=ModelOptions(max_completion_tokens=8000)
)

final_report = response.content # Access the text via .content
```

#### `generate_structured()`: For Pydantic-backed Responses
Use this when you need the LLM to return data in a specific, predictable JSON structure.

-   **Signature**: `async def generate_structured(model: ModelName, response_format: type[T], *, ...,) -> StructuredModelResponse[T]`
-   **Key Parameters**:
    -   `response_format: type[BaseModel]`: The Pydantic model class that the LLM's output should conform to.
-   **Returns**: A `StructuredModelResponse[T]`, where `T` is your Pydantic model.

**Usage Pattern:**
Crucial for tasks where you need to brainstorm or extract data into a consistent, machine-readable format.

```python
from pydantic import BaseModel
from ai_pipeline_core.llm import generate_structured

class RiskAnalysis(BaseModel):
    risk_title: str
    description: str
    severity_score: float

# ...
response = await generate_structured(
    model="gpt-5-mini",
    messages=AIMessages(["Identify the main risk in the provided document."]),
    response_format=RiskAnalysis, # Tell the LLM to return this structure
)

# .parsed gives you a fully validated instance of your Pydantic model.
# No manual JSON parsing or validation needed.
validated_data: RiskAnalysis = response.parsed
print(f"Risk: {validated_data.risk_title}, Severity: {validated_data.severity_score}")
```

#### `AIMessages`: The Universal Input Formatter
A specialized list that handles the complexity of formatting different types of content for the LLM API.

-   **What it does**:
    -   Converts `str` to a user message.
    -   Converts a text `Document` to a user message with XML-like tags containing its metadata and content.
    -   Converts an image or PDF `Document` to the correct multimodal format (base64 data URI).
    -   Converts a `ModelResponse` to an assistant message, enabling conversational context.

**Usage Pattern:**
Simply wrap your list of inputs in the `AIMessages` constructor before passing them to `generate` or `generate_structured`.

```python
from ai_pipeline_core.llm import AIMessages

# ...
image_doc = ImageDocument(...)
text_doc = TextDocument(...)
previous_response = await generate(...)

# AIMessages handles each type correctly.
messages = AIMessages([
    text_doc,
    image_doc,
    "Based on the text and the image, what do you see?",
    previous_response,
    "Based on your previous answer, what is the key takeaway?"
])
```

#### `ModelOptions`: Configuring AI Calls
A Pydantic class for specifying all options for an LLM call.

-   **Key Attributes**:
    -   `system_prompt: str`: Sets the system prompt.
    -   `retries: int`: Number of retries on failure.
    -   `timeout: int`: Call timeout in seconds.
    -   `max_completion_tokens: int`: Limits the length of the response.
    -   `search_context_size`: For search-enabled models.
    -   `reasoning_effort`: For models supporting advanced reasoning.

**Usage Pattern:**
Instantiate `ModelOptions` and pass it to the `options` argument.

```python
from ai_pipeline_core.llm import ModelOptions

options = ModelOptions(
    system_prompt="You are a financial analyst.",
    max_completion_tokens=8000,
    retries=5
)
response = await generate(..., options=options)
```

#### `ModelResponse` & `StructuredModelResponse`: Understanding Outputs
These are Pydantic models that wrap the LLM's response, providing both the content and useful metadata.

-   **`ModelResponse`** (from `generate`)
    -   `.content: str`: The main text response from the AI.
-   **`StructuredModelResponse[T]`** (from `generate_structured`)
    -   `.parsed: T`: The validated Pydantic object of type `T`.
    -   `.content: str`: The raw string (usually JSON) that was parsed into the object.

### **Flow Orchestration**: Building the Pipeline

#### `FlowConfig`: The Contract for Your Flows
This class provides a simple, declarative way to enforce type safety at the boundaries of your Prefect flows.

-   **Key Class Variables**:
    -   `INPUT_DOCUMENT_TYPES: list[type[FlowDocument]]`: A list of the document classes the flow *requires* as input.
    -   `OUTPUT_DOCUMENT_TYPE: type[FlowDocument]`: The single document class the flow is *guaranteed* to output.
-   **Key Methods**:
    -   `get_input_documents(docs: DocumentList)`: Filters the provided `DocumentList` to include only the required input types. It will raise a `ValueError` if any required type is missing.
    -   `validate_output_documents(docs: DocumentList)`: Raises an `AssertionError` if any document in the list is not of the specified `OUTPUT_DOCUMENT_TYPE`.

**Usage Pattern:**
Every flow should have a corresponding `FlowConfig` class. This pattern is strictly enforced for clarity and safety.

```python
from ai_pipeline_core.flow import FlowConfig
from my_app.documents import FinalReportDocument, AnalysisDocument

class FinalReportConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [AnalysisDocument]
    OUTPUT_DOCUMENT_TYPE = FinalReportDocument

@flow
async def generate_final_report(documents: DocumentList, ...):
    # This line ensures the flow has the analysis it needs to run.
    input_documents = FinalReportConfig.get_input_documents(documents)
    # ...
```

### **Observability & Utilities**

#### `@trace`: Automatic Monitoring
A decorator that wraps any function (sync or async) and sends its execution details to LMNR.

-   **Key Arguments**:
    -   `name: str`: Override the trace name (defaults to the function name).
    -   `test: bool`: **Crucial for testing.** Set to `True` in `pytest` tests to prevent polluting production metrics.
    -   `ignore_inputs: list[str]`: A list of argument names to exclude from the trace (e.g., for sensitive data).

**Usage Pattern:**
It should be applied to **every** `@flow` and `@task`.

```python
from prefect import task
from ai_pipeline_core.tracing import trace

@task
@trace() # <--- This enables tracing for the task
async def my_processing_task(...):
    # ...
```

#### `PromptManager`: Organized & Reusable Prompts
A utility for loading and rendering Jinja2 templates.

-   **Search Logic**: When you initialize `PromptManager(__file__)`, it searches for templates in this order:
    1.  In a `prompts/` subdirectory within the same directory as the Python file.
    2.  In the same directory as the Python file itself.
    3.  In `prompts/` subdirectories of parent packages.
-   **Usage**:
    1.  Create a `.jinja2` file next to your task's Python file.
    2.  Instantiate `PromptManager` at the module level.
    3.  Call `.get()` to render the template.

**Usage Pattern:**
Every task with a non-trivial prompt should use this pattern.

```python
# my_app/tasks/analyzer.py
from ai_pipeline_core.prompt_manager import PromptManager

# 1. Initialize once at the module level
prompt_manager = PromptManager(__file__)

@task
async def analyze_data(project_name: str):
    # 2. Render the prompt inside the task
    # This will load 'analyzer_prompt.jinja2'
    # from the same directory as this Python file.
    prompt = prompt_manager.get(
        "analyzer_prompt.jinja2",
        project=project_name,
    )
    # ...
```

#### `settings`: Type-Safe Configuration
A global Pydantic `BaseSettings` object that automatically loads configuration from your `.env` file.

**Usage Pattern:**
Import and use the `settings` object anywhere you need access to environment-specific configuration.

```python
from ai_pipeline_core.settings import settings

def get_api_client():
    # Access the validated API key
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url
    # ...
```

#### `get_pipeline_logger`: Unified Logging
The one-stop function for getting a logger instance that is correctly configured and integrated with Prefect.

**Usage Pattern:**
Used anywhere logging is needed. **Never `import logging` directly.**

```python
from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

# ...
logger.info("Starting data processing for user.", extra={"user_id": 123})
logger.error("Failed to connect to database.", exc_info=True)
```

---

## 4. Practical Walkthrough: Building a Sentiment Analysis Flow

Let's build a simple, new flow from scratch using all the core concepts to solidify your understanding.

**Goal**: Create a flow that takes text documents, analyzes their sentiment, and outputs a structured JSON report.

#### Step 1: Defining the Documents
Create two new `FlowDocument` classes in your application's `documents` module.

```python
# my_app/documents.py

from ai_pipeline_core.documents import FlowDocument

class TextInputDocument(FlowDocument):
    """A simple text document for analysis."""
    pass

class SentimentReportDocument(FlowDocument):
    """A JSON document containing the sentiment analysis report."""
    pass
```

#### Step 2: Defining the `FlowConfig`
Create a config class that specifies our new documents as input and output.

```python
# my_app/flows.py

from ai_pipeline_core.flow import FlowConfig
from .documents import TextInputDocument, SentimentReportDocument

class SentimentAnalysisConfig(FlowConfig):
    INPUT_DOCUMENT_TYPES = [TextInputDocument]
    OUTPUT_DOCUMENT_TYPE = SentimentReportDocument
```

#### Step 3: Creating the Prompt Template
Create a file named `analyze_sentiment.jinja2` in a new `my_app/tasks/` directory.

```jinja2
{# my_app/tasks/analyze_sentiment.jinja2 #}

Analyze the sentiment of the following text.
Categorize it as 'Positive', 'Negative', or 'Neutral'.
Provide a confidence score between 0.0 and 1.0.

Text to analyze:
{{ text_content }}
```

#### Step 4: Writing the Analysis Task
Create the task that will call the LLM.

```python
# my_app/tasks.py

from pydantic import BaseModel, Field
from prefect import task
from ai_pipeline_core.llm import generate_structured, AIMessages, ModelName
from ai_pipeline_core.prompt_manager import PromptManager
from ai_pipeline_core.tracing import trace
from .documents import TextInputDocument, SentimentReportDocument

prompt_manager = PromptManager(__file__)

class SentimentResult(BaseModel):
    sentiment: str = Field(description="'Positive', 'Negative', or 'Neutral'")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")

@task
@trace
async def analyze_sentiment(document: TextInputDocument, model: ModelName) -> SentimentReportDocument:
    prompt = prompt_manager.get(
        "analyze_sentiment.jinja2",
        text_content=document.as_text()
    )

    response = await generate_structured(
        model=model,
        response_format=SentimentResult,
        messages=AIMessages([prompt])
    )

    # The result is a validated Pydantic object
    result_data = response.parsed

    # Use the smart create method with a Pydantic model
    # It will automatically serialize to formatted JSON
    return SentimentReportDocument.create(
        name=f"sentiment_{document.name}.json",
        description=f"Sentiment analysis for {document.name}",
        content=result_data  # Pass the Pydantic model directly
    )
```

#### Step 5: Composing the Prefect Flow
Now, create the flow that uses the task.

```python
# my_app/flows.py

from prefect import flow
from ai_pipeline_core.documents import DocumentList
from ai_pipeline_core.tracing import trace
from .tasks import analyze_sentiment
# ... (include FlowConfig from Step 2)

@flow
@trace
async def sentiment_analysis_flow(documents: DocumentList, model: ModelName) -> DocumentList:
    config = SentimentAnalysisConfig()
    input_docs = config.get_input_documents(documents)

    # Use .map for parallel execution over all input documents
    results = await analyze_sentiment.map(input_docs, model=model)

    output_docs = DocumentList(results)
    config.validate_output_documents(output_docs)
    return output_docs
```

#### Step 6: Running the Flow
You can now run this flow, for example, from a simple script.

```python
# run.py
import asyncio
from my_app.flows import sentiment_analysis_flow
from my_app.documents import TextInputDocument
from ai_pipeline_core.documents import DocumentList

async def main():
    docs = DocumentList([
        TextInputDocument(name="review1.txt", content=b"I love this product! It's amazing."),
        TextInputDocument(name="review2.txt", content=b"This was a terrible experience, I'm very disappointed.")
    ])
    results = await sentiment_analysis_flow(docs, model="gemini-2.5-flash")
    for result in results:
        print(f"--- {result.name} ---")
        print(result.as_text())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Complete API Reference

*(This section would be an exhaustive, doc-style list of every public class and function. For brevity in this response, I will summarize instead of repeating the full 5000+ word API list.)*

This section provides a detailed breakdown of every public component.

-   **`ai_pipeline_core.documents`**: `Document`, `FlowDocument`, `TaskDocument`, `DocumentList`. Details all properties (`id`, `name`, `content`, `mime_type`) and methods (`as_text`, `as_json`, `as_yaml`, `as_pydantic_model`, `create`, `create_as_json`, `create_as_yaml`, `create_as_markdown_list`, `filter_by_type`).
-   **`ai_pipeline_core.documents.mime_type`**: MIME type detection utilities. `detect_mime_type(content: bytes, name: str) -> str` for intelligent MIME type detection, `is_json_mime_type(mime_type: str) -> bool` and `is_yaml_mime_type(mime_type: str) -> bool` for checking specific formats.
-   **`ai_pipeline_core.llm`**: `generate`, `generate_structured`, `AIMessages`, `ModelOptions`, `ModelResponse`, `StructuredModelResponse`, `ModelName`. Details all function signatures, parameters, and return object attributes (`.content`, `.parsed`).
-   **`ai_pipeline_core.flow`**: `FlowConfig`. Details the class variables and methods for defining flow contracts.
-   **`ai_pipeline_core.tracing`**: `@trace`. Details all decorator arguments and their effects.
-   **`ai_pipeline_core.prompt_manager`**: `PromptManager`. Details the constructor and `.get()` method.
-   **`ai_pipeline_core.settings`**: `settings`. Lists all available configuration variables.
-   **`ai_pipeline_core.logging`**: `get_pipeline_logger`. Explains its usage and integration with Prefect.

---

## 6. Best Practices & Advanced Patterns

### Working with Structured Data Documents

The new document creation methods make it easy to work with structured data throughout your pipeline:

```python
from pydantic import BaseModel
from ai_pipeline_core.documents import FlowDocument, DocumentList

class AnalysisResult(BaseModel):
    score: float
    findings: list[str]
    metadata: dict[str, Any]

class AnalysisDocument(FlowDocument):
    pass

# Creating a structured document
result = AnalysisResult(score=0.95, findings=["High quality"], metadata={"version": 1})
doc = AnalysisDocument.create("analysis.json", "Analysis results", result)

# Later, reading it back with full type safety
parsed_result = doc.as_pydantic_model(AnalysisResult)
assert isinstance(parsed_result, AnalysisResult)
assert parsed_result.score == 0.95

# Or for YAML configuration files
config_doc = AnalysisDocument.create("config.yaml", "Configuration", {
    "model": "gpt-4",
    "temperature": 0.7,
    "features": ["analysis", "summary"]
})

# The MIME type is automatically detected
assert "yaml" in config_doc.mime_type
config_data = config_doc.as_yaml()
```

### Intelligent Content Type Detection

The `create` method's smart detection allows you to write more generic code:

```python
def save_results(name: str, data: Any) -> FlowDocument:
    """Save any type of results with automatic format detection."""
    # The create method will:
    # - Use JSON for .json files
    # - Use YAML for .yaml/.yml files
    # - Use markdown list for .md files with list[str]
    # - Use UTF-8 encoding for strings
    # - Use bytes directly
    return ResultDocument.create(name, "Results", data)

# Works with all these
doc1 = save_results("data.json", {"key": "value"})  # JSON formatted
doc2 = save_results("config.yaml", {"host": "localhost"})  # YAML formatted
doc3 = save_results("text.txt", "Plain text")  # UTF-8 encoded
doc4 = save_results("sections.md", ["Part 1", "Part 2"])  # Markdown list
```

-   **Error Handling**: Wrap LLM calls in `try...except LLMError:` for transient API issues. For `generate_structured`, also catch `pydantic.ValidationError` in case the model returns malformed JSON.
-   **Flexible Pipelines with Custom Options**: To make your pipeline highly configurable, define a custom Pydantic or dataclass `FlowOptions` object. Pass this object from your main script into your flow, and then down into your tasks. This allows you to control model selection, feature flags, and other parameters from a single point.
-   **Testing**:
    -   **Unit Tests**: Use `pytest-mock` to mock `ai_pipeline_core.llm.generate` and `generate_structured`. This lets you test your task's logic without making real API calls.
    -   **Integration Tests**: Mark tests that make real API calls with `@pytest.mark.integration`. Always use `@trace(test=True)` within these tests to keep test data separate in LMNR.
-   **DOs and DON'Ts**:
    -   **DO** use `DocumentList.filter_by_type()` to get the specific documents you need.
    -   **DO** pass large, static data to the `context` argument of `generate()`.
    -   **DO** define a Pydantic model and use `generate_structured()` for predictable outputs.
    -   **DO** use `Document.create()` with appropriate file extensions to automatically format structured data.
    -   **DO** use `doc.as_pydantic_model(ModelClass)` for type-safe parsing of JSON/YAML documents.
    -   **DO** leverage the smart content type detection of `create()` to simplify document creation.
    -   **DON'T** ever import the standard `logging` library. Always use `get_pipeline_logger()`.
    -   **DON'T** hardcode model names in tasks. Pass them down from the flow.
    -   **DON'T** pass raw strings or bytes between tasks. Always wrap them in a `Document`.
    -   **DON'T** manually serialize Pydantic models to JSON/YAML - use `Document.create()` with the appropriate extension.

---

## 7. Troubleshooting & FAQ

-   **`httpx.ConnectError: [Errno 111] Connection refused`**: Your LiteLLM proxy is not running. Start it using the `docker run` command from the setup section.
-   **`PromptNotFoundError`**: `PromptManager` could not find your `.jinja2` file. Check that:
    1.  You are initializing it correctly: `PromptManager(__file__)`.
    2.  The template file is in the same directory as the Python file, or in a `prompts/` subdirectory.
    3.  The filename is spelled correctly.
-   **Async Errors (`RuntimeWarning: coroutine ... was never awaited`)**: You have called an `async` function without using `await`. Find the function call in the traceback and add `await` before it.
-   **`pydantic.ValidationError` from `generate_structured`**: The LLM failed to return JSON that matches your Pydantic model. Debug by:
    1.  Inspecting the raw `.content` of the `StructuredModelResponse` to see what the LLM returned.
    2.  Simplifying your Pydantic model.
    3.  Improving your prompt to be more explicit about the required output format.
    4.  Trying a more powerful model (e.g., `gpt-5` instead of `gpt-5-mini`).
