"""@public Document-aware prompt builder with LLM calling, caching, and document extraction."""

import re
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from ai_pipeline_core.documents import Document, DocumentList
from ai_pipeline_core.llm import (
    AIMessages,
    ModelName,
    ModelOptions,
    ModelResponse,
    StructuredModelResponse,
)
from ai_pipeline_core.llm.client import generate, generate_structured
from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.prompt_manager import PromptManager

from .global_cache import GlobalCacheLock

_prompt_manager = PromptManager(__file__)
logger = get_pipeline_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class EnvironmentVariable(BaseModel):
    """@public Named variable injected as XML-wrapped content in LLM messages."""

    name: str
    value: str


class PromptBuilder(BaseModel):
    """@public Document-aware prompt builder for LLM interactions.

    Manages three document hierarchies (core, source, new core), environment variables,
    and provides call/call_structured/generate_document methods with automatic prompt
    caching coordination.

    Context (cached) = [system_prompt, *core_documents, *new_documents, documents_listing]
    Messages (per-call) = [*new_core_documents, *environment_variables, user_prompt]
    """

    model_config = {"arbitrary_types_allowed": True}

    core_documents: DocumentList = Field(default_factory=DocumentList)
    new_documents: DocumentList = Field(default_factory=DocumentList)
    environment: list[EnvironmentVariable] = Field(default_factory=list)
    new_core_documents: DocumentList = Field(default_factory=DocumentList)
    default_options: ModelOptions = Field(
        default=ModelOptions(
            reasoning_effort="high",
            verbosity="high",
            max_completion_tokens=32 * 1024,
        )
    )
    mode: Literal["test", "quick", "full"] = Field(default="full")

    def _get_system_prompt(self) -> str:
        return _prompt_manager.get("system_prompt.jinja2")

    def _get_documents_prompt(self) -> str:
        return _prompt_manager.get(
            "documents_prompt.jinja2",
            core_documents=self.core_documents,
            new_documents=self.new_documents,
        )

    def _get_new_core_documents_prompt(self) -> str:
        return _prompt_manager.get(
            "new_core_documents_prompt.jinja2", new_core_documents=self.new_core_documents
        )

    def _get_context(self) -> AIMessages:
        return AIMessages([
            self._get_system_prompt(),
            *self.core_documents,
            *self.new_documents,
            self._get_documents_prompt(),
        ])

    def _get_messages(self, prompt: str | AIMessages) -> AIMessages:
        messages = AIMessages()
        if self.new_core_documents:
            messages.append(self._get_new_core_documents_prompt())
            for document in self.new_core_documents:
                messages.append(document)
        for variable in self.environment:
            messages.append(
                f"# {variable.name}\n\n<{variable.name}>\n{variable.value}\n</{variable.name}>"
            )
        if isinstance(prompt, AIMessages):
            messages.extend(prompt)
        else:
            messages.append(prompt)
        return messages

    @property
    def approximate_tokens_count(self) -> int:
        """@public Approximate total token count for context + messages."""
        return (
            self._get_context().approximate_tokens_count
            + self._get_messages("").approximate_tokens_count
        )

    def add_variable(self, name: str, value: str | Document | None = None) -> None:
        """@public Add an environment variable injected as XML in messages.

        Variables are NOT available in Jinja2 templates. Instead, tell the LLM
        about the variable in the prompt text.
        """
        assert name != "document", "document is a reserved variable name"
        assert name not in [e.name for e in self.environment], f"Variable {name} already exists"
        if not value:
            return
        if isinstance(value, Document):
            value = value.text
        self.environment.append(EnvironmentVariable(name=name, value=value))

    def remove_variable(self, name: str) -> None:
        """@public Remove an environment variable by name."""
        assert name in [e.name for e in self.environment], f"Variable {name} not found"
        self.environment = [e for e in self.environment if e.name != name]

    def add_new_core_document(self, document: Document) -> None:
        """@public Add a session-created document to new_core_documents."""
        self.new_core_documents.append(document)

    def _get_options(
        self, model: ModelName, options: ModelOptions | None = None
    ) -> tuple[ModelOptions, bool]:
        if not options:
            options = self.default_options

        options = options.model_copy(deep=True)
        options.system_prompt = self._get_system_prompt()

        cache_lock = True
        if "qwen3" in model:
            options.usage_tracking = False
            options.verbosity = None
            options.service_tier = None
            options.cache_ttl = None
            cache_lock = False
        if "grok-4-fast" in model:
            options.max_completion_tokens = 30000

        if self.mode == "test":
            options.reasoning_effort = "low"

        if model.endswith("o3"):
            options.reasoning_effort = "medium"
            options.verbosity = None

        if model.startswith("gpt-5"):
            options.service_tier = "flex"

        return options, cache_lock

    async def call(
        self, model: ModelName, prompt: str | AIMessages, options: ModelOptions | None = None
    ) -> ModelResponse:
        """@public Generate text response with document context and caching."""
        options, use_cache_lock = self._get_options(model, options)
        context = self._get_context()
        messages = self._get_messages(prompt)
        async with GlobalCacheLock(model, context, use_cache_lock) as lock:
            options.extra_body = {
                "metadata": {
                    "wait_time": f"{lock.wait_time:.2f}s",
                    "use_cache": str(lock.use_cache),
                    "approximate_tokens_count": context.approximate_tokens_count,
                }
            }
            return await generate(
                model=model,
                context=context,
                messages=messages,
                options=options,
            )

    async def call_structured(
        self,
        model: ModelName,
        response_format: type[T],
        prompt: str | AIMessages,
        options: ModelOptions | None = None,
    ) -> StructuredModelResponse[T]:
        """@public Generate validated Pydantic model output with document context."""
        options, use_cache_lock = self._get_options(model, options)
        context = self._get_context()
        messages = self._get_messages(prompt)
        async with GlobalCacheLock(model, context, use_cache_lock) as lock:
            options.extra_body = {
                "metadata": {
                    "wait_time": f"{lock.wait_time:.2f}s",
                    "use_cache": str(lock.use_cache),
                }
            }
            return await generate_structured(
                model=model,
                response_format=response_format,
                context=context,
                messages=messages,
                options=options,
            )

    async def generate_document(
        self,
        model: ModelName,
        prompt: str | AIMessages,
        title: str | None = None,
        options: ModelOptions | None = None,
    ) -> str:
        """@public Generate document content extracted from <document> tags."""
        document = await self._call_and_extract_document(model, prompt, options)
        if title:
            document = self._add_title_to_document(document, title)
        return document

    async def _call_and_extract_document(
        self, model: ModelName, prompt: str | AIMessages, options: ModelOptions | None = None
    ) -> str:
        options, _ = self._get_options(model, options)
        if "gpt-5" not in model and "grok-4" not in model and "openrouter/" not in model:
            options.stop = "</document>"

        response = await self.call(model, prompt, options)
        documents: list[str] = re.findall(
            r"<document>(.*?)(?:</document>|$)", response.content, re.DOTALL
        )
        documents = [doc.strip() for doc in documents if len(doc) >= 20]

        if not documents:
            return response.content

        if len(documents) > 1:
            if len(documents[0]) > 20:
                logger.warning(f"Found {len(documents)} documents, returning first one")
            else:
                logger.warning(f"Found {len(documents)} documents, returning largest one")
                documents.sort(key=len, reverse=True)

        return documents[0]

    def _add_title_to_document(self, document: str, title: str) -> str:
        if document.startswith("# "):
            document = f"# {title}\n{document.split('\n', 1)[1]}"
        else:
            document = f"# {title}\n\n{document}"
        return document
