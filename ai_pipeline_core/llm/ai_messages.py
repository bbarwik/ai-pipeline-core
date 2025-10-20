"""AI message handling for LLM interactions.

@public

Provides AIMessages container for managing conversations with mixed content types
including text, documents, and model responses.
"""

import base64
import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Iterable, SupportsIndex, Union

import tiktoken
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)
from prefect.logging import get_logger

from ai_pipeline_core.documents import Document

from .model_response import ModelResponse

AIMessageType = str | Document | ModelResponse
"""Type for messages in AIMessages container.

@public

Represents the allowed types for conversation messages:
- str: Plain text messages
- Document: Structured document content
- ModelResponse: LLM generation responses
"""


class AIMessages(list[AIMessageType]):
    """Container for AI conversation messages supporting mixed types.

    @public

    This class extends list to manage conversation messages between user
    and AI, supporting text, Document objects, and ModelResponse instances.
    Messages are converted to OpenAI-compatible format for LLM interactions.

    Conversion Rules:
        - str: Becomes {"role": "user", "content": text}
        - Document: Becomes {"role": "user", "content": document_content}
          (automatically handles text, images, PDFs based on MIME type)
        - ModelResponse: Becomes {"role": "assistant", "content": response.content}

    Note: Document conversion is automatic. Text content becomes user text messages.

    VISION/PDF MODEL COMPATIBILITY WARNING:
    Images require vision-capable models (e.g., gpt-4o, gemini-pro-vision, claude-3-haiku).
    Non-vision models will raise ValueError when encountering image documents.
    PDFs require models with document processing support - check your model's capabilities
    before including PDF documents in messages. Unsupported models may fall back to
    text extraction or raise errors depending on provider configuration.
    LiteLLM proxy handles the specific encoding requirements for each provider.

    IMPORTANT: Although AIMessages can contain Document entries, the LLM client functions
    expect `messages` to be `AIMessages` or `str`. If you start from a Document or a list
    of Documents, build AIMessages first (e.g., `AIMessages([doc])` or `AIMessages(docs)`).

    CAUTION: AIMessages is a list subclass. Always use list construction (e.g.,
    `AIMessages(["text"])`) or empty constructor with append (e.g.,
    `AIMessages(); messages.append("text")`). Never pass raw strings directly to the
    constructor (`AIMessages("text")`) as this will raise a TypeError to prevent
    accidental character iteration.

    Example:
        >>> from ai_pipeline_core import llm
        >>> messages = AIMessages()
        >>> messages.append("What is the capital of France?")
        >>> response = await llm.generate("gpt-5", messages=messages)
        >>> messages.append(response)  # Add the actual response
    """

    def __init__(self, iterable: Iterable[AIMessageType] | None = None, *, frozen: bool = False):
        """Initialize AIMessages with optional iterable.

        Args:
            iterable: Optional iterable of messages (list, tuple, etc.).
                     Must not be a string.
            frozen: If True, list is immutable from creation.

        Raises:
            TypeError: If a string is passed directly to the constructor.
        """
        if isinstance(iterable, str):
            raise TypeError(
                "AIMessages cannot be constructed from a string directly. "
                "Use AIMessages(['text']) for a single message or "
                "AIMessages() and then append('text')."
            )
        self._frozen = False  # Initialize as unfrozen to allow initial population
        if iterable is None:
            super().__init__()
        else:
            super().__init__(iterable)
        self._frozen = frozen  # Set frozen state after initial population

    def freeze(self) -> None:
        """Permanently freeze the list, preventing modifications.

        Once frozen, the list cannot be unfrozen.
        """
        self._frozen = True

    def copy(self) -> "AIMessages":
        """Create an unfrozen deep copy of the list.

        Returns:
            New unfrozen AIMessages with deep-copied messages.
        """
        copied_messages = deepcopy(list(self))
        return AIMessages(copied_messages, frozen=False)

    def _check_frozen(self) -> None:
        """Check if list is frozen and raise if it is.

        Raises:
            RuntimeError: If the list is frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot modify frozen AIMessages")

    def append(self, message: AIMessageType) -> None:
        """Add a message to the end of the list."""
        self._check_frozen()
        super().append(message)

    def extend(self, messages: Iterable[AIMessageType]) -> None:
        """Add multiple messages to the list."""
        self._check_frozen()
        super().extend(messages)

    def insert(self, index: SupportsIndex, message: AIMessageType) -> None:
        """Insert a message at the specified position."""
        self._check_frozen()
        super().insert(index, message)

    def __setitem__(
        self,
        index: Union[SupportsIndex, slice],
        value: Union[AIMessageType, Iterable[AIMessageType]],
    ) -> None:
        """Set item or slice."""
        self._check_frozen()
        super().__setitem__(index, value)  # type: ignore[arg-type]

    def __iadd__(self, other: Iterable[AIMessageType]) -> "AIMessages":
        """In-place addition (+=).

        Returns:
            This AIMessages instance after modification.
        """
        self._check_frozen()
        return super().__iadd__(other)

    def __delitem__(self, index: Union[SupportsIndex, slice]) -> None:
        """Delete item or slice from list."""
        self._check_frozen()
        super().__delitem__(index)

    def pop(self, index: SupportsIndex = -1) -> AIMessageType:
        """Remove and return item at index.

        Returns:
            AIMessageType removed from the list.
        """
        self._check_frozen()
        return super().pop(index)

    def remove(self, message: AIMessageType) -> None:
        """Remove first occurrence of message."""
        self._check_frozen()
        super().remove(message)

    def clear(self) -> None:
        """Remove all items from list."""
        self._check_frozen()
        super().clear()

    def reverse(self) -> None:
        """Reverse list in place."""
        self._check_frozen()
        super().reverse()

    def sort(
        self, *, key: Callable[[AIMessageType], Any] | None = None, reverse: bool = False
    ) -> None:
        """Sort list in place."""
        self._check_frozen()
        if key is None:
            super().sort(reverse=reverse)  # type: ignore[call-arg]
        else:
            super().sort(key=key, reverse=reverse)

    def get_last_message(self) -> AIMessageType:
        """Get the last message in the conversation.

        Returns:
            The last message in the conversation, which can be a string,
            Document, or ModelResponse.
        """
        return self[-1]

    def get_last_message_as_str(self) -> str:
        """Get the last message as a string, raising if not a string.

        Returns:
            The last message as a string.

        Raises:
            ValueError: If the last message is not a string.

        Safer Pattern:
            Instead of catching ValueError, check type first:
            >>> messages = AIMessages([user_msg, response, followup])
            >>> last = messages.get_last_message()
            >>> if isinstance(last, str):
            ...     text = last
            >>> elif isinstance(last, ModelResponse):
            ...     text = last.content
            >>> elif isinstance(last, Document):
            ...     text = last.text if last.is_text else "<binary>"
        """
        last_message = self.get_last_message()
        if isinstance(last_message, str):
            return last_message
        raise ValueError(f"Wrong message type: {type(last_message)}")

    def to_prompt(self) -> list[ChatCompletionMessageParam]:
        """Convert AIMessages to OpenAI-compatible format.

        Transforms the message list into the format expected by OpenAI API.
        Each message type is converted according to its role and content.

        Returns:
            List of ChatCompletionMessageParam dicts (from openai.types.chat)
            with 'role' and 'content' keys. Ready to be passed to generate()
            or OpenAI API directly.

        Raises:
            ValueError: If message type is not supported.

        Example:
            >>> messages = AIMessages(["Hello", response, "Follow up"])
            >>> prompt = messages.to_prompt()
            >>> # Result: [
            >>> #   {"role": "user", "content": "Hello"},
            >>> #   {"role": "assistant", "content": "..."},
            >>> #   {"role": "user", "content": "Follow up"}
            >>> # ]
        """
        messages: list[ChatCompletionMessageParam] = []

        for message in self:
            if isinstance(message, str):
                messages.append({"role": "user", "content": message})
            elif isinstance(message, Document):
                messages.append({"role": "user", "content": AIMessages.document_to_prompt(message)})
            elif isinstance(message, ModelResponse):  # type: ignore
                messages.append({"role": "assistant", "content": message.content})
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

        return messages

    def to_tracing_log(self) -> list[str]:
        """Convert AIMessages to a list of strings for tracing.

        Returns:
            List of string representations for tracing logs.
        """
        messages: list[str] = []
        for message in self:
            if isinstance(message, Document):
                serialized_document = message.serialize_model()
                filtered_doc = {k: v for k, v in serialized_document.items() if k != "content"}
                messages.append(json.dumps(filtered_doc, indent=2))
            elif isinstance(message, ModelResponse):
                messages.append(message.content)
            else:
                assert isinstance(message, str)
                messages.append(message)
        return messages

    def get_prompt_cache_key(self, system_prompt: str | None = None) -> str:
        """Generate cache key for message set.

        Args:
            system_prompt: Optional system prompt to include in cache key.

        Returns:
            SHA256 hash as hex string for cache key.
        """
        if not system_prompt:
            system_prompt = ""
        return hashlib.sha256((system_prompt + json.dumps(self.to_prompt())).encode()).hexdigest()

    @property
    def approximate_tokens_count(self) -> int:
        """Approximate tokens count for the messages.

        @public

        Uses tiktoken with gpt-4 encoding to estimate total token count
        across all messages in the conversation.

        Returns:
            Approximate tokens count for all messages.

        Raises:
            ValueError: If message contains unsupported type.

        Example:
            >>> messages = AIMessages(["Hello", "World"])
            >>> messages.approximate_tokens_count  # ~2-3 tokens
        """
        count = 0
        for message in self:
            if isinstance(message, str):
                count += len(tiktoken.encoding_for_model("gpt-4").encode(message))
            elif isinstance(message, Document):
                count += message.approximate_tokens_count
            elif isinstance(message, ModelResponse):  # type: ignore
                count += len(tiktoken.encoding_for_model("gpt-4").encode(message.content))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return count

    @staticmethod
    def document_to_prompt(document: Document) -> list[ChatCompletionContentPartParam]:
        """Convert a document to prompt format for LLM consumption.

        Args:
            document: The document to convert.

        Returns:
            List of chat completion content parts for the prompt.
        """
        prompt: list[ChatCompletionContentPartParam] = []

        # Build the text header
        description = (
            f"<description>{document.description}</description>\n" if document.description else ""
        )
        header_text = (
            f"<document>\n<id>{document.id}</id>\n<name>{document.name}</name>\n{description}"
        )

        # Handle text documents
        if document.is_text:
            text_content = document.content.decode("utf-8")
            content_text = f"{header_text}<content>\n{text_content}\n</content>\n</document>\n"
            prompt.append({"type": "text", "text": content_text})
            return prompt

        # Handle non-text documents
        if not document.is_image and not document.is_pdf:
            get_logger(__name__).error(
                f"Document is not a text, image or PDF: {document.name} - {document.mime_type}"
            )
            return []

        # Add header for binary content
        prompt.append({
            "type": "text",
            "text": f"{header_text}<content>\n",
        })

        # Encode binary content
        base64_content = base64.b64encode(document.content).decode("utf-8")
        data_uri = f"data:{document.mime_type};base64,{base64_content}"

        # Add appropriate content type
        if document.is_pdf:
            prompt.append({
                "type": "file",
                "file": {"file_data": data_uri},
            })
        else:  # is_image
            prompt.append({
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": "high"},
            })

        # Close the document tag
        prompt.append({"type": "text", "text": "</content>\n</document>\n"})

        return prompt
