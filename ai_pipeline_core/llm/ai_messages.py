"""AI message handling for LLM interactions.

@public

Provides AIMessages container for managing conversations with mixed content types
including text, documents, and model responses.
"""

import base64
import hashlib
import json

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
    Images are sent to vision-capable models (non-vision models will raise ValueError).
    PDFs are attached when supported by the model, otherwise a text extraction
    fallback is used. LiteLLM proxy handles the specific encoding requirements
    for each provider.

    IMPORTANT: Although AIMessages can contain Document entries, the LLM client functions
    expect `messages` to be `AIMessages` or `str`. If you start from a Document or a list
    of Documents, build AIMessages first (e.g., `AIMessages([doc])` or `AIMessages(docs)`).

    Example:
        >>> from ai_pipeline_core import llm
        >>> messages = AIMessages()
        >>> messages.append("What is the capital of France?")
        >>> response = await llm.generate("gpt-5", messages=messages)
        >>> messages.append(response)  # Add the actual response
    """

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
                del serialized_document["content"]
                messages.append(json.dumps(serialized_document, indent=2))
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
