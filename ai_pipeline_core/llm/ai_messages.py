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
    
    Example:
        >>> messages = AIMessages()
        >>> messages.append("What is the capital of France?")
        >>> messages.append(ModelResponse(content="The capital of France is Paris."))
        >>> prompt = messages.to_prompt()
    """

    def get_last_message(self) -> AIMessageType:
        """Get the last message in the conversation.
        
        @public

        Returns:
            The last message in the conversation, which can be a string,
            Document, or ModelResponse.
        """
        return self[-1]

    def get_last_message_as_str(self) -> str:
        """Get the last message as a string, raising if not a string.
        
        @public

        Returns:
            The last message as a string.

        Raises:
            ValueError: If the last message is not a string.
        """
        last_message = self.get_last_message()
        if isinstance(last_message, str):
            return last_message
        raise ValueError(f"Wrong message type: {type(last_message)}")

    def to_prompt(self) -> list[ChatCompletionMessageParam]:
        """Convert AIMessages to OpenAI-compatible format.

        Returns:
            List of ChatCompletionMessageParam for OpenAI API.

        Raises:
            ValueError: If message type is not supported.
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
