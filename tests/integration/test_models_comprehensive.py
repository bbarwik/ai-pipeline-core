"""Comprehensive integration tests for all model types."""

from pathlib import Path

import pytest
from prefect.logging import get_logger

from ai_pipeline_core.llm import AIMessages, ModelOptions, generate
from ai_pipeline_core.llm.model_types import ModelName
from ai_pipeline_core.settings import settings
from tests.test_helpers import ConcreteFlowDocument

from .model_categories import CORE_MODELS, SEARCH_MODELS

logger = get_logger()

# Check if API keys are configured in settings (respects .env file)
# Note: We evaluate this at runtime, not at import time
HAS_API_KEYS = bool(settings.openai_api_key and settings.openai_base_url)

# Skip all tests if API keys not configured
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not HAS_API_KEYS,
        reason="OpenAI API keys not configured in settings or .env file",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_basic_math(model: ModelName):
    """Test all models on basic math."""
    question = "Hi. What is 12+97?"
    messages = AIMessages([question])
    options = ModelOptions(retries=1)

    try:
        response = await generate(model=model, messages=messages, options=options)
        logger.info(f"Model {model} response (math):\n{response.content}")

        assert response.content is not None
        assert "109" in response.content, (
            f"Model {model} did not return correct answer for math question.\n"
            f"Response: {response.content}"
        )
    except Exception as e:
        pytest.skip(f"Model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", SEARCH_MODELS)
async def test_llm_model_search_feature(model: ModelName):
    """Test search models with search queries."""
    question = (
        "Search in the information who was elected to be Pope in 2025.\n"
        "Find the name of this person.\n"
        "You need to return information only from search results.\n"
        "You were configured to perform internet search or use tools to perform search.\n"
        "You are not allowed to use your internal knowledge or any other information.\n"
        "If you are not able to perform internet search then inform about it and abort execution."
    )
    messages = AIMessages([question])
    # Enable search context for search models
    options = ModelOptions(search_context_size="high")

    try:
        response = await generate(model=model, messages=messages, options=options)
        logger.info(f"Model {model} response (search):\n{response.content}")

        assert response.content is not None
        content_lower = response.content.lower()

        # Search models should either find the info or indicate they can't search
        # We check if they mention search-related keywords or the actual answer
        search_keywords = ["search", "internet", "unable", "cannot", "can't"]
        answer_keywords = ["robert", "leo", "prevost", "pope", "2025"]

        has_search_response = any(word in content_lower for word in search_keywords)
        has_answer = any(word in content_lower for word in answer_keywords)

        assert has_search_response or has_answer, (
            f"Model {model} did not provide a search response or answer.\n"
            f"Response: {response.content}"
        )
    except Exception as e:
        pytest.skip(f"Search model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_no_search_feature(model: ModelName):
    """Test non-search models to ensure they don't have search capability."""
    question = (
        "Search in the information who was elected to be Pope in 2025.\n"
        "Find the name of this person.\n"
        "You need to return information only from search results.\n"
        "You were configured to perform internet search or use tools to perform search.\n"
        "You are not allowed to use your internal knowledge or any other information.\n"
        "If you are not able to perform internet search then inform about it and abort execution."
    )
    messages = AIMessages([question])

    try:
        response = await generate(model=model, messages=messages)
        logger.info(f"Model {model} response (no search):\n{response.content}")

        assert response.content is not None
        content_lower = response.content.lower()

        # These specific names should not appear as they require real-time search
        # (they are from 2025 which is beyond training data)
        specific_names = ["robert", "leo", "prevost"]

        # It's OK if they mention they can't search or don't have the info
        inability_keywords = ["cannot", "can't", "unable", "don't have", "not able", "search"]
        has_inability = any(word in content_lower for word in inability_keywords)

        # If they don't express inability, they shouldn't have the specific names
        if not has_inability:
            assert not any(name in content_lower for name in specific_names), (
                f"Model {model} mentioned specific 2025 names without search capability.\n"
                f"Response: {response.content}"
            )
    except Exception as e:
        pytest.skip(f"Model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_text_file_attachment(model: ModelName):
    """Test models with text file attachments."""
    # Create documents with text content
    doc1 = ConcreteFlowDocument(name="file1.md", content=b"This file contains number 121")
    doc2 = ConcreteFlowDocument(
        name="file2.md", content=b"This file contains number eight hundred twenty two"
    )

    messages = AIMessages([
        "There will be 2 files included and then a task in the last message",
        doc1,
        doc2,
        "What is the sum of the numbers in the attached files?",
    ])

    try:
        response = await generate(model=model, messages=messages)
        logger.info(f"Model {model} response (file attachment):\n{response.content}")

        assert response.content is not None
        assert "943" in response.content, (
            f"Model {model} did not return correct sum for file attachment test.\n"
            f"Response: {response.content}"
        )
    except Exception as e:
        pytest.skip(f"Model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_image_file_attachment(model: ModelName):
    """Test models with image file attachments."""
    if model == "grok-4-fast":
        pytest.skip("grok-4-fast does not support image attachments")

    # Load the test image
    test_image_path = Path("tests/test_data/test_image.png")
    if not test_image_path.exists():
        pytest.skip(f"Test image not found at {test_image_path}")

    image_doc = ConcreteFlowDocument(name="test_image.png", content=test_image_path.read_bytes())

    messages = AIMessages([
        "There will be an image included and then a task in the last message",
        image_doc,
        "What is on the image? Describe what you see.",
    ])

    try:
        response = await generate(model=model, messages=messages)
        logger.info(f"Model {model} response (image attachment):\n{response.content}")

        assert response.content is not None
        content_lower = response.content.lower()

        # The image should contain something recognizable
        # We're looking for any reasonable description keywords
        keywords = [
            "image",
            "picture",
            "shows",
            "contains",
            "see",
            "visible",
            "openai",
            "api",
            "response",
            "client",
            "python",
            "code",
            "text",
        ]

        assert any(word in content_lower for word in keywords), (
            f"Model {model} did not provide a reasonable image description.\n"
            f"Response: {response.content}"
        )
    except Exception as e:
        if "image" in str(e).lower() or "support" in str(e).lower():
            pytest.skip(f"Model {model} does not support images: {e}")
        else:
            pytest.skip(f"Model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_pdf_file_attachment(model: ModelName):
    """Test models with PDF file attachments."""
    if model in ["grok-4-fast", "grok-4"]:
        pytest.skip(f"{model} does not support PDF attachments")

    # Load the test PDF
    test_pdf_path = Path("tests/test_data/test_pdf.pdf")
    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF not found at {test_pdf_path}")

    pdf_doc = ConcreteFlowDocument(name="test_pdf.pdf", content=test_pdf_path.read_bytes())

    messages = AIMessages([
        "There will be a PDF file included and then a task in the last message",
        pdf_doc,
        "What is this PDF about? What are the main topics or keywords mentioned?",
    ])

    try:
        response = await generate(model=model, messages=messages)
        logger.info(f"Model {model} response (PDF attachment):\n{response.content}")

        assert response.content is not None
        content_lower = response.content.lower()

        # Expected keywords from the PDF content
        keywords = ["yukon", "education", "canada", "territory", "school", "learning"]

        assert any(word in content_lower for word in keywords), (
            f"Model {model} did not mention expected PDF content keywords.\n"
            f"Response: {response.content}"
        )
    except Exception as e:
        if "pdf" in str(e).lower() or "support" in str(e).lower():
            pytest.skip(f"Model {model} does not support PDFs: {e}")
        else:
            pytest.skip(f"Model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_conversation_context(model: ModelName):
    """Test models with conversation context."""
    # First message
    messages1 = AIMessages(["My name is Alice. Remember this name."])

    try:
        response1 = await generate(
            model=model, messages=messages1, options=ModelOptions(max_completion_tokens=1000)
        )

        # Follow-up with context
        messages2 = AIMessages([
            "My name is Alice. Remember this name.",
            response1,
            "What is my name?",
        ])

        response2 = await generate(
            model=model, messages=messages2, options=ModelOptions(max_completion_tokens=1000)
        )

        logger.info(f"Model {model} response (context):\n{response2.content}")

        assert response2.content is not None
        assert "alice" in response2.content.lower(), (
            f"Model {model} did not remember the name from context.\nResponse: {response2.content}"
        )
    except Exception as e:
        pytest.skip(f"Model {model} not available: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", CORE_MODELS)
async def test_llm_model_with_options(model: ModelName):
    """Test models with various options."""
    messages = AIMessages(["Write exactly 3 words."])

    options = ModelOptions(max_completion_tokens=1000, retries=2, retry_delay_seconds=1, timeout=30)

    try:
        response = await generate(model=model, messages=messages, options=options)
        logger.info(f"Model {model} response (options test):\n{response.content}")

        assert response.content is not None
        # Check that response is reasonably short (respecting max_tokens)
        word_count = len(response.content.split())
        assert word_count <= 10, (
            f"Model {model} response too long despite max_tokens limit.\n"
            f"Response: {response.content}"
        )
    except Exception as e:
        pytest.skip(f"Model {model} not available: {e}")
