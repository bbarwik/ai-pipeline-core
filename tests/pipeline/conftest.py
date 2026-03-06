"""Shared fixtures for pipeline tests."""

import pytest

from ai_pipeline_core.documents._context import _suppress_document_registration


@pytest.fixture(autouse=True)
def _suppress_registration():
    with _suppress_document_registration():
        yield
