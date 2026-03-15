"""Bug-proving tests for codec boundary issues.

Bugs B2 and B3 from CORE-BUGS.md:
- B2: model_validate() error escapes as raw ValidationError instead of CodecError
- B3: CLI-invoked spans are unreplayable — <locals> in class path
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_pipeline_core._codec import CodecError, UniversalCodec


# ── B2 fixtures (module-scope so codec can encode them) ───────────────────


class _StrictModel(BaseModel):
    count: int = Field(ge=0)


class _FailingLoadModel(BaseModel):
    value: str

    @classmethod
    def __codec_load__(cls, data: dict[str, Any]) -> _FailingLoadModel:
        raise RuntimeError("Load failed intentionally")

    def __codec_state__(self) -> dict[str, Any]:
        return {"value": self.value}


# ── B2: model_validate error escapes as raw ValidationError ─────────────────


async def test_codec_decode_async_wraps_validation_error_as_codec_error() -> None:
    """When a Pydantic model's data fails validation during decode_async, the error
    must be wrapped as CodecError, not escape as raw ValidationError.

    Callers expect CodecError from the decode path (error boundary).
    """
    codec = UniversalCodec()
    encoded = codec.encode(_StrictModel(count=5))
    payload = encoded.value
    # Corrupt: set count to invalid value that fails Pydantic validation
    payload["data"]["count"] = "not_an_integer"

    with pytest.raises(CodecError):
        await codec.decode_async(payload, db=None)


async def test_codec_decode_async_wraps_codec_load_error_as_codec_error() -> None:
    """When __codec_load__ raises, the error must be wrapped as CodecError."""
    codec = UniversalCodec()
    encoded = codec.encode(_FailingLoadModel(value="test"))
    payload = encoded.value

    with pytest.raises(CodecError):
        await codec.decode_async(payload, db=None)


# ── B3: <locals> in class path makes spans unreplayable ─────────────────────


def test_codec_encode_rejects_function_local_class() -> None:
    """A class defined inside a function has '<locals>' in __qualname__,
    making it unresolvable during replay. The codec should reject it at encode time.
    """

    def create_local_model() -> type[BaseModel]:
        class LocalModel(BaseModel):
            x: int = 1

        return LocalModel

    LocalModel = create_local_model()
    assert "<locals>" in LocalModel.__qualname__

    codec = UniversalCodec()
    with pytest.raises(CodecError, match="<locals>"):
        codec.encode(LocalModel(x=42))
