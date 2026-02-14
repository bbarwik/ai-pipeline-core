"""Concurrency and rate limiting for pipeline deployments.

Wraps Prefect's global concurrency limits so app code never imports from Prefect.
Limits are declared on PipelineDeployment subclasses and enforced via pipeline_concurrency().

Supports three limit kinds:
- CONCURRENT: Lease-based slots held during operation, released on exit.
- PER_MINUTE: Token bucket with limit/60 decay per second (allows bursting).
- PER_HOUR: Token bucket with limit/3600 decay per second (allows bursting).

When Prefect is unavailable, CONCURRENT limits fall back to per-process asyncio.Semaphore.
PER_MINUTE/PER_HOUR have no local fallback and proceed unthrottled.
"""

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from prefect.client.orchestration import get_client
from prefect.concurrency.asyncio import (
    AcquireConcurrencySlotTimeoutError,
    ConcurrencySlotAcquisitionError,
    concurrency,
    rate_limit,
)

from ai_pipeline_core.logging import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_VALID_LIMIT_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class LimitKind(StrEnum):
    """Kind of concurrency/rate limit.

    CONCURRENT: Slots held for duration of operation (lease-based).
        limit=500 means at most 500 simultaneous operations across all runs.

    PER_MINUTE: Token bucket with limit/60 decay per second.
        Allows bursting up to `limit` immediately, then refills gradually.
        NOT a sliding window.

    PER_HOUR: Token bucket with limit/3600 decay per second. Same burst semantics.
    """

    CONCURRENT = "concurrent"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"


@dataclass(frozen=True, slots=True)
class PipelineLimit:
    """Concurrency/rate limit configuration.

    limit: Maximum slots. For CONCURRENT: max simultaneous operations.
           For PER_MINUTE/PER_HOUR: token bucket capacity (burst size).
    kind: Type of limit enforcement.
    timeout: Max seconds to wait for slot acquisition.
    """

    limit: int
    kind: LimitKind = LimitKind.CONCURRENT
    timeout: int = 600

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")


# ---------------------------------------------------------------------------
# Internal state — ContextVar with mutable shared status
# ---------------------------------------------------------------------------


class _SharedStatus:
    """Mutable shared state across all tasks within a single pipeline run.

    Stored inside the ContextVar's frozen dataclass so all tasks forked via
    asyncio.gather share the same instance. ContextVar propagation copies the
    dataclass reference, but all copies point to the same _SharedStatus object.
    """

    __slots__ = ("_fallback_semaphores", "prefect_available")

    def __init__(self) -> None:
        self.prefect_available: bool = True
        self._fallback_semaphores: dict[str, asyncio.Semaphore] = {}

    def get_fallback_semaphore(self, name: str, limit: int) -> asyncio.Semaphore:
        """Lazily create per-process fallback semaphore."""
        if name not in self._fallback_semaphores:
            self._fallback_semaphores[name] = asyncio.Semaphore(limit)
        return self._fallback_semaphores[name]


@dataclass(frozen=True, slots=True)
class _LimitsState:
    """Immutable container holding limit config and mutable shared status."""

    limits: Mapping[str, PipelineLimit]
    status: _SharedStatus


_EMPTY_STATE = _LimitsState(limits=MappingProxyType({}), status=_SharedStatus())
_limits_state: ContextVar[_LimitsState] = ContextVar("_pipeline_limits_state", default=_EMPTY_STATE)


def _set_limits_state(state: _LimitsState) -> Token[_LimitsState]:
    """Set limits state in ContextVar, returning token for reset."""
    return _limits_state.set(state)


def _reset_limits_state(token: Token[_LimitsState]) -> None:
    """Reset limits state to previous value."""
    _limits_state.reset(token)


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _acquire_local_semaphore(name: str, cfg: PipelineLimit, status: _SharedStatus, wait_seconds: int) -> AsyncGenerator[None, None]:
    """Acquire local semaphore fallback and yield."""
    sem = status.get_fallback_semaphore(name, cfg.limit)
    try:
        await asyncio.wait_for(sem.acquire(), timeout=wait_seconds)
    except TimeoutError:
        raise AcquireConcurrencySlotTimeoutError(f"Local semaphore timeout for '{name}' after {wait_seconds}s") from None
    try:
        yield
    finally:
        sem.release()


@asynccontextmanager
async def pipeline_concurrency(
    name: str,
    *,
    timeout: int | None = None,
) -> AsyncGenerator[None, None]:
    """Acquire a concurrency/rate-limit slot for an operation.

    For CONCURRENT limits: slot held during block, released on exit.
    For PER_MINUTE/PER_HOUR: slot acquired (decays automatically), exit is no-op.

    Falls back to per-process asyncio.Semaphore when Prefect is unavailable (CONCURRENT only).
    Timeout always raises AcquireConcurrencySlotTimeoutError.
    """
    state = _limits_state.get()
    cfg = state.limits.get(name)
    if cfg is None:
        available = ", ".join(sorted(state.limits)) or "(none)"
        raise KeyError(f"pipeline_concurrency({name!r}) not registered. Available limits: {available}. Declare it on PipelineDeployment.concurrency_limits.")

    effective_timeout = timeout if timeout is not None else cfg.timeout

    # Prefect unavailable — use local semaphore fallback
    if not state.status.prefect_available:
        if cfg.kind == LimitKind.CONCURRENT:
            async with _acquire_local_semaphore(name, cfg, state.status, effective_timeout):
                yield
        else:
            yield
        return

    # Prefect available — use global concurrency/rate limiting
    yielded = False
    try:
        match cfg.kind:
            case LimitKind.CONCURRENT:
                async with concurrency(name, occupy=1, timeout_seconds=effective_timeout, strict=False):
                    yielded = True
                    yield
            case LimitKind.PER_MINUTE | LimitKind.PER_HOUR:
                await rate_limit(name, occupy=1, timeout_seconds=effective_timeout, strict=False)
                yielded = True
                yield
    except AcquireConcurrencySlotTimeoutError:
        raise
    except ConcurrencySlotAcquisitionError as e:
        logger.warning("Prefect concurrency unavailable for %r, falling back to local semaphore: %s", name, e)
        state.status.prefect_available = False
        if yielded:
            return
        # Use local fallback for this call
        if cfg.kind == LimitKind.CONCURRENT:
            async with _acquire_local_semaphore(name, cfg, state.status, effective_timeout):
                yield
        else:
            yield


# ---------------------------------------------------------------------------
# Upsert — called at start of PipelineDeployment.run()
# ---------------------------------------------------------------------------


def _slot_decay_per_second(cfg: PipelineLimit) -> float:
    """Compute Prefect slot_decay_per_second. Returns 0.0 for CONCURRENT (explicit, not None)."""
    match cfg.kind:
        case LimitKind.CONCURRENT:
            return 0.0
        case LimitKind.PER_MINUTE:
            return cfg.limit / _SECONDS_PER_MINUTE
        case LimitKind.PER_HOUR:
            return cfg.limit / _SECONDS_PER_HOUR


async def _ensure_concurrency_limits(limits: Mapping[str, PipelineLimit]) -> None:
    """Idempotently create/update all concurrency limits in Prefect server.

    Called at start of PipelineDeployment.run(). If Prefect is unavailable,
    logs warning and sets prefect_available=False (falls back to local semaphores).
    """
    if not limits:
        return
    state = _limits_state.get()
    try:
        async with get_client() as client:
            for name, cfg in limits.items():
                await client.upsert_global_concurrency_limit_by_name(
                    name=name,
                    limit=cfg.limit,
                    slot_decay_per_second=_slot_decay_per_second(cfg),
                )
                logger.debug("Concurrency limit '%s': limit=%d, kind=%s", name, cfg.limit, cfg.kind)
    except Exception as e:
        logger.warning("Prefect unavailable; falling back to local semaphores: %s", e)
        state.status.prefect_available = False


# ---------------------------------------------------------------------------
# Validation — called from PipelineDeployment.__init_subclass__
# ---------------------------------------------------------------------------


def _validate_concurrency_limits(
    deployment_name: str,
    raw: Mapping[str, PipelineLimit],
) -> Mapping[str, PipelineLimit]:
    """Validate concurrency_limits at class definition time. Returns immutable copy."""
    if not raw:
        return MappingProxyType({})
    for name, config in raw.items():
        if not isinstance(name, str):
            raise TypeError(f"{deployment_name}.concurrency_limits key must be str, got {type(name).__name__}")
        if not _VALID_LIMIT_NAME.match(name):
            raise TypeError(f"{deployment_name}.concurrency_limits: invalid name '{name}'. Must match [a-zA-Z0-9_-]+")
        if not isinstance(config, PipelineLimit):
            raise TypeError(f"{deployment_name}.concurrency_limits['{name}'] must be PipelineLimit, got {type(config).__name__}")
        if not isinstance(config.kind, LimitKind):
            raise TypeError(f"{deployment_name}.concurrency_limits['{name}'].kind must be LimitKind, got {type(config.kind).__name__}")
    return MappingProxyType(dict(raw))


__all__ = [
    "LimitKind",
    "PipelineLimit",
    "_LimitsState",
    "_SharedStatus",
    "_ensure_concurrency_limits",
    "_reset_limits_state",
    "_set_limits_state",
    "_validate_concurrency_limits",
    "pipeline_concurrency",
]
