"""Storage abstraction for local filesystem and Google Cloud Storage.

@public

Provides async storage operations with automatic retry for GCS.
Supports local filesystem and GCS backends with a unified API.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Coroutine, Protocol, TypeVar

from prefect_gcp.cloud_storage import GcsBucket

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import settings

logger = get_pipeline_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy for async operations with exponential backoff.

    @public

    Args:
        attempts: Maximum number of attempts (default 3)
        base_delay: Initial delay in seconds (default 1.0)
        max_delay: Maximum delay between retries (default 10.0)
    """

    attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0


def retry_async(
    policy: RetryPolicy,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Decorator for async functions with exponential backoff retry.

    @public

    Args:
        policy: RetryPolicy configuration

    Returns:
        Decorated async function with retry logic
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(policy.attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < policy.attempts - 1:
                        delay = min(policy.base_delay * (2**attempt), policy.max_delay)
                        logger.warning(
                            f"Storage operation failed: {e}. Retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{policy.attempts})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Storage operation failed after {policy.attempts} attempts: {e}"
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error: no exception but no result")

        return wrapper

    return decorator


@dataclass(frozen=True)
class ObjectInfo:
    """Storage object metadata.

    @public

    Attributes:
        key: Relative path (POSIX-style, no leading slash)
        size: Size in bytes (-1 if unknown)
        is_dir: True if this is a directory
    """

    key: str
    size: int
    is_dir: bool


def _norm_rel(path: str) -> str:
    """Normalize path to POSIX-style relative format.

    Returns:
        Normalized relative path string.
    """
    if not path:
        return ""
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    clean: list[str] = []
    for p in parts:
        if p == ".." and clean:
            clean.pop()
        elif p != "..":
            clean.append(p)
    return "/".join(clean)


def _join_rel(*parts: str) -> str:
    """Join and normalize path parts.

    Returns:
        Joined and normalized path string.
    """
    valid_parts = [_norm_rel(p) for p in parts if p]
    return _norm_rel("/".join(valid_parts)) if valid_parts else ""


class _AsyncBackend(Protocol):
    """Protocol for storage backend implementations."""

    async def url_for(self, key: str) -> str: ...
    async def exists(self, key: str) -> bool: ...
    async def list(self, prefix: str, recursive: bool, include_dirs: bool) -> list[ObjectInfo]: ...
    async def read_bytes(self, key: str) -> bytes: ...
    async def write_bytes(self, key: str, data: bytes) -> None: ...
    async def delete(self, key: str, missing_ok: bool) -> None: ...
    async def copy_from(self, other: _AsyncBackend, src_prefix: str, dst_prefix: str) -> None: ...


class _FileBackend:
    """Local filesystem backend using async I/O."""

    def __init__(self, root: Path):
        self.root = root

    def _abs(self, key: str) -> Path:
        return (self.root / _norm_rel(key)).resolve()

    async def url_for(self, key: str) -> str:
        return f"file://{self._abs(key)}"

    async def exists(self, key: str) -> bool:
        p = self._abs(key)
        return await asyncio.to_thread(p.exists)

    async def list(self, prefix: str, recursive: bool, include_dirs: bool) -> list[ObjectInfo]:
        base = self._abs(prefix)
        out: list[ObjectInfo] = []
        if not await asyncio.to_thread(base.exists):
            return out

        if recursive:

            def _walk() -> list[ObjectInfo]:
                items: list[ObjectInfo] = []
                for root, dirs, files in os.walk(base):
                    root_path = Path(root)
                    if include_dirs:
                        for d in dirs:
                            rel = str((root_path / d).relative_to(self.root)).replace("\\", "/")
                            items.append(ObjectInfo(key=_norm_rel(rel), size=-1, is_dir=True))
                    for f in files:
                        p = root_path / f
                        rel = str(p.relative_to(self.root)).replace("\\", "/")
                        items.append(
                            ObjectInfo(key=_norm_rel(rel), size=p.stat().st_size, is_dir=False)
                        )
                return items

            return await asyncio.to_thread(_walk)
        else:

            def _scan() -> list[ObjectInfo]:
                items: list[ObjectInfo] = []
                with os.scandir(base) as it:
                    for entry in it:
                        rel = str((Path(entry.path)).relative_to(self.root)).replace("\\", "/")
                        if entry.is_dir():
                            if include_dirs:
                                items.append(ObjectInfo(key=_norm_rel(rel), size=-1, is_dir=True))
                        else:
                            size = entry.stat().st_size
                            items.append(ObjectInfo(key=_norm_rel(rel), size=size, is_dir=False))
                return items

            return await asyncio.to_thread(_scan)

    async def read_bytes(self, key: str) -> bytes:
        p = self._abs(key)
        return await asyncio.to_thread(p.read_bytes)

    async def write_bytes(self, key: str, data: bytes) -> None:
        p = self._abs(key)

        def _write() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

        await asyncio.to_thread(_write)

    async def delete(self, key: str, missing_ok: bool) -> None:
        p = self._abs(key)

        def _delete() -> None:
            if not p.exists():
                if not missing_ok:
                    raise FileNotFoundError(str(p))
                return
            if p.is_dir():
                # Delete directory recursively
                for root, dirs, files in os.walk(p, topdown=False):
                    for name in files:
                        Path(root, name).unlink(missing_ok=True)
                    for name in dirs:
                        Path(root, name).rmdir()
                p.rmdir()
            else:
                p.unlink()

        await asyncio.to_thread(_delete)

    async def copy_from(self, other: _AsyncBackend, src_prefix: str, dst_prefix: str) -> None:
        objs = await other.list(src_prefix, recursive=True, include_dirs=False)
        for obj in objs:
            src_key = _join_rel(src_prefix, obj.key) if src_prefix else obj.key
            dst_key = _join_rel(dst_prefix, obj.key) if dst_prefix else obj.key
            data = await other.read_bytes(src_key)
            await self.write_bytes(dst_key, data)


class _GCSBackend:
    """Google Cloud Storage backend using Prefect GcsBucket block."""

    def __init__(self, *, bucket_block: GcsBucket, prefix: str, policy: RetryPolicy):
        self.bucket_block = bucket_block
        self.prefix = _norm_rel(prefix)
        self._retry = policy

    def _abs_key(self, key: str) -> str:
        rel = _norm_rel(key)
        return f"{self.prefix}/{rel}" if self.prefix else rel

    async def url_for(self, key: str) -> str:
        abs_key = self._abs_key(key)
        return f"gs://{self.bucket_block.bucket}/{abs_key}"

    @retry_async(RetryPolicy())
    async def exists(self, key: str) -> bool:
        abs_key = self._abs_key(key)
        blobs = await self.bucket_block.list_blobs(abs_key)  # type: ignore
        for b in blobs:
            if b.name.rstrip("/") == self._with_bucket_folder(abs_key).rstrip("/"):  # type: ignore
                return True
        return False

    def _with_bucket_folder(self, path: str) -> str:
        """Compose the full object path including the block's bucket_folder.

        Returns:
            Full object path with bucket folder prefix.
        """
        folder = self.bucket_block.bucket_folder or ""
        if folder and not folder.endswith("/"):
            folder = folder + "/"
        if folder and path.startswith(folder):
            return path
        return f"{folder}{path}" if folder else path

    @retry_async(RetryPolicy())
    async def list(self, prefix: str, recursive: bool, include_dirs: bool) -> list[ObjectInfo]:
        abs_prefix = self._abs_key(prefix)
        blobs = await self.bucket_block.list_blobs(abs_prefix)  # type: ignore
        effective_root = self._with_bucket_folder(abs_prefix).rstrip("/")
        out: list[ObjectInfo] = []
        seen_dirs = set()

        for blob in blobs:
            full = blob.name.rstrip("/")  # type: ignore
            if not full:
                continue

            # Make relative path by stripping effective_root
            if effective_root and full.startswith(effective_root + "/"):
                rel = full[len(effective_root) + 1 :]
            elif effective_root == full:
                rel = ""
            else:
                continue  # Out of scope

            if not recursive and "/" in rel:
                # Only top-level entries
                first = rel.split("/", 1)[0]
                if include_dirs:
                    seen_dirs.add(first)
                continue

            # Treat everything as files (GCS has no real directories)
            if rel:
                out.append(ObjectInfo(key=_norm_rel(rel), size=-1, is_dir=False))

        if include_dirs and not recursive:
            for d in sorted(seen_dirs):
                out.append(ObjectInfo(key=_norm_rel(d), size=-1, is_dir=True))

        return out

    @retry_async(RetryPolicy())
    async def read_bytes(self, key: str) -> bytes:
        abs_key = self._abs_key(key)
        return await self.bucket_block.read_path(abs_key)  # type: ignore

    @retry_async(RetryPolicy())
    async def write_bytes(self, key: str, data: bytes) -> None:
        abs_key = self._abs_key(key)
        await self.bucket_block.write_path(abs_key, data)  # type: ignore

    @retry_async(RetryPolicy())
    async def delete(self, key: str, missing_ok: bool) -> None:
        abs_key = self._abs_key(key)
        blobs = await self.bucket_block.list_blobs(abs_key)  # type: ignore
        deleted_any = False

        for blob in blobs:
            try:
                await self.bucket_block.delete_object(blob.name)  # type: ignore
                deleted_any = True
            except Exception:
                if not missing_ok:
                    raise

        if not deleted_any and not missing_ok:
            # Try single object delete
            try:
                await self.bucket_block.delete_object(abs_key)  # type: ignore
            except Exception:
                if not missing_ok:
                    raise

    @retry_async(RetryPolicy())
    async def copy_from(self, other: _AsyncBackend, src_prefix: str, dst_prefix: str) -> None:
        objs = await other.list(src_prefix, recursive=True, include_dirs=False)
        for obj in objs:
            src_key = _join_rel(src_prefix, obj.key) if src_prefix else obj.key
            dst_key = _join_rel(dst_prefix, obj.key) if dst_prefix else obj.key
            data = await other.read_bytes(src_key)
            await self.write_bytes(dst_key, data)


class Storage:
    """Unified async storage interface for local filesystem and Google Cloud Storage.

    @public

    Supports:
        - Local filesystem (file:// or relative paths)
        - Google Cloud Storage (gs:// URIs with Prefect GcsBucket block)
        - Future: AWS S3 support planned

    Examples:
        >>> # Local filesystem
        >>> storage = Storage.from_uri("./data")
        >>> storage = Storage.from_uri("file:///absolute/path")
        >>>
        >>> # Google Cloud Storage (uses settings.gcs_block by default)
        >>> storage = Storage.from_uri("gs://bucket/prefix")  # Uses GCS_BLOCK from settings
        >>> storage = Storage.from_uri("gs://bucket/prefix", gcs_block="my-gcs-block")  # Override
        >>>
        >>> # Use with subdirectories
        >>> doc_storage = storage.with_base("documents")
        >>> await doc_storage.write_text("file.txt", "content")
    """

    def __init__(self, scheme: str, backend: _AsyncBackend, base_prefix: str = ""):
        """Initialize storage client with backend and base prefix."""
        self.scheme = scheme
        self._backend = backend
        self._base = _norm_rel(base_prefix)

    @staticmethod
    def from_uri(
        uri: str, *, gcs_block: str | None = None, retry: RetryPolicy | None = None
    ) -> Storage:
        """Create Storage instance from URI.

        Args:
            uri: Storage URI (file://, gs://, or relative path)
            gcs_block: Prefect GcsBucket block name for GCS (defaults to settings.gcs_block)
            retry: Custom retry policy

        Returns:
            Storage instance configured for the URI

        Raises:
            ValueError: If URI scheme is unsupported or GCS block missing
        """
        policy = retry or RetryPolicy()

        # Local filesystem (no scheme)
        if "://" not in uri:
            root = Path(uri).expanduser().resolve()
            return Storage("file", _FileBackend(root), "")  # type: ignore

        scheme, rest = uri.split("://", 1)
        if scheme == "file":
            root = Path(rest).expanduser().resolve()
            return Storage("file", _FileBackend(root), "")  # type: ignore

        if scheme != "gs":
            raise ValueError(f"Unsupported URI scheme: {scheme}. Use file:// or gs://")

        # Parse GCS URI
        if "/" in rest:
            bucket_name, prefix = rest.split("/", 1)
        else:
            bucket_name, prefix = rest, ""

        # Get GCS block from parameter or settings
        if gcs_block:
            block_name = gcs_block
        else:
            block_name = settings.gcs_block

        if not block_name:
            raise ValueError(
                "GCS access requires a Prefect GcsBucket block name. "
                "Pass gcs_block= or set GCS_BLOCK in settings/environment."
            )

        bucket_block = GcsBucket.load(block_name)

        # Verify bucket name matches
        if bucket_block.bucket != bucket_name:  # type: ignore
            logger.warning(
                f"GcsBucket block '{block_name}' points to bucket '{bucket_block.bucket}' "  # type: ignore
                f"but URI requested '{bucket_name}'. Using block's bucket."
            )

        backend = _GCSBackend(bucket_block=bucket_block, prefix=prefix, policy=policy)  # type: ignore
        return Storage("gs", backend, "")  # type: ignore

    def with_base(self, subpath: str) -> Storage:
        """Create a new Storage instance with a different base path.

        Args:
            subpath: Subdirectory path (use "" for root)

        Returns:
            New Storage instance rooted at the subdirectory
        """
        return Storage(self.scheme, self._backend, _join_rel(self._base, subpath))

    def _rel_to_root(self, path: str) -> str:
        """Convert path relative to base to path relative to root.

        Returns:
            Path relative to storage root.
        """
        return _join_rel(self._base, path)

    async def url_for(self, path: str) -> str:
        """Get the full URL for a storage path.

        Returns:
            Full URL for the given path.
        """
        rel = self._rel_to_root(path)
        return await self._backend.url_for(rel)

    async def exists(self, path: str) -> bool:
        """Check if a path exists.

        Returns:
            True if path exists, False otherwise.
        """
        rel = self._rel_to_root(path)
        return await self._backend.exists(rel)

    async def list(
        self, prefix: str = "", *, recursive: bool = True, include_dirs: bool = True
    ) -> list[ObjectInfo]:
        """List objects under a prefix.

        Args:
            prefix: Path prefix to list under
            recursive: Include nested objects
            include_dirs: Include directory entries

        Returns:
            List of ObjectInfo for matching objects
        """
        rel = self._rel_to_root(prefix)
        items = await self._backend.list(rel, recursive=recursive, include_dirs=include_dirs)

        # Re-relativize to the requested prefix
        def _strip_base(k: str) -> str:
            if not rel:
                return k
            if k == rel:
                return ""
            if k.startswith(rel + "/"):
                return k[len(rel) + 1 :]
            return k

        return [
            ObjectInfo(key=_norm_rel(_strip_base(it.key)), size=it.size, is_dir=it.is_dir)
            for it in items
        ]

    async def read_bytes(self, path: str) -> bytes:
        """Read file contents as bytes.

        Returns:
            File contents as bytes.
        """
        rel = self._rel_to_root(path)
        return await self._backend.read_bytes(rel)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file."""
        rel = self._rel_to_root(path)
        await self._backend.write_bytes(rel, data)

    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read file contents as text.

        Returns:
            File contents as string.
        """
        return (await self.read_bytes(path)).decode(encoding)

    async def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        """Write text to a file."""
        await self.write_bytes(path, text.encode(encoding))

    async def delete(self, path: str, *, missing_ok: bool = True) -> None:
        """Delete a file or directory."""
        rel = self._rel_to_root(path)
        await self._backend.delete(rel, missing_ok)

    async def copy_from(
        self, other: Storage, *, src_prefix: str = "", dst_prefix: str = ""
    ) -> None:
        """Copy files from another storage instance.

        Args:
            other: Source storage instance
            src_prefix: Source path prefix
            dst_prefix: Destination path prefix
        """
        src_rel = other._rel_to_root(src_prefix)
        dst_rel = self._rel_to_root(dst_prefix)
        await self._backend.copy_from(other._backend, src_rel, dst_rel)
