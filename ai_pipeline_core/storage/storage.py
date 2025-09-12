"""Storage abstraction for local filesystem and Google Cloud Storage.

Provides async storage operations with automatic retry for GCS.
Supports local filesystem and GCS backends with a unified API.
"""

from __future__ import annotations

import asyncio
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any

from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect_gcp.cloud_storage import GcpCredentials, GcsBucket
from pydantic import BaseModel, Field

from ai_pipeline_core.logging import get_pipeline_logger
from ai_pipeline_core.settings import settings

__all__ = ["Storage", "LocalStorage", "GcsStorage", "RetryPolicy", "ObjectInfo"]

logger = get_pipeline_logger(__name__)


# ---------- Models ----------


class RetryPolicy(BaseModel, frozen=True):
    """Retry policy for async operations with exponential backoff.

    Args:
        attempts: Maximum number of attempts (default 3)
        base_delay: Initial delay in seconds (default 0.5)
        max_delay: Maximum delay between retries (default 5.0)
        jitter: Random jitter factor (default 0.15)
        retry_exceptions: Tuple of exceptions to retry on
    """

    attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=0.5, ge=0.0)
    max_delay: float = Field(default=5.0, ge=0.0)
    jitter: float = Field(default=0.15, ge=0.0)
    retry_exceptions: tuple[type[BaseException], ...] = Field(default_factory=tuple)


class ObjectInfo(BaseModel, frozen=True):
    """Storage object metadata.

    Attributes:
        key: Relative path (POSIX-style, no leading slash)
        size: Size in bytes (-1 if unknown)
        is_dir: True if this is a directory
    """

    key: str
    size: int
    is_dir: bool


# ---------- Helpers ----------


def _posix_rel(s: str) -> str:
    if not s:
        return ""
    parts: list[str] = []
    for t in s.replace("\\", "/").split("/"):
        if t in ("", "."):
            continue
        if t == "..":
            if parts:
                parts.pop()
            continue
        parts.append(t)
    return "/".join(parts)


def _join_posix(*parts: str) -> str:
    return _posix_rel("/".join(p for p in map(_posix_rel, parts) if p))


# ---------- Abstract facade ----------


class Storage(ABC):
    """Abstract storage interface for file operations.

    Provides a unified API for local filesystem and Google Cloud Storage.
    Supports async operations with automatic retry for cloud storage.

    Example:
        >>> # Load from local filesystem
        >>> storage = await Storage.from_uri("./data")
        >>>
        >>> # Load from GCS
        >>> storage = await Storage.from_uri("gs://bucket/data")
    """

    @classmethod
    async def from_uri(cls, uri: str, retry: RetryPolicy | None = None) -> "Storage":
        """Create storage instance from URI.

        Args:
            uri: Storage URI (local path, file://, or gs://)
            retry: Optional retry policy for GCS operations

        Returns:
            Storage instance for the given URI

        Raises:
            ValueError: If URI scheme is unsupported or path is invalid
        """
        # Handle local paths without file:// prefix
        if "://" not in uri:
            # Treat as local filesystem path
            base = Path(uri).expanduser().resolve()
            if base.exists() and not base.is_dir():
                raise ValueError("Local path must point to a directory")
            return LocalStorage(base)

        scheme, rest = uri.split("://", 1)

        if scheme == "file":
            base = Path("/" + rest.lstrip("/")).expanduser().resolve()
            if base.exists() and not base.is_dir():
                raise ValueError("file:// URI must point to a directory")
            return LocalStorage(base)

        if scheme == "gs":
            bucket, *maybe_prefix = rest.split("/", 1)
            folder = _posix_rel(maybe_prefix[0] if maybe_prefix else "")
            return GcsStorage(
                bucket=bucket,
                bucket_folder=folder,
                gcp_credentials=None,  # Will try to load from settings if configured
                retry=retry,
            )

        raise ValueError(f"Unsupported scheme: {scheme}")

    # Core API — abstract in the base
    @abstractmethod
    def url_for(self, path: str) -> str:
        """Get URL for path."""
        ...

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...

    @abstractmethod
    async def list(
        self, prefix: str = "", *, recursive: bool = True, include_dirs: bool = True
    ) -> list[ObjectInfo]:
        """List objects with prefix."""
        ...

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """Read bytes from path."""
        ...

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to path."""
        ...

    @abstractmethod
    async def delete(self, path: str, *, missing_ok: bool = True) -> None:
        """Delete path."""
        ...

    async def copy_from(
        self, other: "Storage", *, src_prefix: str = "", dst_prefix: str = ""
    ) -> None:
        """Copy from another storage."""
        items = await other.list(src_prefix, recursive=True, include_dirs=False)
        for it in items:
            data = await other.read_bytes(_join_posix(src_prefix, it.key))
            await self.write_bytes(_join_posix(dst_prefix, it.key), data)

    def with_base(self, subpath: str) -> "Storage":
        """Create sub-storage with base path."""
        raise NotImplementedError("Subclasses must implement with_base")

    async def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from path.

        Args:
            path: Path to read from
            encoding: Text encoding to use

        Returns:
            Text content of the file
        """
        data = await self.read_bytes(path)
        return data.decode(encoding)

    async def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        """Write text to path."""
        await self.write_bytes(path, text.encode(encoding))


# ---------- Local filesystem ----------


class LocalStorage(Storage):
    """Local filesystem storage implementation."""

    def __init__(self, base: Path):
        """Initialize with base path."""
        self._base = base

    def with_base(self, subpath: str) -> "Storage":
        """Create sub-storage with base path.

        Args:
            subpath: Relative path for sub-storage

        Returns:
            New LocalStorage instance with updated base path
        """
        return LocalStorage(self._base / _posix_rel(subpath))

    def _abs(self, rel: str) -> Path:
        return (self._base / _posix_rel(rel)).resolve()

    def url_for(self, path: str) -> str:
        """Get file URL for path.

        Args:
            path: Relative path

        Returns:
            File URL for the path
        """
        return self._abs(path).as_uri()

    async def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check

        Returns:
            True if path exists, False otherwise
        """
        return self._abs(path).exists()

    async def list(
        self, prefix: str = "", *, recursive: bool = True, include_dirs: bool = True
    ) -> list[ObjectInfo]:
        """List objects with prefix.

        Args:
            prefix: Path prefix to list
            recursive: Whether to list recursively
            include_dirs: Whether to include directories

        Returns:
            List of object information
        """
        base = self._abs(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [ObjectInfo(key="", size=base.stat().st_size, is_dir=False)]

        out: list[ObjectInfo] = []
        if recursive:
            for root, dirs, files in os.walk(base):
                r = Path(root)
                if include_dirs:
                    for d in dirs:
                        out.append(
                            ObjectInfo(
                                key=(r / d).relative_to(base).as_posix(), size=-1, is_dir=True
                            )
                        )
                for f in files:
                    fp = r / f
                    out.append(
                        ObjectInfo(
                            key=fp.relative_to(base).as_posix(),
                            size=fp.stat().st_size,
                            is_dir=False,
                        )
                    )
            return out

        with os.scandir(base) as it:
            for e in it:
                if e.is_dir():
                    if include_dirs:
                        out.append(ObjectInfo(key=e.name, size=-1, is_dir=True))
                else:
                    out.append(ObjectInfo(key=e.name, size=e.stat().st_size, is_dir=False))
        return out

    async def read_bytes(self, path: str) -> bytes:
        """Read bytes from path.

        Args:
            path: Path to read from

        Returns:
            Binary content of the file
        """
        return self._abs(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to path.

        Args:
            path: Path to write to
            data: Binary data to write
        """
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def delete(self, path: str, *, missing_ok: bool = True) -> None:
        """Delete path.

        Args:
            path: Path to delete
            missing_ok: If True, don't raise error if path doesn't exist

        Raises:
            FileNotFoundError: If path doesn't exist and missing_ok is False
        """
        p = self._abs(path)
        if not p.exists():
            if not missing_ok:
                raise FileNotFoundError(str(p))
            return
        if p.is_dir():
            for root, dirs, files in os.walk(p, topdown=False):
                for f in files:
                    Path(root, f).unlink(missing_ok=True)
                for d in dirs:
                    Path(root, d).rmdir()
            p.rmdir()
        else:
            p.unlink()


# ---------- Google Cloud Storage ----------


class GcsStorage(Storage):
    """Google Cloud Storage implementation."""

    def __init__(
        self,
        bucket: str,
        bucket_folder: str = "",
        gcp_credentials: GcpCredentials | None = None,
        retry: RetryPolicy | None = None,
    ):
        """Initialize GCS storage.

        Args:
            bucket: GCS bucket name
            bucket_folder: Optional folder within bucket
            gcp_credentials: Optional GCP credentials
            retry: Optional retry policy for operations
        """
        # If no credentials provided, try to load from settings
        if gcp_credentials is None and hasattr(settings, "gcs_service_account_file"):
            service_account_file = getattr(settings, "gcs_service_account_file", "")
            if service_account_file:
                try:
                    gcp_credentials = GcpCredentials(
                        service_account_file=Path(service_account_file)
                    )
                except Exception:
                    # If loading fails, pass None to GcsBucket
                    pass

        if not gcp_credentials:
            gcp_credentials = GcpCredentials()

        # GcsBucket expects credentials or nothing, not None
        self.block = GcsBucket(
            bucket=bucket, bucket_folder=bucket_folder, gcp_credentials=gcp_credentials
        )
        self.retry = retry or RetryPolicy()

    async def create_bucket(self) -> None:
        """Create the GCS bucket if it doesn't exist."""
        await self.block.create_bucket()  # type: ignore[attr-defined]

    def with_base(self, subpath: str) -> "Storage":
        """Create sub-storage with base path.

        Args:
            subpath: Relative path for sub-storage

        Returns:
            New GcsStorage instance with updated base path
        """
        new_folder = _join_posix(self.block.bucket_folder or "", subpath)
        # Get credentials if they exist
        creds = getattr(self.block, "gcp_credentials", None)
        return GcsStorage(
            bucket=self.block.bucket,  # type: ignore[arg-type]
            bucket_folder=new_folder,
            gcp_credentials=creds if creds is not None else None,
            retry=self.retry,
        )

    def _base(self) -> str:
        return self.block.bucket_folder or ""

    def _abs_name(self, rel: str) -> str:
        base = self._base()
        return str(PurePosixPath(base) / _posix_rel(rel)) if base else _posix_rel(rel)

    def _rel_from_abs(self, abs_name: str) -> str:
        base = self._base()
        if base and abs_name.startswith(base):
            return _posix_rel(abs_name[len(base) :])
        return _posix_rel(abs_name)

    def _rex(self) -> tuple[type[BaseException], ...]:
        return self.retry.retry_exceptions or (Exception,)

    async def _retry(self, label: str, fn) -> Any:  # type: ignore[no-untyped-def]
        last: BaseException | None = None
        for i in range(1, self.retry.attempts + 1):
            try:
                return await fn()
            except asyncio.CancelledError:
                raise
            except self._rex() as e:  # type: ignore[misc]
                last = e
                if i == self.retry.attempts:
                    break
                delay = min(self.retry.base_delay * (2 ** (i - 1)), self.retry.max_delay)
                delay += delay * self.retry.jitter * random.random()
                logger.warning(
                    f"GCS {label} failed: {e!s}. "
                    f"retry {i}/{self.retry.attempts - 1} in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
        assert last is not None
        logger.error(f"GCS {label} failed after {self.retry.attempts} attempts: {last!s}")
        raise last

    def url_for(self, path: str) -> str:
        """Get GCS URL for path.

        Args:
            path: Relative path

        Returns:
            GCS URL in format gs://bucket/path
        """
        return f"gs://{self.block.bucket}/{self._abs_name(path)}"

    async def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check

        Returns:
            True if path exists, False otherwise
        """
        name = self._abs_name(path)

        async def blob_exists() -> bool:
            """Check if blob exists.

            Returns:
                True if blob exists
            """
            bucket = await self.block.get_bucket()  # type: ignore[attr-defined]
            blob = bucket.blob(name)  # type: ignore[attr-defined]
            try:
                return await run_sync_in_worker_thread(blob.exists)
            except Exception:
                return False

        async def prefix_exists() -> bool:
            """Check if prefix exists.

            Returns:
                True if any objects exist with this prefix
            """
            blobs = await self.block.list_blobs(path)  # type: ignore[attr-defined]
            prefix_name = name.rstrip("/") + "/"
            return any(
                getattr(b, "name", None) == name
                or (getattr(b, "name", "").startswith(prefix_name) if hasattr(b, "name") else False)
                for b in blobs
            )

        if await self._retry("exists", blob_exists):
            return True
        return await self._retry("exists-prefix", prefix_exists)

    async def list(
        self, prefix: str = "", *, recursive: bool = True, include_dirs: bool = True
    ) -> list[ObjectInfo]:
        """List objects with prefix.

        Args:
            prefix: Path prefix to list
            recursive: Whether to list recursively
            include_dirs: Whether to include directories

        Returns:
            List of object information
        """
        blobs = await self._retry("list", lambda: self.block.list_blobs(prefix))
        base_abs = self._abs_name(prefix).rstrip("/")
        out: list[ObjectInfo] = []
        dirs: set[str] = set()

        def rel(name: str) -> str | None:
            """Get relative path from name.

            Args:
                name: Absolute blob name

            Returns:
                Relative path or None if not under prefix
            """
            n = name.rstrip("/")
            if not base_abs:
                return n
            if n == base_abs:
                return ""
            if n.startswith(base_abs + "/"):
                return n[len(base_abs) + 1 :]
            return None

        for b in blobs:
            r = rel(b.name)
            if r is None:
                continue
            if not recursive and "/" in r:
                if include_dirs:
                    dirs.add(r.split("/", 1)[0])
                continue
            size = getattr(b, "size", -1)
            out.append(ObjectInfo(key=_posix_rel(r), size=size, is_dir=False))

        if include_dirs and not recursive:
            out.extend(ObjectInfo(key=d, size=-1, is_dir=True) for d in sorted(dirs))

        if not out and prefix:
            bucket = await self.block.get_bucket()  # type: ignore[attr-defined]
            blob = bucket.blob(base_abs)  # type: ignore[attr-defined]
            if await run_sync_in_worker_thread(blob.exists):
                try:
                    await run_sync_in_worker_thread(blob.reload)
                    size = getattr(blob, "size", None)
                except Exception:
                    size = None
                out.append(
                    ObjectInfo(key="", size=int(size) if size is not None else -1, is_dir=False)
                )

        return out

    async def read_bytes(self, path: str) -> bytes:
        """Read bytes from path.

        Args:
            path: Path to read from

        Returns:
            Binary content of the file
        """
        # GcsBucket.read_path expects a key relative to bucket_folder
        return await self._retry("read_bytes", lambda: self.block.read_path(path))

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to path.

        Args:
            path: Path to write to
            data: Binary data to write
        """
        await self._retry("write_bytes", lambda: self.block.write_path(path, data))

    async def delete(self, path: str, *, missing_ok: bool = True) -> None:
        """Delete path.

        Args:
            path: Path to delete
            missing_ok: If True, don't raise error if path doesn't exist

        Raises:
            FileNotFoundError: If path doesn't exist and missing_ok is False
        """
        name = self._abs_name(path)
        bucket = await self.block.get_bucket()  # type: ignore[attr-defined]

        async def delete_exact() -> bool:
            """Try to delete exact blob.

            Returns:
                True if deletion succeeded
            """
            try:
                blob = bucket.blob(name)  # type: ignore[attr-defined]
                await run_sync_in_worker_thread(blob.delete)
                return True
            except Exception:
                return False

        if await self._retry("delete", delete_exact):
            return

        blobs = await self._retry("list-for-delete", lambda: self.block.list_blobs(path))
        if not blobs:
            if not missing_ok:
                raise FileNotFoundError(name)
            return
        await asyncio.gather(*[run_sync_in_worker_thread(b.delete) for b in blobs])
