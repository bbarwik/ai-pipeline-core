"""Unit tests for Storage base class and LocalStorage."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_pipeline_core.storage import ObjectInfo, RetryPolicy, Storage
from ai_pipeline_core.storage.storage import LocalStorage


class TestStorageFromUri:
    """Test Storage.from_uri() factory method."""

    @pytest.mark.asyncio
    async def test_from_uri_local_path(self):
        """Test from_uri with local filesystem path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = await Storage.from_uri(tmpdir)
            assert isinstance(storage, LocalStorage)
            assert storage._base == Path(tmpdir).resolve()  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_from_uri_file_scheme(self):
        """Test from_uri with file:// scheme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = await Storage.from_uri(f"file://{tmpdir}")
            assert isinstance(storage, LocalStorage)
            assert storage._base == Path(tmpdir).resolve()  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_from_uri_file_scheme_triple_slash(self):
        """Test from_uri with file:/// scheme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = await Storage.from_uri(f"file:///{tmpdir}")
            assert isinstance(storage, LocalStorage)
            # Should handle the triple slash correctly
            assert storage._base == Path(tmpdir).resolve()  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_from_uri_gs_scheme(self):
        """Test from_uri with gs:// scheme."""
        with patch("ai_pipeline_core.storage.storage.GcpCredentials") as mock_creds:
            with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket:
                mock_creds.return_value = MagicMock()
                mock_bucket.return_value = MagicMock()
                storage = await Storage.from_uri("gs://bucket/folder")
                # Should create GcsStorage
                from ai_pipeline_core.storage.storage import GcsStorage

                assert isinstance(storage, GcsStorage)
                mock_bucket.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_uri_gs_no_folder(self):
        """Test from_uri with gs:// scheme without folder."""
        with patch("ai_pipeline_core.storage.storage.GcpCredentials") as mock_creds:
            with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket:
                mock_creds_instance = MagicMock()
                mock_creds.return_value = mock_creds_instance
                mock_instance = MagicMock()
                mock_instance.bucket_folder = ""
                mock_bucket.return_value = mock_instance
                storage = await Storage.from_uri("gs://bucket")
                from ai_pipeline_core.storage.storage import GcsStorage

                assert isinstance(storage, GcsStorage)
                # GcsBucket should be created with empty bucket_folder
                mock_bucket.assert_called_once_with(
                    bucket="bucket", bucket_folder="", gcp_credentials=mock_creds_instance
                )

    @pytest.mark.asyncio
    async def test_from_uri_with_retry_policy(self):
        """Test from_uri with custom retry policy."""
        with patch("ai_pipeline_core.storage.storage.GcpCredentials") as mock_creds:
            with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket:
                mock_creds.return_value = MagicMock()
                mock_bucket.return_value = MagicMock()
                retry = RetryPolicy(attempts=5, base_delay=1.0)
                storage = await Storage.from_uri("gs://bucket/folder", retry=retry)
                from ai_pipeline_core.storage.storage import GcsStorage

                assert isinstance(storage, GcsStorage)
                assert storage.retry == retry

    @pytest.mark.asyncio
    async def test_from_uri_local_file_error(self):
        """Test from_uri raises error when local path is a file."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="must point to a directory"):
                await Storage.from_uri(tmpfile.name)

    @pytest.mark.asyncio
    async def test_from_uri_file_scheme_file_error(self):
        """Test from_uri raises error when file:// points to a file."""
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="must point to a directory"):
                await Storage.from_uri(f"file://{tmpfile.name}")

    @pytest.mark.asyncio
    async def test_from_uri_unsupported_scheme(self):
        """Test from_uri raises error for unsupported scheme."""
        with pytest.raises(ValueError, match="Unsupported scheme: s3"):
            await Storage.from_uri("s3://bucket/folder")

    @pytest.mark.asyncio
    async def test_from_uri_expanduser(self):
        """Test from_uri expands user home directory."""
        with patch("pathlib.Path.expanduser") as mock_expand:
            with patch("pathlib.Path.resolve") as mock_resolve:
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.is_dir", return_value=True):
                        mock_path = MagicMock()
                        mock_expand.return_value = mock_path
                        mock_resolve.return_value = mock_path
                        mock_path.resolve.return_value = mock_path
                        mock_path.exists.return_value = True
                        mock_path.is_dir.return_value = True

                        await Storage.from_uri("~/mydir")
                        mock_expand.assert_called_once()


class TestStorageBase:
    """Test Storage base class methods."""

    def test_with_base_not_implemented(self):
        """Test that base Storage.with_base raises NotImplementedError."""

        # Create a minimal concrete implementation for testing
        class MinimalStorage(Storage):
            def url_for(self, path: str) -> str:
                return ""

            async def exists(self, path: str) -> bool:
                return False

            async def list(
                self, prefix: str = "", *, recursive: bool = True, include_dirs: bool = True
            ) -> list[ObjectInfo]:
                return []

            async def read_bytes(self, path: str) -> bytes:
                return b""

            async def write_bytes(self, path: str, data: bytes) -> None:
                pass

            async def delete(self, path: str, *, missing_ok: bool = True) -> None:
                pass

        storage = MinimalStorage()
        with pytest.raises(NotImplementedError):
            storage.with_base("subpath")

    @pytest.mark.asyncio
    async def test_copy_from(self):
        """Test Storage.copy_from() method."""
        # Create mock source and destination storages
        source = MagicMock()
        source.list = AsyncMock(
            return_value=[
                ObjectInfo(key="file1.txt", size=100, is_dir=False),
                ObjectInfo(key="dir/file2.txt", size=200, is_dir=False),
            ]
        )
        source.read_bytes = AsyncMock(side_effect=[b"content1", b"content2"])

        dest = MagicMock()
        dest.write_bytes = AsyncMock()
        dest.copy_from = Storage.copy_from.__get__(dest, Storage)

        await dest.copy_from(source, src_prefix="src", dst_prefix="dst")

        # Verify reads from source
        source.list.assert_called_once_with("src", recursive=True, include_dirs=False)
        assert source.read_bytes.call_count == 2
        source.read_bytes.assert_any_call("src/file1.txt")
        source.read_bytes.assert_any_call("src/dir/file2.txt")

        # Verify writes to destination
        assert dest.write_bytes.call_count == 2
        dest.write_bytes.assert_any_call("dst/file1.txt", b"content1")
        dest.write_bytes.assert_any_call("dst/dir/file2.txt", b"content2")

    @pytest.mark.asyncio
    async def test_read_text(self):
        """Test Storage.read_text() method."""
        storage = MagicMock()
        storage.read_bytes = AsyncMock(return_value="Hello, 世界!".encode("utf-8"))
        storage.read_text = Storage.read_text.__get__(storage, Storage)

        text = await storage.read_text("file.txt")
        assert text == "Hello, 世界!"
        storage.read_bytes.assert_called_once_with("file.txt")

    @pytest.mark.asyncio
    async def test_read_text_custom_encoding(self):
        """Test Storage.read_text() with custom encoding."""
        storage = MagicMock()
        storage.read_bytes = AsyncMock(return_value="Hello".encode("utf-16"))
        storage.read_text = Storage.read_text.__get__(storage, Storage)

        text = await storage.read_text("file.txt", encoding="utf-16")
        assert text == "Hello"

    @pytest.mark.asyncio
    async def test_write_text(self):
        """Test Storage.write_text() method."""
        storage = MagicMock()
        storage.write_bytes = AsyncMock()
        storage.write_text = Storage.write_text.__get__(storage, Storage)

        await storage.write_text("file.txt", "Hello, 世界!")
        storage.write_bytes.assert_called_once_with("file.txt", "Hello, 世界!".encode("utf-8"))

    @pytest.mark.asyncio
    async def test_write_text_custom_encoding(self):
        """Test Storage.write_text() with custom encoding."""
        storage = MagicMock()
        storage.write_bytes = AsyncMock()
        storage.write_text = Storage.write_text.__get__(storage, Storage)

        await storage.write_text("file.txt", "Hello", encoding="utf-16")
        storage.write_bytes.assert_called_once_with("file.txt", "Hello".encode("utf-16"))


class TestLocalStorage:
    """Test LocalStorage implementation."""

    @pytest.fixture
    def local_storage(self):
        """Create LocalStorage with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalStorage(Path(tmpdir))

    def test_init(self):
        """Test LocalStorage initialization."""
        base = Path("/tmp/test")
        storage = LocalStorage(base)
        assert storage._base == base  # pyright: ignore[reportPrivateUsage]

    def test_with_base(self):
        """Test LocalStorage.with_base() method."""
        base = Path("/tmp/test")
        storage = LocalStorage(base)
        sub_storage = storage.with_base("sub/path")
        assert isinstance(sub_storage, LocalStorage)
        assert sub_storage._base == base / "sub/path"  # pyright: ignore[reportPrivateUsage]

    def test_url_for(self):
        """Test LocalStorage.url_for() method."""
        storage = LocalStorage(Path("/tmp/test"))
        url = storage.url_for("file.txt")
        assert url == Path("/tmp/test/file.txt").resolve().as_uri()

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test LocalStorage.exists() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # File doesn't exist
            assert await storage.exists("nonexistent.txt") is False

            # Create a file
            (Path(tmpdir) / "exists.txt").write_text("content")
            assert await storage.exists("exists.txt") is True

            # Create a directory
            (Path(tmpdir) / "subdir").mkdir()
            assert await storage.exists("subdir") is True

    @pytest.mark.asyncio
    async def test_list_recursive(self):
        """Test LocalStorage.list() with recursive option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create test structure
            (Path(tmpdir) / "file1.txt").write_text("content1")
            (Path(tmpdir) / "dir1").mkdir()
            (Path(tmpdir) / "dir1" / "file2.txt").write_text("content2")
            (Path(tmpdir) / "dir1" / "dir2").mkdir()
            (Path(tmpdir) / "dir1" / "dir2" / "file3.txt").write_text("content3")

            # List recursively with dirs
            items = await storage.list("", recursive=True, include_dirs=True)
            file_keys = sorted([i.key for i in items if not i.is_dir])
            dir_keys = sorted([i.key for i in items if i.is_dir])

            assert "file1.txt" in file_keys
            assert "dir1/file2.txt" in file_keys
            assert "dir1/dir2/file3.txt" in file_keys
            assert "dir1" in dir_keys
            assert "dir1/dir2" in dir_keys

            # List recursively without dirs
            items = await storage.list("", recursive=True, include_dirs=False)
            keys = sorted([i.key for i in items])
            assert "file1.txt" in keys
            assert "dir1/file2.txt" in keys
            assert "dir1/dir2/file3.txt" in keys
            assert all(not i.is_dir for i in items)

    @pytest.mark.asyncio
    async def test_list_non_recursive(self):
        """Test LocalStorage.list() without recursion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create test structure
            (Path(tmpdir) / "file1.txt").write_text("content1")
            (Path(tmpdir) / "dir1").mkdir()
            (Path(tmpdir) / "dir1" / "file2.txt").write_text("content2")

            # List non-recursively with dirs
            items = await storage.list("", recursive=False, include_dirs=True)
            file_keys = [i.key for i in items if not i.is_dir]
            dir_keys = [i.key for i in items if i.is_dir]

            assert "file1.txt" in file_keys
            assert "dir1" in dir_keys
            assert "dir1/file2.txt" not in file_keys

            # List non-recursively without dirs
            items = await storage.list("", recursive=False, include_dirs=False)
            keys = [i.key for i in items]
            assert "file1.txt" in keys
            assert "dir1" not in keys

    @pytest.mark.asyncio
    async def test_list_with_prefix(self):
        """Test LocalStorage.list() with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create test structure
            (Path(tmpdir) / "dir1").mkdir()
            (Path(tmpdir) / "dir1" / "file1.txt").write_text("content1")
            (Path(tmpdir) / "dir1" / "subdir").mkdir()
            (Path(tmpdir) / "dir1" / "subdir" / "file2.txt").write_text("content2")

            # List with prefix
            items = await storage.list("dir1", recursive=True, include_dirs=False)
            keys = sorted([i.key for i in items])
            assert keys == ["file1.txt", "subdir/file2.txt"]

    @pytest.mark.asyncio
    async def test_list_single_file(self):
        """Test LocalStorage.list() when prefix points to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create a file
            (Path(tmpdir) / "single.txt").write_text("content")

            # List the file directly
            items = await storage.list("single.txt")
            assert len(items) == 1
            assert items[0].key == ""
            assert items[0].size == 7  # len("content")
            assert items[0].is_dir is False

    @pytest.mark.asyncio
    async def test_list_nonexistent(self):
        """Test LocalStorage.list() with non-existent prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))
            items = await storage.list("nonexistent")
            assert items == []

    @pytest.mark.asyncio
    async def test_read_write_bytes(self):
        """Test LocalStorage read/write bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Write and read back
            data = b"test binary data \x00\x01\x02"
            await storage.write_bytes("test.bin", data)
            read_data = await storage.read_bytes("test.bin")
            assert read_data == data

            # Write to nested path (should create dirs)
            await storage.write_bytes("dir1/dir2/nested.bin", data)
            assert (Path(tmpdir) / "dir1" / "dir2" / "nested.bin").exists()
            read_data = await storage.read_bytes("dir1/dir2/nested.bin")
            assert read_data == data

    @pytest.mark.asyncio
    async def test_delete_file(self):
        """Test LocalStorage.delete() for files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create and delete a file
            (Path(tmpdir) / "file.txt").write_text("content")
            await storage.delete("file.txt")
            assert not (Path(tmpdir) / "file.txt").exists()

            # Delete non-existent with missing_ok=True (default)
            await storage.delete("nonexistent.txt", missing_ok=True)

            # Delete non-existent with missing_ok=False
            with pytest.raises(FileNotFoundError):
                await storage.delete("nonexistent.txt", missing_ok=False)

    @pytest.mark.asyncio
    async def test_delete_directory(self):
        """Test LocalStorage.delete() for directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create directory structure
            (Path(tmpdir) / "dir1").mkdir()
            (Path(tmpdir) / "dir1" / "file1.txt").write_text("content1")
            (Path(tmpdir) / "dir1" / "dir2").mkdir()
            (Path(tmpdir) / "dir1" / "dir2" / "file2.txt").write_text("content2")

            # Delete the directory
            await storage.delete("dir1")
            assert not (Path(tmpdir) / "dir1").exists()

    @pytest.mark.asyncio
    async def test_delete_empty_directory(self):
        """Test LocalStorage.delete() for empty directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(Path(tmpdir))

            # Create empty directory
            (Path(tmpdir) / "empty").mkdir()

            # Delete it
            await storage.delete("empty")
            assert not (Path(tmpdir) / "empty").exists()

    def test_abs_path_handling(self):
        """Test LocalStorage._abs() path resolution."""
        storage = LocalStorage(Path("/tmp/base"))

        # Normal paths
        assert storage._abs("file.txt") == Path("/tmp/base/file.txt").resolve()  # pyright: ignore[reportPrivateUsage]
        assert storage._abs("dir/file.txt") == Path("/tmp/base/dir/file.txt").resolve()  # pyright: ignore[reportPrivateUsage]

        # Paths with . and .. are normalized by _posix_rel
        # _posix_rel removes .. components, so ../file.txt becomes file.txt
        assert storage._abs("./file.txt") == Path("/tmp/base/file.txt").resolve()  # pyright: ignore[reportPrivateUsage]
        assert storage._abs("../file.txt") == Path("/tmp/base/file.txt").resolve()  # pyright: ignore[reportPrivateUsage]  # .. is removed
        assert storage._abs("dir/../file.txt") == Path("/tmp/base/file.txt").resolve()  # pyright: ignore[reportPrivateUsage]


class TestHelperFunctions:
    """Test helper functions for path manipulation."""

    def test_posix_rel(self):
        """Test _posix_rel path normalization."""
        from ai_pipeline_core.storage.storage import (
            _posix_rel,  # pyright: ignore[reportPrivateUsage]
        )

        # Basic cases
        assert _posix_rel("file.txt") == "file.txt"
        assert _posix_rel("dir/file.txt") == "dir/file.txt"
        assert _posix_rel("/abs/path/file.txt") == "abs/path/file.txt"

        # Windows paths
        assert _posix_rel("dir\\file.txt") == "dir/file.txt"
        assert _posix_rel("C:\\dir\\file.txt") == "C:/dir/file.txt"

        # Dots
        assert _posix_rel("./file.txt") == "file.txt"
        assert _posix_rel("dir/./file.txt") == "dir/file.txt"
        assert _posix_rel("dir/../file.txt") == "file.txt"
        assert _posix_rel("../file.txt") == "file.txt"
        assert _posix_rel("../../file.txt") == "file.txt"

        # Multiple slashes
        assert _posix_rel("dir//file.txt") == "dir/file.txt"
        assert _posix_rel("dir///file.txt") == "dir/file.txt"

        # Empty and special cases
        assert _posix_rel("") == ""
        assert _posix_rel("/") == ""
        assert _posix_rel("//") == ""
        assert _posix_rel(".") == ""
        assert _posix_rel("..") == ""

    def test_join_posix(self):
        """Test _join_posix path joining."""
        from ai_pipeline_core.storage.storage import (
            _join_posix,  # pyright: ignore[reportPrivateUsage]
        )

        # Basic joining
        assert _join_posix("dir", "file.txt") == "dir/file.txt"
        assert _join_posix("dir1", "dir2", "file.txt") == "dir1/dir2/file.txt"

        # With empty parts
        assert _join_posix("", "file.txt") == "file.txt"
        assert _join_posix("dir", "", "file.txt") == "dir/file.txt"
        assert _join_posix("", "", "file.txt") == "file.txt"

        # With dots
        assert _join_posix("dir", ".", "file.txt") == "dir/file.txt"
        # _join_posix first applies _posix_rel to each part individually
        # _posix_rel("..") returns empty string, so it's filtered out
        # Result is "dir1/file.txt"
        assert _join_posix("dir1", "..", "file.txt") == "dir1/file.txt"
        # To actually go up a directory, the .. needs to be part of a path
        assert _join_posix("dir1/..", "file.txt") == "file.txt"

        # Mixed separators
        assert _join_posix("dir1", "dir2\\subdir", "file.txt") == "dir1/dir2/subdir/file.txt"

        # Already joined paths
        assert _join_posix("dir1/dir2", "dir3/file.txt") == "dir1/dir2/dir3/file.txt"
