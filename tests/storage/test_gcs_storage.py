"""Unit tests for GcsStorage with mocked GCS dependencies."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prefect_gcp.cloud_storage import GcpCredentials

from ai_pipeline_core.storage import ObjectInfo, RetryPolicy
from ai_pipeline_core.storage.storage import GcsStorage


class TestGcsStorage:
    """Test suite for GcsStorage backend with mocked dependencies."""

    @pytest.fixture
    def mock_gcs_bucket(self):
        """Mock GcsBucket from prefect_gcp."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock:
            bucket_instance = MagicMock()
            bucket_instance.bucket = "test-bucket"
            bucket_instance.bucket_folder = "test-folder"
            bucket_instance.gcp_credentials = None
            mock.return_value = bucket_instance
            yield bucket_instance

    @pytest.fixture
    def gcs_storage(self, mock_gcs_bucket):
        """Create GcsStorage instance with mocked bucket."""
        storage = GcsStorage(
            bucket="test-bucket",
            bucket_folder="test-folder",
            gcp_credentials=None,
            retry=RetryPolicy(attempts=1),  # Single attempt for tests
        )
        storage.block = mock_gcs_bucket
        return storage

    @pytest.mark.asyncio
    async def test_init_with_credentials(self):
        """Test GcsStorage initialization with custom credentials."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket:
            mock_creds = MagicMock(spec=GcpCredentials)
            storage = GcsStorage(
                bucket="my-bucket",
                bucket_folder="my-folder",
                gcp_credentials=mock_creds,
                retry=RetryPolicy(attempts=2, base_delay=0.1),
            )

            mock_bucket.assert_called_once_with(
                bucket="my-bucket",
                bucket_folder="my-folder",
                gcp_credentials=mock_creds,
            )
            assert storage.retry.attempts == 2
            assert storage.retry.base_delay == 0.1

    @pytest.mark.asyncio
    async def test_init_loads_credentials_from_settings(self):
        """Test GcsStorage loads credentials from settings when available."""
        mock_settings = MagicMock()
        mock_settings.gcs_service_account_file = "/path/to/key.json"

        with patch("ai_pipeline_core.storage.storage.settings", mock_settings):
            with patch("ai_pipeline_core.storage.storage.Path") as mock_path:
                with patch("ai_pipeline_core.storage.storage.GcpCredentials") as mock_creds:
                    with patch("ai_pipeline_core.storage.storage.GcsBucket"):
                        mock_path.return_value = Path("/path/to/key.json")
                        mock_creds.return_value = MagicMock()

                        GcsStorage(bucket="test-bucket")

                        mock_creds.assert_called_once_with(
                            service_account_file=Path("/path/to/key.json")
                        )

    @pytest.mark.asyncio
    async def test_create_bucket(self, gcs_storage):
        """Test create_bucket method."""
        gcs_storage.block.create_bucket = AsyncMock()
        await gcs_storage.create_bucket()
        gcs_storage.block.create_bucket.assert_called_once()

    def test_url_for(self, gcs_storage):
        """Test URL generation for GCS paths."""
        assert gcs_storage.url_for("file.txt") == "gs://test-bucket/test-folder/file.txt"
        assert (
            gcs_storage.url_for("sub/dir/file.txt")
            == "gs://test-bucket/test-folder/sub/dir/file.txt"
        )
        assert gcs_storage.url_for("") == "gs://test-bucket/test-folder"

    def test_url_for_no_folder(self, mock_gcs_bucket):
        """Test URL generation without bucket folder."""
        storage = GcsStorage(bucket="test-bucket", bucket_folder="")
        storage.block = mock_gcs_bucket
        storage.block.bucket_folder = ""

        assert storage.url_for("file.txt") == "gs://test-bucket/file.txt"
        assert storage.url_for("dir/file.txt") == "gs://test-bucket/dir/file.txt"

    def test_with_base(self):
        """Test creating sub-storage with base path."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket_class:
            # Setup initial bucket
            mock_bucket = MagicMock()
            mock_bucket.bucket = "test-bucket"
            mock_bucket.bucket_folder = "test-folder"
            mock_bucket.gcp_credentials = None

            # Setup sub bucket (returned by second GcsBucket call)
            mock_sub_bucket = MagicMock()
            mock_sub_bucket.bucket = "test-bucket"
            mock_sub_bucket.bucket_folder = "test-folder/sub/path"
            mock_sub_bucket.gcp_credentials = None

            mock_bucket_class.side_effect = [mock_bucket, mock_sub_bucket]

            storage = GcsStorage(bucket="test-bucket", bucket_folder="test-folder")
            storage.block = mock_bucket

            sub_storage = storage.with_base("sub/path")

            assert isinstance(sub_storage, GcsStorage)
            # Verify the second GcsBucket was created with correct params
            assert mock_bucket_class.call_count == 2
            second_call_args = mock_bucket_class.call_args_list[1]
            assert second_call_args[1]["bucket"] == "test-bucket"
            assert second_call_args[1]["bucket_folder"] == "test-folder/sub/path"

    @pytest.mark.asyncio
    async def test_exists_blob(self, gcs_storage):
        """Test exists() for a blob that exists."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            mock_run_sync.return_value = True
            result = await gcs_storage.exists("file.txt")

        assert result is True
        mock_bucket.blob.assert_called_once_with("test-folder/file.txt")

    @pytest.mark.asyncio
    async def test_exists_prefix(self, gcs_storage):
        """Test exists() for a prefix/directory."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob

        # Mock list_blobs to return blobs with matching prefix
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-folder/dir/file1.txt"
        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)
        gcs_storage.block.list_blobs = AsyncMock(return_value=[mock_blob1])

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            mock_run_sync.return_value = False  # blob.exists returns False
            result = await gcs_storage.exists("dir")

        assert result is True
        gcs_storage.block.list_blobs.assert_called_once_with("dir")

    @pytest.mark.asyncio
    async def test_exists_not_found(self, gcs_storage):
        """Test exists() when object doesn't exist."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)
        gcs_storage.block.list_blobs = AsyncMock(return_value=[])

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            mock_run_sync.return_value = False
            result = await gcs_storage.exists("nonexistent.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_recursive(self, gcs_storage):
        """Test listing objects recursively."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-folder/file1.txt"
        mock_blob1.size = 100

        mock_blob2 = MagicMock()
        mock_blob2.name = "test-folder/dir/file2.txt"
        mock_blob2.size = 200

        gcs_storage.block.list_blobs = AsyncMock(return_value=[mock_blob1, mock_blob2])

        results = await gcs_storage.list("", recursive=True, include_dirs=False)

        assert len(results) == 2
        assert results[0] == ObjectInfo(key="file1.txt", size=100, is_dir=False)
        assert results[1] == ObjectInfo(key="dir/file2.txt", size=200, is_dir=False)

    @pytest.mark.asyncio
    async def test_list_non_recursive(self, gcs_storage):
        """Test listing objects non-recursively with directories."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-folder/file1.txt"
        mock_blob1.size = 100

        mock_blob2 = MagicMock()
        mock_blob2.name = "test-folder/dir1/file2.txt"
        mock_blob2.size = 200

        mock_blob3 = MagicMock()
        mock_blob3.name = "test-folder/dir2/sub/file3.txt"
        mock_blob3.size = 300

        gcs_storage.block.list_blobs = AsyncMock(return_value=[mock_blob1, mock_blob2, mock_blob3])

        results = await gcs_storage.list("", recursive=False, include_dirs=True)

        # Should have 1 file and 2 directories
        assert len(results) == 3

        files = [r for r in results if not r.is_dir]
        dirs = [r for r in results if r.is_dir]

        assert len(files) == 1
        assert files[0] == ObjectInfo(key="file1.txt", size=100, is_dir=False)

        assert len(dirs) == 2
        assert ObjectInfo(key="dir1", size=-1, is_dir=True) in dirs
        assert ObjectInfo(key="dir2", size=-1, is_dir=True) in dirs

    @pytest.mark.asyncio
    async def test_list_with_prefix(self, gcs_storage):
        """Test listing objects with a prefix."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-folder/subdir/file1.txt"
        mock_blob1.size = 100

        mock_blob2 = MagicMock()
        mock_blob2.name = "test-folder/subdir/nested/file2.txt"
        mock_blob2.size = 200

        gcs_storage.block.list_blobs = AsyncMock(return_value=[mock_blob1, mock_blob2])

        results = await gcs_storage.list("subdir", recursive=True, include_dirs=False)

        assert len(results) == 2
        assert results[0] == ObjectInfo(key="file1.txt", size=100, is_dir=False)
        assert results[1] == ObjectInfo(key="nested/file2.txt", size=200, is_dir=False)

    @pytest.mark.asyncio
    async def test_list_single_file(self, gcs_storage):
        """Test listing when prefix points to a single file."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.size = 500

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)
        gcs_storage.block.list_blobs = AsyncMock(return_value=[])

        mock_bucket.blob.return_value = mock_blob

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            # First call: blob.exists() returns True
            # Second call: blob.reload() succeeds
            mock_run_sync.side_effect = [True, None]

            results = await gcs_storage.list("single-file.txt", recursive=True, include_dirs=False)

        assert len(results) == 1
        assert results[0] == ObjectInfo(key="", size=500, is_dir=False)

    @pytest.mark.asyncio
    async def test_read_bytes(self, gcs_storage):
        """Test reading bytes from a file."""
        expected_data = b"test content"
        gcs_storage.block.read_path = AsyncMock(return_value=expected_data)

        data = await gcs_storage.read_bytes("file.txt")

        assert data == expected_data
        gcs_storage.block.read_path.assert_called_once_with("file.txt")

    @pytest.mark.asyncio
    async def test_read_text(self, gcs_storage):
        """Test reading text from a file."""
        text_content = "Hello, World! 🌍"
        gcs_storage.block.read_path = AsyncMock(return_value=text_content.encode("utf-8"))

        text = await gcs_storage.read_text("file.txt")

        assert text == text_content
        gcs_storage.block.read_path.assert_called_once_with("file.txt")

    @pytest.mark.asyncio
    async def test_write_bytes(self, gcs_storage):
        """Test writing bytes to a file."""
        data = b"test data"
        gcs_storage.block.write_path = AsyncMock()

        await gcs_storage.write_bytes("file.txt", data)

        gcs_storage.block.write_path.assert_called_once_with("file.txt", data)

    @pytest.mark.asyncio
    async def test_write_text(self, gcs_storage):
        """Test writing text to a file."""
        text = "Test text with unicode: 🚀"
        gcs_storage.block.write_path = AsyncMock()

        await gcs_storage.write_text("file.txt", text)

        gcs_storage.block.write_path.assert_called_once_with("file.txt", text.encode("utf-8"))

    @pytest.mark.asyncio
    async def test_delete_single_blob(self, gcs_storage):
        """Test deleting a single blob."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            # blob.delete() succeeds
            mock_run_sync.return_value = None

            await gcs_storage.delete("file.txt")

        mock_bucket.blob.assert_called_once_with("test-folder/file.txt")
        mock_run_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_prefix(self, gcs_storage):
        """Test deleting multiple blobs with a prefix."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_blob1 = MagicMock()
        mock_blob1.name = "test-folder/dir/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = "test-folder/dir/file2.txt"

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)
        gcs_storage.block.list_blobs = AsyncMock(return_value=[mock_blob1, mock_blob2])

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            # First call to delete exact blob fails (returns False)
            # Then calls to delete each blob from list succeed
            mock_run_sync.side_effect = [Exception("Not found"), None, None]

            await gcs_storage.delete("dir")

        # Should have tried to delete each blob
        assert mock_run_sync.call_count == 3

    @pytest.mark.asyncio
    async def test_delete_missing_ok_true(self, gcs_storage):
        """Test deleting non-existent file with missing_ok=True."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)
        gcs_storage.block.list_blobs = AsyncMock(return_value=[])

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            mock_run_sync.side_effect = Exception("Not found")

            # Should not raise
            await gcs_storage.delete("nonexistent.txt", missing_ok=True)

    @pytest.mark.asyncio
    async def test_delete_missing_ok_false(self, gcs_storage):
        """Test deleting non-existent file with missing_ok=False."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        gcs_storage.block.get_bucket = AsyncMock(return_value=mock_bucket)
        gcs_storage.block.list_blobs = AsyncMock(return_value=[])

        with patch("ai_pipeline_core.storage.storage.run_sync_in_worker_thread") as mock_run_sync:
            mock_run_sync.side_effect = Exception("Not found")

            with pytest.raises(FileNotFoundError):
                await gcs_storage.delete("nonexistent.txt", missing_ok=False)

    @pytest.mark.asyncio
    async def test_copy_from(self, gcs_storage):
        """Test copying from another storage."""
        # Create mock source storage
        source_storage = MagicMock()
        source_storage.list = AsyncMock(
            return_value=[
                ObjectInfo(key="file1.txt", size=100, is_dir=False),
                ObjectInfo(key="dir/file2.txt", size=200, is_dir=False),
            ]
        )
        source_storage.read_bytes = AsyncMock(side_effect=[b"content1", b"content2"])

        gcs_storage.block.write_path = AsyncMock()

        await gcs_storage.copy_from(source_storage, src_prefix="src", dst_prefix="dst")

        # Check that files were read from source
        assert source_storage.read_bytes.call_count == 2
        source_storage.read_bytes.assert_any_call("src/file1.txt")
        source_storage.read_bytes.assert_any_call("src/dir/file2.txt")

        # Check that files were written to destination
        assert gcs_storage.block.write_path.call_count == 2
        gcs_storage.block.write_path.assert_any_call("dst/file1.txt", b"content1")
        gcs_storage.block.write_path.assert_any_call("dst/dir/file2.txt", b"content2")

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry mechanism on failures."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.bucket = "test-bucket"
            mock_bucket.bucket_folder = ""
            mock_bucket.gcp_credentials = None
            mock_bucket_class.return_value = mock_bucket

            storage = GcsStorage(
                bucket="test-bucket",
                retry=RetryPolicy(attempts=3, base_delay=0.01, max_delay=0.02),
            )
            storage.block = mock_bucket

            # Set up read_path to fail twice then succeed
            call_count = 0

            async def failing_read_path(path):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception(f"Attempt {call_count} failed")
                return b"success"

            storage.block.read_path = failing_read_path

            result = await storage.read_bytes("file.txt")

            assert result == b"success"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test that retry gives up after max attempts."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.bucket = "test-bucket"
            mock_bucket.bucket_folder = ""
            mock_bucket.gcp_credentials = None
            mock_bucket_class.return_value = mock_bucket

            storage = GcsStorage(
                bucket="test-bucket",
                retry=RetryPolicy(attempts=2, base_delay=0.01),
            )
            storage.block = mock_bucket

            # Set up read_path to always fail
            async def always_failing_read_path(path):
                raise Exception("Always fails")

            storage.block.read_path = always_failing_read_path

            with pytest.raises(Exception, match="Always fails"):
                await storage.read_bytes("file.txt")

    @pytest.mark.asyncio
    async def test_retry_with_custom_exceptions(self):
        """Test retry with specific exception types."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.bucket = "test-bucket"
            mock_bucket.bucket_folder = ""
            mock_bucket.gcp_credentials = None
            mock_bucket_class.return_value = mock_bucket

            storage = GcsStorage(
                bucket="test-bucket",
                retry=RetryPolicy(
                    attempts=2,
                    base_delay=0.01,
                    retry_exceptions=(ValueError,),
                ),
            )
            storage.block = mock_bucket

            # Should retry on ValueError
            call_count = 0

            async def value_error_then_success(path):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ValueError("Retry this")
                return b"success"

            storage.block.read_path = value_error_then_success
            result = await storage.read_bytes("file.txt")
            assert result == b"success"
            assert call_count == 2

            # Should NOT retry on TypeError
            async def type_error(path):
                raise TypeError("Don't retry this")

            storage.block.read_path = type_error

            with pytest.raises(TypeError, match="Don't retry this"):
                await storage.read_bytes("file.txt")

    @pytest.mark.asyncio
    async def test_retry_respects_cancellation(self):
        """Test that retry respects asyncio cancellation."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.bucket = "test-bucket"
            mock_bucket.bucket_folder = ""
            mock_bucket.gcp_credentials = None
            mock_bucket_class.return_value = mock_bucket

            storage = GcsStorage(
                bucket="test-bucket",
                retry=RetryPolicy(attempts=3, base_delay=1.0),  # Long delay
            )
            storage.block = mock_bucket

            # Set up read_path to always fail
            async def always_failing(path):
                raise Exception("Fail")

            storage.block.read_path = always_failing

            # Create a task and cancel it during retry
            task = asyncio.create_task(storage.read_bytes("file.txt"))
            await asyncio.sleep(0.01)  # Let it start
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

    def test_abs_name_handling(self, gcs_storage):
        """Test internal path handling methods."""
        # Test _abs_name
        assert gcs_storage._abs_name("file.txt") == "test-folder/file.txt"  # pyright: ignore[reportPrivateUsage]
        assert gcs_storage._abs_name("dir/file.txt") == "test-folder/dir/file.txt"  # pyright: ignore[reportPrivateUsage]
        assert gcs_storage._abs_name("") == "test-folder"  # pyright: ignore[reportPrivateUsage]

        # Test _rel_from_abs
        assert gcs_storage._rel_from_abs("test-folder/file.txt") == "file.txt"  # pyright: ignore[reportPrivateUsage]
        assert gcs_storage._rel_from_abs("test-folder/dir/file.txt") == "dir/file.txt"  # pyright: ignore[reportPrivateUsage]
        assert gcs_storage._rel_from_abs("other-folder/file.txt") == "other-folder/file.txt"  # pyright: ignore[reportPrivateUsage]

    def test_path_normalization(self):
        """Test that paths are properly normalized."""
        with patch("ai_pipeline_core.storage.storage.GcsBucket") as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.bucket = "test-bucket"
            mock_bucket.bucket_folder = "base/folder"  # Normalized version
            mock_bucket.gcp_credentials = None
            mock_bucket_class.return_value = mock_bucket

            storage = GcsStorage(bucket="test-bucket", bucket_folder="base//folder/")
            storage.block = mock_bucket
            storage.block.bucket_folder = "base/folder"  # Set the normalized version

            # Should normalize the folder path
            assert storage.url_for("file.txt") == "gs://test-bucket/base/folder/file.txt"
            assert (
                storage.url_for("../../../etc/passwd") == "gs://test-bucket/base/folder/etc/passwd"
            )
            assert (
                storage.url_for("./subdir/../file.txt") == "gs://test-bucket/base/folder/file.txt"
            )
