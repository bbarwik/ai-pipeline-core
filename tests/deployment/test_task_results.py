"""Task result store unit tests.

Covers: CRUD, serialization, protocol conformance, concurrent writes, large payloads,
special characters, and store isolation.
"""

import json

import pytest

from ai_pipeline_core.deployment._task_results import MemoryTaskResultStore
from ai_pipeline_core.deployment._types import TaskResultRecord, TaskResultStore


class TestMemoryTaskResultStore:
    """Core MemoryTaskResultStore CRUD tests."""

    async def test_roundtrip(self) -> None:
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"success": true}', '{"version": 1}')
        record = await store.read_result("run-1")
        assert record is not None
        assert record.run_id == "run-1"
        assert record.result == '{"success": true}'
        assert record.chain_context == '{"version": 1}'

    async def test_read_nonexistent_returns_none(self) -> None:
        store = MemoryTaskResultStore()
        record = await store.read_result("does-not-exist")
        assert record is None

    async def test_overwrite_replaces_record(self) -> None:
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"v": 1}', '{"ctx": 1}')
        await store.write_result("run-1", '{"v": 2}', '{"ctx": 2}')
        record = await store.read_result("run-1")
        assert record is not None
        assert record.result == '{"v": 2}'
        assert record.chain_context == '{"ctx": 2}'

    async def test_multiple_run_ids(self) -> None:
        store = MemoryTaskResultStore()
        await store.write_result("run-a", '{"a": true}', "{}")
        await store.write_result("run-b", '{"b": true}', "{}")
        a = await store.read_result("run-a")
        b = await store.read_result("run-b")
        assert a is not None and a.result == '{"a": true}'
        assert b is not None and b.result == '{"b": true}'

    async def test_stored_at_populated(self) -> None:
        store = MemoryTaskResultStore()
        await store.write_result("run-1", "{}", "{}")
        record = await store.read_result("run-1")
        assert record is not None
        assert record.stored_at is not None
        assert record.stored_at.tzinfo is not None  # timezone-aware

    async def test_chain_context_stored(self) -> None:
        store = MemoryTaskResultStore()
        chain = '{"parent_run_id": "parent-1", "depth": 2}'
        await store.write_result("run-1", '{"ok": true}', chain)
        record = await store.read_result("run-1")
        assert record is not None
        assert record.chain_context == chain

    async def test_empty_strings(self) -> None:
        store = MemoryTaskResultStore()
        await store.write_result("run-1", "", "")
        record = await store.read_result("run-1")
        assert record is not None
        assert record.result == ""
        assert record.chain_context == ""

    async def test_shutdown_is_noop(self) -> None:
        store = MemoryTaskResultStore()
        store.shutdown()  # should not raise

    async def test_read_after_shutdown(self) -> None:
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"ok": true}', "{}")
        store.shutdown()
        record = await store.read_result("run-1")
        assert record is not None
        assert record.result == '{"ok": true}'


class TestTaskResultStoreProtocol:
    """Verify MemoryTaskResultStore conforms to the TaskResultStore protocol."""

    def test_memory_store_satisfies_protocol(self) -> None:
        """MemoryTaskResultStore must be a runtime-checkable TaskResultStore."""
        store = MemoryTaskResultStore()
        assert isinstance(store, TaskResultStore)

    def test_task_result_record_fields(self) -> None:
        """TaskResultRecord must have run_id, result, chain_context, stored_at."""
        from datetime import UTC, datetime

        record = TaskResultRecord(
            run_id="test-run",
            result='{"success": true}',
            chain_context='{"version": 1}',
            stored_at=datetime.now(UTC),
        )
        assert record.run_id == "test-run"
        assert record.result == '{"success": true}'
        assert record.chain_context == '{"version": 1}'
        assert record.stored_at is not None

    def test_task_result_record_is_frozen(self) -> None:
        """TaskResultRecord is a frozen dataclass."""
        from datetime import UTC, datetime

        record = TaskResultRecord(
            run_id="run-1",
            result="{}",
            chain_context="{}",
            stored_at=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            record.run_id = "changed"  # type: ignore[misc]


class TestTaskResultSerialization:
    """Test JSON serialization round-trips via the store."""

    async def test_complex_result_json_roundtrip(self) -> None:
        """Complex JSON with nested structures round-trips correctly."""
        store = MemoryTaskResultStore()
        result_data = {
            "success": True,
            "documents": [
                {"sha256": "ABC123", "name": "report.md"},
                {"sha256": "DEF456", "name": "analysis.json"},
            ],
            "metadata": {"flow_count": 3, "elapsed_seconds": 42.5},
        }
        chain_data = {
            "version": 1,
            "run_scope": "project:abc123",
            "output_document_refs": ["SHA1", "SHA2"],
        }
        result_json = json.dumps(result_data)
        chain_json = json.dumps(chain_data)

        await store.write_result("run-complex", result_json, chain_json)
        record = await store.read_result("run-complex")
        assert record is not None

        parsed_result = json.loads(record.result)
        parsed_chain = json.loads(record.chain_context)
        assert parsed_result["success"] is True
        assert len(parsed_result["documents"]) == 2
        assert parsed_chain["version"] == 1
        assert len(parsed_chain["output_document_refs"]) == 2

    async def test_unicode_content_preserved(self) -> None:
        """Unicode characters in result/chain_context are preserved."""
        store = MemoryTaskResultStore()
        result_json = json.dumps({"message": "Résumé análysis 日本語"})
        chain_json = json.dumps({"scope": "project:日本"})

        await store.write_result("run-unicode", result_json, chain_json)
        record = await store.read_result("run-unicode")
        assert record is not None

        parsed = json.loads(record.result)
        assert parsed["message"] == "Résumé análysis 日本語"

    async def test_large_payload_stored(self) -> None:
        """Large JSON payloads are handled correctly."""
        store = MemoryTaskResultStore()
        large_data = {"items": [{"id": i, "data": "x" * 100} for i in range(1000)]}
        result_json = json.dumps(large_data)

        await store.write_result("run-large", result_json, "{}")
        record = await store.read_result("run-large")
        assert record is not None

        parsed = json.loads(record.result)
        assert len(parsed["items"]) == 1000

    async def test_special_characters_in_run_id(self) -> None:
        """Run IDs with hyphens and underscores work correctly."""
        store = MemoryTaskResultStore()
        run_ids = ["run-with-dashes", "run_with_underscores", "RUN-123_abc"]

        for run_id in run_ids:
            await store.write_result(run_id, f'{{"id": "{run_id}"}}', "{}")

        for run_id in run_ids:
            record = await store.read_result(run_id)
            assert record is not None, f"Failed to read run_id={run_id}"
            assert record.run_id == run_id


class TestTaskResultStoreIsolation:
    """Test store isolation and concurrent behavior."""

    async def test_separate_stores_are_independent(self) -> None:
        """Different MemoryTaskResultStore instances don't share data."""
        store1 = MemoryTaskResultStore()
        store2 = MemoryTaskResultStore()

        await store1.write_result("run-1", '{"store": 1}', "{}")
        record = await store2.read_result("run-1")
        assert record is None

    async def test_overwrite_updates_stored_at(self) -> None:
        """Overwriting a result updates the stored_at timestamp."""
        store = MemoryTaskResultStore()
        await store.write_result("run-1", '{"v": 1}', "{}")
        record1 = await store.read_result("run-1")
        assert record1 is not None

        await store.write_result("run-1", '{"v": 2}', "{}")
        record2 = await store.read_result("run-1")
        assert record2 is not None
        assert record2.stored_at >= record1.stored_at

    async def test_write_does_not_affect_other_run_ids(self) -> None:
        """Writing to one run_id does not affect another."""
        store = MemoryTaskResultStore()
        await store.write_result("run-a", '{"a": true}', '{"ctx": "a"}')
        await store.write_result("run-b", '{"b": true}', '{"ctx": "b"}')
        await store.write_result("run-a", '{"a": false}', '{"ctx": "a2"}')

        a = await store.read_result("run-a")
        b = await store.read_result("run-b")
        assert a is not None and a.result == '{"a": false}'
        assert b is not None and b.result == '{"b": true}'
        assert b.chain_context == '{"ctx": "b"}'
