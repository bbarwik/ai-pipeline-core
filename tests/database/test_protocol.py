"""Tests verifying MemoryDatabase satisfies both protocol contracts."""

from pathlib import Path

from ai_pipeline_core.database import DatabaseReader, DatabaseWriter, MemoryDatabase
from ai_pipeline_core.database._protocol import _DocumentBlobReader


class TestProtocolConformance:
    def test_memory_database_is_writer(self) -> None:
        db = MemoryDatabase()
        assert isinstance(db, DatabaseWriter)

    def test_memory_database_is_reader(self) -> None:
        db = MemoryDatabase()
        assert isinstance(db, DatabaseReader)

    def test_writer_protocol_methods_exist(self) -> None:
        expected = {
            "insert_node",
            "update_node",
            "save_document",
            "save_document_batch",
            "save_blob",
            "save_blob_batch",
            "save_logs_batch",
            "update_document_summary",
            "flush",
            "shutdown",
        }
        db = MemoryDatabase()
        for method_name in expected:
            assert hasattr(db, method_name), f"Missing writer method: {method_name}"
            assert callable(getattr(db, method_name))

    def test_writer_protocol_supports_remote_exists(self) -> None:
        db = MemoryDatabase()
        assert hasattr(db, "supports_remote")
        assert db.supports_remote is False

    def test_reader_protocol_methods_exist(self) -> None:
        expected = {
            "get_node",
            "get_children",
            "get_deployment_tree",
            "get_deployment_by_run_id",
            "get_deployment_by_run_scope",
            "get_document",
            "find_document_by_name",
            "get_documents_batch",
            "get_blob",
            "get_blobs_batch",
            "get_documents_by_deployment",
            "get_documents_by_node",
            "get_all_document_shas_for_deployment",
            "check_existing_documents",
            "find_documents_by_source",
            "get_document_ancestry",
            "find_documents_by_origin",
            "list_run_scopes",
            "search_documents",
            "get_deployment_cost_totals",
            "get_documents_by_run_scope",
            "list_deployments",
            "get_cached_completion",
            "get_node_logs",
            "get_deployment_logs",
        }
        db = MemoryDatabase()
        for method_name in expected:
            assert hasattr(db, method_name), f"Missing reader method: {method_name}"
            assert callable(getattr(db, method_name))

    def test_get_documents_by_deployment_contract_mentions_deployment_chain(self) -> None:
        method_doc = _DocumentBlobReader.get_documents_by_deployment.__doc__
        assert method_doc is not None
        assert "deployment chain" in method_doc

        readme_path = Path(__file__).resolve().parents[2] / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")
        assert "`get_documents_by_deployment(deployment_id)` — Load documents for a deployment chain" in readme_text
