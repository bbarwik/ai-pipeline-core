"""Tests for ai_pipeline_core.deployment.hooks module."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from ai_pipeline_core.deployment.hooks import (
    DeploymentHook,
    DeploymentHookResult,
    load_deployment_hooks,
)


class TestDeploymentHookResult:
    """Tests for DeploymentHookResult dataclass."""

    def test_defaults_are_empty(self):
        """Default values should be empty collections."""
        result = DeploymentHookResult()
        assert result.artifacts == []
        assert result.job_variables == {}

    def test_accepts_values(self):
        """Should accept artifacts and job_variables."""
        result = DeploymentHookResult(
            artifacts=[("path/file.txt", b"content"), ("other.json", b"{}")],
            job_variables={"env": {"VAR": "value"}, "other": 123},
        )
        assert len(result.artifacts) == 2
        assert result.artifacts[0] == ("path/file.txt", b"content")
        assert result.job_variables["env"]["VAR"] == "value"


class TestDeploymentHook:
    """Tests for DeploymentHook ABC."""

    def test_is_abstract(self):
        """DeploymentHook cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            DeploymentHook()  # type: ignore

    def test_requires_name_property(self):
        """Subclass must implement name property."""

        class IncompleteHook(DeploymentHook):
            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return None

        with pytest.raises(TypeError, match="abstract"):
            IncompleteHook()  # type: ignore

    def test_requires_process_method(self):
        """Subclass must implement process method."""

        class IncompleteHook(DeploymentHook):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError, match="abstract"):
            IncompleteHook()  # type: ignore

    def test_complete_implementation_works(self):
        """A complete implementation should be instantiable."""

        class CompleteHook(DeploymentHook):
            @property
            def name(self):
                return "complete"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return DeploymentHookResult()

        hook = CompleteHook()
        assert hook.name == "complete"


class TestLoadDeploymentHooks:
    """Tests for load_deployment_hooks() function."""

    def test_empty_config_returns_empty_list(self):
        """Empty config should return empty list."""
        hooks = load_deployment_hooks({})
        assert hooks == []

    def test_no_hooks_key_returns_empty_list(self):
        """Config without hooks key should return empty list."""
        hooks = load_deployment_hooks({"tool": {"deploy": {}}})
        assert hooks == []

    def test_empty_hooks_list_returns_empty(self):
        """Empty hooks list should return empty list."""
        hooks = load_deployment_hooks({"tool": {"deploy": {"hooks": []}}})
        assert hooks == []

    def test_loads_valid_hook(self, monkeypatch):
        """Should successfully load a valid hook module."""

        class FakeHook(DeploymentHook):
            @property
            def name(self):
                return "fake_hook"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return DeploymentHookResult()

        fake_module = ModuleType("fake_hook_module")
        fake_module.get_hook = lambda: FakeHook()  # type: ignore

        monkeypatch.setitem(sys.modules, "fake_hook_module", fake_module)

        pyproject = {"tool": {"deploy": {"hooks": ["fake_hook_module"]}}}
        hooks = load_deployment_hooks(pyproject)

        assert len(hooks) == 1
        assert hooks[0].name == "fake_hook"

    def test_loads_multiple_hooks_in_order(self, monkeypatch):
        """Should load multiple hooks in specified order."""

        class HookA(DeploymentHook):
            @property
            def name(self):
                return "hook_a"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return None

        class HookB(DeploymentHook):
            @property
            def name(self):
                return "hook_b"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return None

        module_a = ModuleType("module_a")
        module_a.get_hook = lambda: HookA()  # type: ignore
        module_b = ModuleType("module_b")
        module_b.get_hook = lambda: HookB()  # type: ignore

        monkeypatch.setitem(sys.modules, "module_a", module_a)
        monkeypatch.setitem(sys.modules, "module_b", module_b)

        pyproject = {"tool": {"deploy": {"hooks": ["module_a", "module_b"]}}}
        hooks = load_deployment_hooks(pyproject)

        assert len(hooks) == 2
        assert hooks[0].name == "hook_a"
        assert hooks[1].name == "hook_b"

    def test_missing_module_raises_runtime_error(self):
        """Should raise RuntimeError for missing module."""
        pyproject = {"tool": {"deploy": {"hooks": ["nonexistent_module_xyz_123"]}}}

        with pytest.raises(RuntimeError, match="Failed to load deployment hook"):
            load_deployment_hooks(pyproject)

    def test_error_includes_module_path(self):
        """Error should include the problematic module path."""
        pyproject = {"tool": {"deploy": {"hooks": ["nonexistent_module_xyz_123"]}}}

        try:
            load_deployment_hooks(pyproject)
            pytest.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            assert "nonexistent_module_xyz_123" in str(e)

    def test_missing_get_hook_raises(self, monkeypatch):
        """Should raise for module without get_hook() function."""
        empty_module = ModuleType("empty_module")
        monkeypatch.setitem(sys.modules, "empty_module", empty_module)

        pyproject = {"tool": {"deploy": {"hooks": ["empty_module"]}}}

        with pytest.raises(RuntimeError, match="must have get_hook"):
            load_deployment_hooks(pyproject)

    def test_wrong_return_type_raises(self, monkeypatch):
        """Should raise when get_hook() returns wrong type."""
        bad_module = ModuleType("bad_module")
        bad_module.get_hook = lambda: "not a hook"  # type: ignore
        monkeypatch.setitem(sys.modules, "bad_module", bad_module)

        pyproject = {"tool": {"deploy": {"hooks": ["bad_module"]}}}

        with pytest.raises(RuntimeError, match="must return DeploymentHook"):
            load_deployment_hooks(pyproject)

    def test_exception_in_get_hook_wrapped(self, monkeypatch):
        """Exceptions in get_hook() should be wrapped in RuntimeError."""
        bad_module = ModuleType("exception_module")

        def failing_get_hook():
            raise ValueError("Hook creation failed")

        bad_module.get_hook = failing_get_hook  # type: ignore
        monkeypatch.setitem(sys.modules, "exception_module", bad_module)

        pyproject = {"tool": {"deploy": {"hooks": ["exception_module"]}}}

        with pytest.raises(RuntimeError, match="Failed to load deployment hook"):
            load_deployment_hooks(pyproject)


class TestHookIntegration:
    """Integration-style tests for hook workflow."""

    async def test_hook_can_return_none(self):
        """Hook returning None should indicate 'not applicable'."""

        class OptionalHook(DeploymentHook):
            @property
            def name(self):
                return "optional"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return None

        hook = OptionalHook()
        result = await hook.process(Path("/tmp"), {}, Path("/tmp/build"), "gs://bucket")
        assert result is None

    async def test_hook_can_return_artifacts(self):
        """Hook can return artifacts to upload."""

        class ArtifactHook(DeploymentHook):
            @property
            def name(self):
                return "artifact"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return DeploymentHookResult(
                    artifacts=[
                        ("file1.txt", b"content1"),
                        ("subdir/file2.txt", b"content2"),
                    ]
                )

        hook = ArtifactHook()
        result = await hook.process(Path("/tmp"), {}, Path("/tmp/build"), "gs://bucket")

        assert result is not None
        assert len(result.artifacts) == 2
        assert result.artifacts[0] == ("file1.txt", b"content1")

    async def test_hook_can_return_job_variables(self):
        """Hook can return job variables for deployment."""

        class EnvHook(DeploymentHook):
            @property
            def name(self):
                return "env"

            async def process(self, project_root, pyproject, build_dir, upload_uri):
                return DeploymentHookResult(
                    job_variables={
                        "env": {"MY_VAR": "my_value"},
                        "resources": {"cpu": 2},
                    }
                )

        hook = EnvHook()
        result = await hook.process(Path("/tmp"), {}, Path("/tmp/build"), "gs://bucket")

        assert result is not None
        assert result.job_variables["env"]["MY_VAR"] == "my_value"
        assert result.job_variables["resources"]["cpu"] == 2
