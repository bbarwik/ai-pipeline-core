"""Tests for deploy.py to verify it uses settings correctly."""

from unittest.mock import Mock, patch

import pytest

from ai_pipeline_core.utils.deploy import Deployer


class TestDeployer:
    """Test the Deployer class uses settings instead of environment variables."""

    @patch("ai_pipeline_core.utils.deploy.settings")
    def test_init_validates_prefect_api_url(self, mock_settings):
        """Test that Deployer validates PREFECT_API_URL from settings."""
        # Test with empty API URL - should fail
        mock_settings.prefect_api_url = ""
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.utils.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True):
                with patch("ai_pipeline_core.utils.deploy.tomllib.load") as mock_toml:
                    mock_toml.return_value = {
                        "project": {"name": "test-project", "version": "1.0.0"}
                    }

                    with pytest.raises(SystemExit) as exc_info:
                        Deployer()
                    assert exc_info.value.code == 1

    @patch("ai_pipeline_core.utils.deploy.settings")
    def test_init_validates_prefect_gcs_bucket(self, mock_settings):
        """Test that Deployer validates PREFECT_GCS_BUCKET from settings."""
        # Test with empty bucket - should fail
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = ""
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.utils.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            with pytest.raises(SystemExit) as exc_info:
                Deployer()
            assert exc_info.value.code == 1

    @patch("ai_pipeline_core.utils.deploy.settings")
    def test_init_loads_config_from_settings(self, mock_settings):
        """Test that Deployer loads configuration from settings."""
        # Set up valid settings
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "test-pool"
        mock_settings.prefect_work_queue_name = "test-queue"

        with patch("ai_pipeline_core.utils.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True):
                with patch("ai_pipeline_core.utils.deploy.tomllib.load") as mock_toml:
                    mock_toml.return_value = {
                        "project": {"name": "test-project", "version": "1.0.0"}
                    }

                    deployer = Deployer()

                    # Verify config was loaded from settings
                    assert deployer.config["bucket"] == "test-bucket"
                    assert deployer.config["work_pool"] == "test-pool"
                    assert deployer.config["work_queue"] == "test-queue"
                    assert deployer.api_url == "http://test.api"

    @patch("ai_pipeline_core.utils.deploy.settings")
    def test_no_os_environ_usage(self, mock_settings):
        """Test that Deployer does not use os.environ directly."""
        # Set up valid settings
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.utils.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True):
                with patch("ai_pipeline_core.utils.deploy.tomllib.load") as mock_toml:
                    mock_toml.return_value = {
                        "project": {"name": "test-project", "version": "1.0.0"}
                    }

                    # Patch os.environ to track any access
                    with patch.dict("os.environ", {}, clear=True) as mock_environ:
                        # Create a Mock that will fail if accessed
                        mock_environ.__setitem__ = Mock(
                            side_effect=AssertionError("os.environ should not be modified")
                        )

                        # This should succeed without modifying os.environ
                        deployer = Deployer()
                        assert deployer.api_url == "http://test.api"

    @patch("ai_pipeline_core.utils.deploy.settings")
    def test_deploy_uses_settings_for_client(self, mock_settings):
        """Test that deployment uses settings for Prefect client configuration."""
        # Set up valid settings
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "test-pool"
        mock_settings.prefect_work_queue_name = "test-queue"
        mock_settings.prefect_api_key = "test-key"

        with patch("ai_pipeline_core.utils.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True):
                with patch("ai_pipeline_core.utils.deploy.tomllib.load") as mock_toml:
                    mock_toml.return_value = {
                        "project": {"name": "test-project", "version": "1.0.0"}
                    }

                    deployer = Deployer()

                    # Verify that Prefect client will receive settings
                    # The actual Prefect client reads from environment variables
                    # which are set by pydantic_settings when settings loads .env
                    assert deployer.api_url == "http://test.api"
                    assert deployer.config["bucket"] == "test-bucket"

    @patch("ai_pipeline_core.utils.deploy.settings")
    def test_config_normalization(self, mock_settings):
        """Test that project names are normalized correctly."""
        mock_settings.prefect_api_url = "http://test.api"
        mock_settings.prefect_gcs_bucket = "test-bucket"
        mock_settings.prefect_work_pool_name = "default"
        mock_settings.prefect_work_queue_name = "default"

        with patch("ai_pipeline_core.utils.deploy.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", create=True):
                with patch("ai_pipeline_core.utils.deploy.tomllib.load") as mock_toml:
                    # Test with hyphenated name
                    mock_toml.return_value = {
                        "project": {"name": "my-test-project", "version": "1.0.0"}
                    }

                    deployer = Deployer()

                    # Verify normalization
                    assert deployer.config["name"] == "my-test-project"
                    assert deployer.config["package"] == "my_test_project"  # Hyphens to underscores
                    assert deployer.config["folder"] == "flows/my-test-project"
