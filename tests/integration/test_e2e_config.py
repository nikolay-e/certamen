"""End-to-end tests for configuration and model management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from arbitrium_core import Arbitrium
from arbitrium_core.application.bootstrap import health_check_models
from arbitrium_core.domain.errors import ConfigurationError
from arbitrium_core.infrastructure.config.loader import validate_config
from tests.integration.conftest import MockModel


class TestConfigurationLoading:
    """Test configuration loading from YAML files."""

    @pytest.mark.asyncio
    async def test_load_config_from_file(
        self,
        basic_config: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test loading configuration from a YAML file."""
        # Create temporary config file
        config_file = tmp_output_dir / "test_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(basic_config, f)

        # Load config
        arbitrium = await Arbitrium.from_config(
            config_path=config_file,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Verify config loaded
        assert arbitrium.config_data is not None
        assert "models" in arbitrium.config_data
        assert "retry" in arbitrium.config_data
        assert "features" in arbitrium.config_data

    @pytest.mark.asyncio
    async def test_load_config_from_settings_dict(
        self,
        basic_config: dict,
    ) -> None:
        """Test loading configuration from settings dictionary."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Verify config loaded
        assert arbitrium.config_data is not None
        assert arbitrium.config_data["models"] == basic_config["models"]

    @pytest.mark.asyncio
    async def test_invalid_config_file_raises_error(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that invalid config file raises ConfigurationError."""
        # Create invalid config file (model missing required provider field)
        invalid_config = {
            "models": {"test_model": {"display_name": "Test"}},
            # Missing: provider and model_name in model config
        }

        config_file = tmp_output_dir / "invalid_config.yml"
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError):
            await Arbitrium.from_config(
                config_path=config_file,
                skip_secrets=True,
            )

    @pytest.mark.asyncio
    async def test_missing_config_file_raises_error(
        self,
    ) -> None:
        """Test that missing config file raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            await Arbitrium.from_config(
                config_path="/nonexistent/config.yml",
                skip_secrets=True,
            )


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_valid_config_passes_validation(
        self,
        basic_config: dict,
    ) -> None:
        """Test that valid configuration passes validation."""
        is_valid, errors = validate_config(basic_config)

        assert is_valid
        assert len(errors) == 0

    def test_minimal_config_with_defaults_passes_validation(
        self,
    ) -> None:
        """Test that minimal config with pydantic defaults passes validation."""
        minimal_config = {
            "models": {"test": {"model_name": "test", "provider": "mock"}},
            # retry, features, prompts, outputs_dir all have defaults
        }

        is_valid, errors = validate_config(minimal_config)

        assert is_valid
        assert len(errors) == 0

    def test_empty_models_fails_validation(
        self,
    ) -> None:
        """Test that empty models section fails validation."""
        invalid_config = {
            "models": {},  # Empty but present
            "retry": {"max_attempts": 3},
            "features": {"save_reports_to_disk": False},
            "prompts": {"initial": "test"},
            "outputs_dir": "/tmp/test",
        }

        is_valid, errors = validate_config(invalid_config)

        assert not is_valid
        assert any("models" in error.lower() for error in errors)

    def test_wrong_type_fails_validation(
        self,
    ) -> None:
        """Test that wrong type for section fails validation."""
        invalid_config = {
            "models": {"test": {"model_name": "test", "provider": "mock"}},
            "retry": "not_a_dict",  # Should be dict
            "features": {},
            "prompts": {},
            "outputs_dir": "/tmp/test",
        }

        is_valid, errors = validate_config(invalid_config)

        assert not is_valid
        assert any("retry" in error.lower() for error in errors)


class TestConfigurationMerging:
    """Test configuration merging with defaults."""

    @pytest.mark.asyncio
    async def test_user_config_overrides_defaults(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that user config values override defaults."""
        user_config = {
            "models": {
                "model_a": {
                    "provider": "mock",
                    "model_name": "test-a",
                    "temperature": 0.9,  # Override default
                    "context_window": 4000,
                    "max_tokens": 1000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        arbitrium = await Arbitrium.from_settings(
            settings=user_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # User value should override default
        model_config = arbitrium.config.get_model_config("model_a")
        assert model_config["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_defaults_fill_missing_values(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that defaults fill in missing config values."""
        minimal_config = {
            "models": {
                "model_a": {
                    "provider": "mock",
                    "model_name": "test-a",
                    "context_window": 4000,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Should have default retry settings
        assert "retry" in arbitrium.config_data
        assert "max_attempts" in arbitrium.config_data["retry"]

        # Should have default features
        assert "features" in arbitrium.config_data

        # Should have default prompts
        assert "prompts" in arbitrium.config_data

    @pytest.mark.asyncio
    async def test_only_specified_models_are_loaded(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that only models mentioned in config are loaded."""
        # Config with specific models
        config = {
            "models": {
                "model_a": {
                    "provider": "mock",
                    "model_name": "test-a",
                    "context_window": 4000,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
                "model_b": {
                    "provider": "mock",
                    "model_name": "test-b",
                    "context_window": 4000,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        arbitrium = await Arbitrium.from_settings(
            settings=config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Should have exactly the specified models
        model_keys = arbitrium.config.get_active_model_keys()
        assert len(model_keys) == 2
        assert "model_a" in model_keys
        assert "model_b" in model_keys


class TestModelInitialization:
    """Test model initialization and management."""

    @pytest.mark.asyncio
    async def test_models_are_created_from_config(
        self,
        basic_config: dict,
    ) -> None:
        """Test that models are created from configuration."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Should have models
        assert len(arbitrium.all_models) > 0
        assert len(arbitrium.all_models) == len(basic_config["models"])

    @pytest.mark.asyncio
    async def test_empty_models_config_creates_no_models(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that empty models config creates no models."""
        empty_config = {
            "models": {},
            "retry": {"max_attempts": 3},
            "features": {},
            "prompts": {"initial": "test"},
            "outputs_dir": str(tmp_output_dir),
        }

        # Should fail validation since models is empty
        with pytest.raises(ConfigurationError):
            await Arbitrium.from_settings(
                settings=empty_config,
                skip_secrets=True,
            )

    @pytest.mark.asyncio
    async def test_model_properties_are_set_correctly(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that model properties are set from config."""
        config = {
            "models": {
                "test_model": {
                    "provider": "mock",
                    "model_name": "test-123",
                    "display_name": "Test Model 123",
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "context_window": 8000,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        arbitrium = await Arbitrium.from_settings(
            settings=config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Get the model
        model = arbitrium.all_models["test_model"]

        # Verify properties
        assert model.model_name == "test-123"
        assert model.display_name == "Test Model 123"
        assert model.temperature == 0.8
        assert model.max_tokens == 2000


class TestHealthChecks:
    """Test model health checking."""

    @pytest.mark.asyncio
    async def test_healthy_models_pass_health_check(
        self,
        basic_config: dict,
    ) -> None:
        """Test that healthy mock models pass health check."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=False,  # Enable health check
        )

        # Mock models should pass
        assert len(arbitrium.healthy_models) > 0
        assert arbitrium.is_ready

    @pytest.mark.asyncio
    async def test_failed_models_tracked_separately(
        self,
        basic_config: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test that failed models are tracked separately."""
        # Create arbitrium without health check first
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Manually add a failing model

        failing_model = MockModel(
            model_name="failing",
            display_name="Failing Model",
            should_fail=True,
        )

        # Perform health check manually
        test_models = {
            "healthy": arbitrium.all_models["model_a"],
            "failing": failing_model,
        }  # type: ignore[index]

        healthy, failed = await health_check_models(test_models)

        # Should have one healthy, one failed
        assert len(healthy) == 1
        assert len(failed) == 1
        assert "healthy" in healthy
        assert "failing" in failed

    @pytest.mark.asyncio
    async def test_skip_health_check_flag_works(
        self,
        basic_config: dict,
    ) -> None:
        """Test that skip_health_check flag works."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,  # Skip health check
        )

        # All models should be in healthy_models (no filtering)
        assert len(arbitrium.healthy_models) == len(arbitrium.all_models)
        assert len(arbitrium.failed_models) == 0

    @pytest.mark.asyncio
    async def test_is_ready_property(
        self,
        basic_config: dict,
    ) -> None:
        """Test is_ready property reflects model availability."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Should be ready with healthy models
        assert arbitrium.is_ready
        assert arbitrium.healthy_model_count > 0


class TestSecretsLoading:
    """Test secrets loading behavior."""

    @pytest.mark.asyncio
    async def test_skip_secrets_flag_works(
        self,
        basic_config: dict,
    ) -> None:
        """Test that skip_secrets flag prevents secret loading."""
        # Should not raise error even without secrets
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        assert arbitrium is not None
        assert arbitrium.is_ready

    @pytest.mark.asyncio
    async def test_local_providers_skip_secrets(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that local providers skip secret loading."""
        config = {
            "models": {
                "ollama_model": {
                    "provider": "ollama",
                    "model_name": "llama2",
                    "context_window": 4096,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        # Should not attempt to load secrets for local providers
        arbitrium = await Arbitrium.from_settings(
            settings=config,
            skip_secrets=False,  # Secrets loading enabled
            skip_health_check=True,
        )

        assert arbitrium is not None


class TestOutputsDirectory:
    """Test outputs directory handling."""

    @pytest.mark.asyncio
    async def test_outputs_dir_is_optional_with_default(
        self,
    ) -> None:
        """Test that outputs_dir is optional and defaults to None."""
        config_without_outputs = {
            "models": {
                "test": {
                    "provider": "mock",
                    "model_name": "test",
                },
            },
            # No outputs_dir specified - should use default
        }

        # Should pass validation with defaults
        is_valid, errors = validate_config(config_without_outputs)
        assert is_valid
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_outputs_dir_can_be_relative_or_absolute(
        self,
        basic_config: dict,
    ) -> None:
        """Test that outputs_dir accepts both relative and absolute paths."""
        # Test relative path
        basic_config["outputs_dir"] = "./test_output"
        arbitrium1 = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )
        assert arbitrium1 is not None

        # Test absolute path
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_config["outputs_dir"] = tmpdir
            arbitrium2 = await Arbitrium.from_settings(
                settings=basic_config,
                skip_secrets=True,
                skip_health_check=True,
            )
            assert arbitrium2 is not None

    @pytest.mark.asyncio
    async def test_outputs_dir_is_accessible_from_arbitrium(
        self,
        basic_config: dict,
    ) -> None:
        """Test that outputs_dir is accessible from Arbitrium instance."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Should be accessible from config
        assert "outputs_dir" in arbitrium.config_data
        assert arbitrium.config_data["outputs_dir"] is not None


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""

    @pytest.mark.asyncio
    async def test_malformed_yaml_raises_error(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that malformed YAML raises appropriate error."""
        # Create malformed YAML file
        config_file = tmp_output_dir / "malformed.yml"
        with open(config_file, "w") as f:
            f.write("models:\n  test: {invalid yaml syntax")

        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError):
            await Arbitrium.from_config(
                config_path=config_file,
                skip_secrets=True,
            )

    @pytest.mark.asyncio
    async def test_missing_model_required_fields(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that models with missing required fields fail validation."""
        config = {
            "models": {
                "incomplete_model": {
                    # Missing provider and model_name
                    "display_name": "Test",
                },
            },
            "retry": {"max_attempts": 3},
            "features": {},
            "prompts": {"initial": "test"},
            "outputs_dir": str(tmp_output_dir),
        }

        # Should fail validation
        is_valid, errors = validate_config(config)
        assert not is_valid
        assert any("model_name" in error.lower() for error in errors)
        assert any("provider" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_config_deep_merge_preserves_nested_values(
        self,
        tmp_output_dir: Path,
    ) -> None:
        """Test that deep merge preserves nested configuration values."""
        config = {
            "models": {
                "test": {
                    "provider": "mock",
                    "model_name": "test",
                    "context_window": 4000,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            },
            "retry": {
                "max_attempts": 5,  # Override default
                "initial_delay": 10,  # Override default
                # max_delay should come from defaults
            },
            "outputs_dir": str(tmp_output_dir),
        }

        arbitrium = await Arbitrium.from_settings(
            settings=config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Check merged retry config
        retry_config = arbitrium.config_data["retry"]
        assert retry_config["max_attempts"] == 5  # User value
        assert retry_config["initial_delay"] == 10  # User value
        assert "max_delay" in retry_config  # From defaults
