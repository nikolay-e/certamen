"""Configuration handling for Arbitrium Framework."""

from pathlib import Path
from typing import Any

import yaml

from arbitrium.logging import get_contextual_logger

# Module-level logger
logger = get_contextual_logger("arbitrium.config")


def validate_config(config_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validates the configuration data.

    Framework-level validation - no default values allowed.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: list[str] = []
    schema = {
        "models": {"required": True, "type": dict, "non_empty": True},
        "retry": {"required": True, "type": dict},
        "features": {"required": True, "type": dict},
        "prompts": {"required": True, "type": dict},
        "outputs_dir": {"required": True, "type": str, "non_empty": True},
    }

    logger.debug(f"Validating config sections: {list(config_data.keys())}")

    for section, rules in schema.items():
        logger.debug(f"Checking section '{section}': required={rules['required']}, present={section in config_data}")

        if rules["required"] and section not in config_data:
            error_msg = f"Missing required section '{section}'"
            logger.error(error_msg)
            errors.append(error_msg)
            continue  # Skip further checks for this section

        if rules.get("non_empty") and not config_data.get(section):
            error_msg = f"Section '{section}' is empty but must contain values"
            logger.error(error_msg)
            errors.append(error_msg)
            continue

        if "type" in rules:
            expected_type = rules["type"]
            section_value = config_data.get(section)
            expected_type_name = getattr(expected_type, "__name__", str(expected_type))
            section_type_name = type(section_value).__name__ if section_value is not None else "None"
            logger.debug(f"Section '{section}' type check: value={section_type_name}, expected={expected_type_name}")
            if section_value is not None and not isinstance(section_value, expected_type):  # type: ignore[arg-type]
                type_name = getattr(expected_type, "__name__", str(expected_type))
                error_msg = f"Section '{section}' has wrong type (expected {type_name}, got {section_type_name})"
                logger.error(error_msg)
                errors.append(error_msg)

    for model_name, model_config in config_data.get("models", {}).items():
        if "model_name" not in model_config:
            error_msg = f"Model '{model_name}' is missing required field 'model_name'"
            logger.error(error_msg)
            errors.append(error_msg)
        if "provider" not in model_config:
            error_msg = f"Model '{model_name}' is missing required field 'provider'"
            logger.error(error_msg)
            errors.append(error_msg)

    is_valid = len(errors) == 0
    logger.debug(f"Config validation result: is_valid={is_valid}, error_count={len(errors)}")
    return is_valid, errors


class Config:
    """Configuration manager for Arbitrium Framework."""

    def __init__(self, config_path: str | None = None) -> None:
        """
        Initialize configuration from the given path.

        Args:
            config_path: Path to configuration file (REQUIRED when calling load())

        Note:
            The framework does not provide default config paths. This must be
            explicitly specified by the caller (CLI, benchmark, or user code).
        """
        self.config_path = Path(config_path) if config_path else None
        self.config_data: dict[str, Any] = {}
        self.retry_settings: dict[str, Any] = {}
        self.feature_flags: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary where override values take precedence
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _load_defaults(self) -> dict[str, Any]:
        """
        Load all default configurations from the defaults/ directory.

        Returns:
            Merged default configuration dictionary
        """
        defaults_dir = Path(__file__).parent / "defaults"
        defaults: dict[str, Any] = {}

        if not defaults_dir.exists():
            logger.debug(f"Defaults directory not found: {defaults_dir}")
            return defaults

        yaml_files = sorted(defaults_dir.glob("*.yml"))
        logger.debug(f"Loading {len(yaml_files)} default config files from {defaults_dir}")

        for yml_file in yaml_files:
            try:
                with open(yml_file, encoding="utf-8") as f:
                    file_data = yaml.safe_load(f)
                    if file_data:
                        defaults = self._deep_merge(defaults, file_data)
                        logger.debug(f"Loaded defaults from {yml_file.name}")
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse default config {yml_file.name}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load default config {yml_file.name}: {e}")

        return defaults

    def _load_config_file(self) -> bool:
        """Load and parse the YAML config file."""
        if self.config_path is None:
            logger.error("No config path specified. The framework requires an explicit config file path.")
            return False

        if not self.config_path.exists():
            logger.error(f"Config file not found at {self.config_path.resolve()}")
            return False

        with open(self.config_path, encoding="utf-8") as f:
            try:
                user_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML config file: {e}")
                return False

        # Load defaults first
        self.config_data = self._load_defaults()
        logger.debug(f"Loaded defaults with sections: {list(self.config_data.keys())}")

        # Merge user config on top of defaults
        if user_config:
            # Special handling for models section - only use models specified in user config
            if "models" in user_config:
                user_model_keys = list(user_config["models"].keys())
                default_models = self.config_data.get("models", {})

                logger.debug(f"User specified models: {user_model_keys}")
                logger.debug(f"Available default models: {list(default_models.keys())}")

                # Build models dict with only user-specified models
                filtered_models = {}
                for model_key in user_model_keys:
                    # Start with defaults for this model (if exists)
                    base_model = default_models.get(model_key, {})
                    if not base_model:
                        logger.warning(
                            f"Model '{model_key}' not found in defaults. " f"You must provide full configuration (provider, model_name, etc.)"
                        )
                    # Merge with user overrides (handle None/empty dict)
                    user_model = user_config["models"][model_key] or {}
                    if base_model or user_model:
                        filtered_models[model_key] = self._deep_merge(base_model, user_model)

                logger.info(f"Loaded {len(filtered_models)} models: {list(filtered_models.keys())}")

                # Merge everything except models, then set models explicitly
                user_config_without_models = {k: v for k, v in user_config.items() if k != "models"}
                self.config_data = self._deep_merge(self.config_data, user_config_without_models)
                self.config_data["models"] = filtered_models
            else:
                # No models in user config - use all defaults
                self.config_data = self._deep_merge(self.config_data, user_config)

            logger.debug(f"Merged user config, final sections: {list(self.config_data.keys())}")

        logger.info(f"Loaded configuration from {self.config_path}")
        return True

    def load(self) -> bool:
        """Load configuration from file."""
        try:
            if not self._load_config_file():
                return False

            is_valid, errors = validate_config(self.config_data)
            if not is_valid:
                logger.error("Configuration validation failed. Halting execution.")
                for error in errors:
                    logger.error(f"  - {error}")
                return False

            self._setup_config_shortcuts()
            return True
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load configuration: {e!s}")
            return False
        except Exception as e:
            logger.critical(f"Unexpected error loading config: {e!s}", exc_info=True)
            return False

    def _setup_config_shortcuts(self) -> None:
        """Set up shortcut attributes for commonly accessed config sections."""
        retry_config = self.config_data.get("retry", {})
        self.retry_settings = {
            "max_attempts": retry_config.get("max_attempts", 3),
            "initial_delay": retry_config.get("initial_delay", 15),
            "max_delay": retry_config.get("max_delay", 120),
        }
        self.feature_flags = self.config_data.get("features", {})
        self.prompts = self.config_data.get("prompts", {})

    def get_model_config(self, model_key: str) -> dict[str, Any]:
        """Get configuration for a specific model, with feature flags merged in."""
        base_config = self.config_data.get("models", {}).get(model_key, {})
        if not base_config:
            return {}

        model_config = base_config.copy()
        features = self.config_data.get("features", {})

        if "llm_compression" not in model_config:
            model_config["llm_compression"] = features.get("llm_compression", True)
        if "compression_model" not in model_config:
            model_config["compression_model"] = features.get("compression_model", "ollama/qwen:1.8b")

        result: dict[str, Any] = model_config
        return result

    def get_active_model_keys(self) -> list[str]:
        """Get list of all configured model keys."""
        return list(self.config_data.get("models", {}).keys())
