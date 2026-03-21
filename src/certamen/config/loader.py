"""Configuration handling for Certamen Framework."""

from pathlib import Path
from typing import Any

import yaml

from certamen.logging import get_contextual_logger

# Module-level logger
logger = get_contextual_logger("certamen.config")


def _get_validation_schema() -> dict[str, dict[str, Any]]:
    return {
        "models": {"required": True, "type": dict, "non_empty": True},
        "retry": {"required": True, "type": dict},
        "features": {"required": True, "type": dict},
        "prompts": {"required": True, "type": dict},
        "outputs_dir": {"required": True, "allow_none": True},
    }


def _check_section_required(
    section: str,
    rules: dict[str, Any],
    config_data: dict[str, Any],
    errors: list[str],
) -> bool:
    if rules["required"] and section not in config_data:
        error_msg = f"Missing required section '{section}'"
        logger.error(error_msg)
        errors.append(error_msg)
        return False
    return True


def _check_section_non_empty(
    section: str,
    rules: dict[str, Any],
    config_data: dict[str, Any],
    errors: list[str],
) -> bool:
    section_value = config_data.get(section)

    # Allow None if explicitly permitted
    if rules.get("allow_none") and section_value is None:
        return True

    if rules.get("non_empty") and not section_value:
        error_msg = f"Section '{section}' is empty but must contain values"
        logger.error(error_msg)
        errors.append(error_msg)
        return False
    return True


def _check_section_type(
    section: str,
    rules: dict[str, Any],
    config_data: dict[str, Any],
    errors: list[str],
) -> None:
    if "type" not in rules:
        return

    expected_type = rules["type"]
    section_value = config_data.get(section)
    expected_type_name = getattr(expected_type, "__name__", str(expected_type))
    section_type_name = (
        type(section_value).__name__ if section_value is not None else "None"
    )

    logger.debug(
        f"Section '{section}' type check: value={section_type_name}, expected={expected_type_name}"
    )

    if section_value is not None and not isinstance(
        section_value, expected_type
    ):
        type_name = getattr(expected_type, "__name__", str(expected_type))
        error_msg = f"Section '{section}' has wrong type (expected {type_name}, got {section_type_name})"
        logger.error(error_msg)
        errors.append(error_msg)


def _validate_section(
    section: str,
    rules: dict[str, Any],
    config_data: dict[str, Any],
    errors: list[str],
) -> None:
    logger.debug(
        f"Checking section '{section}': required={rules['required']}, present={section in config_data}"
    )

    if not _check_section_required(section, rules, config_data, errors):
        return
    if not _check_section_non_empty(section, rules, config_data, errors):
        return
    _check_section_type(section, rules, config_data, errors)


def _validate_model_config(
    model_name: str, model_config: dict[str, Any], errors: list[str]
) -> None:
    if "model_name" not in model_config:
        error_msg = (
            f"Model '{model_name}' is missing required field 'model_name'"
        )
        logger.error(error_msg)
        errors.append(error_msg)
    if "provider" not in model_config:
        error_msg = (
            f"Model '{model_name}' is missing required field 'provider'"
        )
        logger.error(error_msg)
        errors.append(error_msg)


def validate_config(config_data: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    schema = _get_validation_schema()

    logger.debug(f"Validating config sections: {list(config_data.keys())}")

    # Validate each section
    for section, rules in schema.items():
        _validate_section(section, rules, config_data, errors)

    # Validate model configurations
    for model_name, model_config in config_data.get("models", {}).items():
        _validate_model_config(model_name, model_config, errors)

    is_valid = len(errors) == 0
    logger.debug(
        f"Config validation result: is_valid={is_valid}, error_count={len(errors)}"
    )
    return is_valid, errors


class Config:
    """Configuration manager for Certamen Framework."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.config_data: dict[str, Any] = {}
        self.retry_settings: dict[str, Any] = {}
        self.feature_flags: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _load_defaults(self) -> dict[str, Any]:
        from certamen.config.defaults import get_defaults

        defaults = get_defaults()
        logger.debug(f"Loaded defaults with sections: {list(defaults.keys())}")
        return defaults

    def _validate_config_path(self) -> bool:
        if self.config_path is None:
            logger.error(
                "No config path specified. The framework requires an explicit config file path."
            )
            return False

        if not self.config_path.exists():
            logger.error(
                f"Config file not found at {self.config_path.resolve()}"
            )
            return False

        return True

    def _parse_yaml_file(self) -> dict[str, Any] | None:
        try:
            with open(self.config_path, encoding="utf-8") as f:  # type: ignore[arg-type]
                result = yaml.safe_load(f)
                if result is None:
                    return None
                return dict(result) if isinstance(result, dict) else None
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config file: {e}")
            return None

    def _merge_single_model(
        self,
        model_key: str,
        user_model_config: dict[str, Any],
        default_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        base_model = default_models.get(model_key, {})
        if not base_model:
            logger.warning(
                f"Model '{model_key}' not found in defaults. "
                f"You must provide full configuration (provider, model_name, etc.)"
            )

        user_model = user_model_config or {}
        if base_model or user_model:
            return self._deep_merge(base_model, user_model)
        return None

    def _build_filtered_models(
        self,
        user_config: dict[str, Any],
        default_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        user_model_keys = list(user_config["models"].keys())

        logger.debug(f"User specified models: {user_model_keys}")
        logger.debug(
            f"Available default models: {list(default_models.keys())}"
        )

        filtered_models = {}
        for model_key in user_model_keys:
            merged_model = self._merge_single_model(
                model_key, user_config["models"][model_key], default_models
            )
            if merged_model:
                filtered_models[model_key] = merged_model

        logger.info(
            f"Loaded {len(filtered_models)} models: {list(filtered_models.keys())}"
        )
        return filtered_models

    def _merge_with_models_handling(self, user_config: dict[str, Any]) -> None:
        # If no models section, or models is explicitly empty, use all defaults
        if "models" not in user_config or not user_config.get("models"):
            logger.debug(
                "No models specified in config - using all default models"
            )
            self.config_data = self._deep_merge(self.config_data, user_config)
            return

        default_models = self.config_data.get("models", {})
        filtered_models = self._build_filtered_models(
            user_config, default_models
        )

        # Merge everything except models, then set models explicitly
        user_config_without_models = {
            k: v for k, v in user_config.items() if k != "models"
        }
        self.config_data = self._deep_merge(
            self.config_data, user_config_without_models
        )
        self.config_data["models"] = filtered_models

    def _load_config_file(self) -> bool:
        if not self._validate_config_path():
            return False

        user_config = self._parse_yaml_file()
        if user_config is None:
            return False

        # Load defaults first
        self.config_data = self._load_defaults()
        logger.debug(
            f"Loaded defaults with sections: {list(self.config_data.keys())}"
        )

        # Merge user config on top of defaults
        if user_config:
            self._merge_with_models_handling(user_config)
            logger.debug(
                f"Merged user config, final sections: {list(self.config_data.keys())}"
            )

        logger.info(f"Loaded configuration from {self.config_path}")
        return True

    def load(self) -> bool:
        """Load configuration from file."""
        try:
            if not self._load_config_file():
                return False

            is_valid, errors = validate_config(self.config_data)
            if not is_valid:
                logger.error(
                    "Configuration validation failed. Halting execution."
                )
                for error in errors:
                    logger.error(f"  - {error}")
                return False

            self._setup_config_shortcuts()
            return True
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load configuration: {e!s}")
            return False
        except Exception as e:
            logger.critical(
                f"Unexpected error loading config: {e!s}", exc_info=True
            )
            return False

    def _setup_config_shortcuts(self) -> None:
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
            model_config["llm_compression"] = features.get(
                "llm_compression", True
            )
        if "compression_model" not in model_config:
            # None means auto-select model with highest context window
            model_config["compression_model"] = features.get(
                "compression_model", None
            )

        result: dict[str, Any] = model_config
        return result

    def get_active_model_keys(self) -> list[str]:
        """Get list of all configured model keys."""
        return list(self.config_data.get("models", {}).keys())
