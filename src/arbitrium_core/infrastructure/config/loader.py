from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from arbitrium_core.infrastructure.config.models import (
    ArbitriumConfig,
    ModelConfig,
)
from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.mapping_utils import deep_merge

logger = get_contextual_logger("arbitrium.config")


def validate_config(config_data: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    logger.debug(f"Validating config sections: {list(config_data.keys())}")

    # Check models section is non-empty
    models = config_data.get("models", {})
    if not models:
        errors.append("Section 'models' is empty but must contain values")

    # Validate each model with pydantic
    for model_key, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            errors.append(f"Model '{model_key}' must be a dictionary")
            continue
        try:
            ModelConfig(**model_cfg)
        except ValidationError as e:
            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])
                errors.append(f"Model '{model_key}': {field} - {err['msg']}")

    # Validate full config structure
    try:
        ArbitriumConfig(**config_data)
    except ValidationError as e:
        for err in e.errors():
            loc = err["loc"]
            if loc and loc[0] == "models":
                continue  # Already handled above
            field = ".".join(str(x) for x in loc)
            errors.append(f"{field}: {err['msg']}")

    is_valid = len(errors) == 0
    logger.debug(
        f"Config validation: is_valid={is_valid}, errors={len(errors)}"
    )
    return is_valid, errors


class Config:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.config_data: dict[str, Any] = {}
        self.retry_settings: dict[str, Any] = {}
        self.feature_flags: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def _load_defaults(self) -> dict[str, Any]:
        from arbitrium_core.infrastructure.config.defaults import get_defaults

        defaults = get_defaults()
        logger.debug(f"Loaded defaults with sections: {list(defaults.keys())}")
        return defaults

    def _validate_config_path(self) -> bool:
        if self.config_path is None:
            logger.error("No config path specified.")
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
                f"Provide full configuration (provider, model_name, etc.)"
            )
        user_model = user_model_config or {}
        if base_model or user_model:
            return deep_merge(base_model, user_model)
        return None

    def _build_filtered_models(
        self,
        user_config: dict[str, Any],
        default_models: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        user_model_keys = list(user_config["models"].keys())
        logger.debug(f"User specified models: {user_model_keys}")

        filtered_models = {}
        for model_key in user_model_keys:
            merged = self._merge_single_model(
                model_key, user_config["models"][model_key], default_models
            )
            if merged:
                filtered_models[model_key] = merged

        logger.info(
            f"Loaded {len(filtered_models)} models: {list(filtered_models.keys())}"
        )
        return filtered_models

    def _merge_with_models_handling(self, user_config: dict[str, Any]) -> None:
        if "models" not in user_config or not user_config.get("models"):
            logger.debug("No models specified - using all defaults")
            user_config_without_models = {
                k: v for k, v in user_config.items() if k != "models"
            }
            self.config_data = deep_merge(
                self.config_data, user_config_without_models
            )
            return

        default_models = self.config_data.get("models", {})
        filtered_models = self._build_filtered_models(
            user_config, default_models
        )

        user_config_without_models = {
            k: v for k, v in user_config.items() if k != "models"
        }
        self.config_data = deep_merge(
            self.config_data, user_config_without_models
        )
        self.config_data["models"] = filtered_models

    def _load_config_file(self) -> bool:
        if not self._validate_config_path():
            return False
        user_config = self._parse_yaml_file()
        if user_config is None:
            return False

        self.config_data = self._load_defaults()
        if user_config:
            self._merge_with_models_handling(user_config)
        logger.info(f"Loaded configuration from {self.config_path}")
        return True

    def load(self) -> bool:
        try:
            if not self._load_config_file():
                return False

            is_valid, errors = validate_config(self.config_data)
            if not is_valid:
                logger.error("Configuration validation failed.")
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
            "initial_delay": retry_config.get("initial_delay", 10.0),
            "max_delay": retry_config.get("max_delay", 60.0),
        }
        self.feature_flags = self.config_data.get("features", {})
        self.prompts = self.config_data.get("prompts", {})

    def get_model_config(self, model_key: str) -> dict[str, Any]:
        models: dict[str, Any] = self.config_data.get("models", {})
        base_config: dict[str, Any] = models.get(model_key, {})
        if not base_config:
            return {}

        model_config: dict[str, Any] = dict(base_config)
        features: dict[str, Any] = self.config_data.get("features", {})

        if "llm_compression" not in model_config:
            model_config["llm_compression"] = features.get(
                "llm_compression", True
            )
        if "compression_model" not in model_config:
            model_config["compression_model"] = features.get(
                "compression_model", None
            )

        return model_config

    def get_active_model_keys(self) -> list[str]:
        return list(self.config_data.get("models", {}).keys())

    def load_from_dict(
        self, settings: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        logger.info("Loading configuration from dictionary")
        self.config_data = self._load_defaults()

        if settings:
            self._merge_with_models_handling(settings)

        is_valid, errors = validate_config(self.config_data)
        if is_valid:
            self._setup_config_shortcuts()

        return is_valid, errors
