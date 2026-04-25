from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import providers to trigger registration at bootstrap time
import certamen.infrastructure.llm.providers  # noqa: F401

if TYPE_CHECKING:
    from certamen.engine import Certamen

from certamen.domain.errors import ConfigurationError
from certamen.domain.model_selection import select_model_by_capacity
from certamen.infrastructure.cache.sqlite_cache import ResponseCache
from certamen.infrastructure.config.loader import Config
from certamen.infrastructure.llm.registry import ProviderRegistry
from certamen.infrastructure.secrets.env_secrets import load_secrets
from certamen.ports.llm import BaseModel
from certamen.shared.constants import HEALTH_CHECK_PROMPT
from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger("certamen.bootstrap")

_global_cache: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    global _global_cache  # noqa: PLW0603
    if _global_cache is None:
        _global_cache = ResponseCache(enabled=True)
        logger.info("Response cache enabled (certamen_cache.db)")
    return _global_cache


async def _create_single_model(
    model_key: str, model_config: dict[str, object], response_cache: Any | None
) -> BaseModel | None:
    provider = model_config.get("provider", "")
    if not provider or not isinstance(provider, str):
        logger.warning("No provider specified for %s, skipping", model_key)
        return None

    try:
        model = await ProviderRegistry.create(
            provider, model_key, model_config, response_cache
        )
        logger.info(
            "Created %s model for %s: %s",
            provider,
            model_key,
            model.display_name,
        )
        return model
    except ValueError as e:
        logger.error(
            "Failed to create model %s: %s. Available providers: %s",
            model_key,
            e,
            ProviderRegistry.list_providers(),
        )
        raise


def _setup_compression_models(models: dict[str, BaseModel]) -> None:
    if not models:
        return

    needs_compression_selection = any(
        model.compression_model is None for model in models.values()
    )

    if not needs_compression_selection:
        return

    compression_model_key = select_model_by_capacity(
        models, include_max_tokens=False
    )

    if not compression_model_key:
        logger.warning(
            "Could not auto-select compression model: no models have context_window set"
        )
        return

    compression_model_name = models[compression_model_key].model_name
    logger.info(
        "Auto-selected compression model from active models: %s (%s) with %s token context",
        compression_model_key,
        compression_model_name,
        f"{models[compression_model_key].context_window:,}",
    )

    for model in models.values():
        if model.compression_model is None:
            model.compression_model = compression_model_name
            logger.debug(
                "Set compression_model=%s for %s",
                compression_model_name,
                model.model_key,
            )


async def create_models(config: dict[str, Any]) -> dict[str, BaseModel]:
    logger.info("Creating models from config...")
    models_config = config["models"]

    if not isinstance(models_config, dict):
        return {}

    cache = get_response_cache()
    models: dict[str, BaseModel] = {}

    for model_key, model_config in models_config.items():
        if not isinstance(model_config, dict):
            continue

        logger.info("Creating model: %s", model_key)
        model = await _create_single_model(model_key, model_config, cache)

        if model is not None:
            models[model_key] = model

    _setup_compression_models(models)

    return models


def _get_active_providers(config: dict[str, Any]) -> set[str]:
    return {
        model_cfg.get("provider", "").lower()
        for model_cfg in config.get("models", {}).values()
        if model_cfg.get("provider")
    }


def _should_skip_secrets_loading(
    config: dict[str, Any], active_providers: set[str]
) -> bool:
    local_providers = {"ollama"}
    if active_providers and active_providers.issubset(local_providers):
        logger.info("All models use local providers, skipping secret loading")
        return True

    if "secrets" not in config:
        logger.info("No secrets section in config, skipping secret loading")
        return True

    return False


def load_secrets_for_providers(config: dict[str, Any]) -> None:
    active_providers = _get_active_providers(config)

    if _should_skip_secrets_loading(config, active_providers):
        return

    try:
        load_secrets(config, list(active_providers))
    except ConfigurationError as e:
        # nosemgrep: python-logger-credential-disclosure
        logger.warning("Failed to load secrets: %s", e)
        logger.warning("Continuing without secrets - remote models may fail")


async def _check_single_model(
    model_key: str, model: BaseModel
) -> tuple[str, BaseModel | None, Exception | None]:
    try:
        response = await model.generate(HEALTH_CHECK_PROMPT)

        if response.is_error():
            error = RuntimeError(response.error or "Unknown error")
            logger.warning("%s: %s", model.full_display_name, error)
            return model_key, None, error
        else:
            logger.info("%s: Healthy", model.full_display_name)
            return model_key, model, None

    except Exception as e:
        logger.warning("%s: %s", model.full_display_name, e)
        return model_key, None, e


async def health_check_models(
    models: dict[str, BaseModel],
) -> tuple[dict[str, BaseModel], dict[str, Exception]]:
    import asyncio

    logger.info("Performing health check on %s models...", len(models))

    tasks = [
        _check_single_model(model_key, model)
        for model_key, model in models.items()
    ]

    results = await asyncio.gather(*tasks)

    healthy: dict[str, BaseModel] = {}
    failed: dict[str, Exception] = {}

    for model_key, model, error in results:
        if error is not None:
            failed[model_key] = error
        elif model is not None:
            healthy[model_key] = model

    logger.info(
        "Health check complete: %s healthy, %s failed",
        len(healthy),
        len(failed),
    )
    return healthy, failed


async def build_certamen(
    config: Config | dict[str, Any] | str | Path,
    skip_secrets: bool = False,
    skip_health_check: bool = False,
) -> "Certamen":
    from certamen.engine import Certamen

    # Handle different config input types
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        logger.info("Loading configuration from %s", config_path)
        config_obj = Config(str(config_path))
        if not config_obj.load():
            raise ConfigurationError(
                f"Failed to load configuration from {config_path}"
            )
    elif isinstance(config, dict):
        config_obj = Config()
        is_valid, errors = config_obj.load_from_dict(config)
        if not is_valid:
            error_details = "\n  - ".join(errors)
            raise ConfigurationError(
                f"Invalid configuration provided. Missing or invalid sections:\n  - {error_details}\n\n"
                f"Required sections: models, retry, features, prompts, outputs_dir"
            )
    elif isinstance(config, Config):
        config_obj = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    config_data = config_obj.config_data

    if not skip_secrets:
        load_secrets_for_providers(config_data)

    logger.info("Creating models from configuration")
    all_models = await create_models(config_data)

    if not all_models:
        logger.warning("No models configured")
        return Certamen(
            config=config_obj,
            all_models={},
            healthy_models={},
            failed_models={},
        )

    if skip_health_check:
        healthy_models = dict(all_models)
        failed_models: dict[str, Exception] = {}
    else:
        healthy_models, failed_models = await health_check_models(all_models)

    logger.info(
        "Certamen initialized: %s healthy, %s failed",
        len(healthy_models),
        len(failed_models),
    )

    return Certamen(
        config=config_obj,
        all_models=all_models,
        healthy_models=healthy_models,
        failed_models=failed_models,
    )
