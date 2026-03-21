"""Factory functions for creating models."""

from typing import Any

from certamen.config.defaults import select_model_with_highest_context
from certamen.logging import get_contextual_logger
from certamen.models.base import BaseModel
from certamen.models.cache import ResponseCache

# Import providers to trigger registration
from certamen.models.providers import LiteLLMProvider  # noqa: F401
from certamen.models.registry import ProviderRegistry

logger = get_contextual_logger("certamen.models.factory")

_global_cache: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    """Get or create global response cache instance."""
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
        logger.warning(f"No provider specified for {model_key}, skipping")
        return None

    try:
        model = await ProviderRegistry.create(
            provider, model_key, model_config, response_cache
        )
        logger.info(
            f"Created {provider} model for {model_key}: {model.display_name}"
        )
        return model
    except ValueError as e:
        logger.error(
            f"Failed to create model {model_key}: {e}. "
            f"Available providers: {ProviderRegistry.list_providers()}"
        )
        raise


def _setup_compression_models(models: dict[str, BaseModel]) -> None:
    """Auto-select and set compression model for models that need it.

    Args:
        models: Dictionary of created models
    """
    if not models:
        return

    # Check if any models need compression model selection
    needs_compression_selection = any(
        model.compression_model is None for model in models.values()
    )

    if not needs_compression_selection:
        return

    # Select model with highest context from active models
    compression_model_key = select_model_with_highest_context(models)

    if not compression_model_key:
        logger.warning(
            "Could not auto-select compression model: no models have context_window set"
        )
        return

    compression_model_name = models[compression_model_key].model_name
    logger.info(
        f"Auto-selected compression model from active models: "
        f"{compression_model_key} ({compression_model_name}) "
        f"with {models[compression_model_key].context_window:,} token context"
    )

    # Set compression model for all models that have None
    for model in models.values():
        if model.compression_model is None:
            model.compression_model = compression_model_name
            logger.debug(
                f"Set compression_model={compression_model_name} for {model.model_key}"
            )


async def create_models_from_config(
    config: dict[str, object],
) -> dict[str, BaseModel]:
    logger.info("Creating models from config...")
    models_config = config["models"]

    if not isinstance(models_config, dict):
        return {}

    cache = get_response_cache()
    models: dict[str, BaseModel] = {}

    for model_key, model_config in models_config.items():
        if not isinstance(model_config, dict):
            continue

        logger.info(f"Creating model: {model_key}")
        model = await _create_single_model(model_key, model_config, cache)

        if model is not None:
            models[model_key] = model

    _setup_compression_models(models)

    return models
