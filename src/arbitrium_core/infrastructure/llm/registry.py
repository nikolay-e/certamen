"""Provider registry for extensible model creation."""

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Protocol

from arbitrium_core.ports.llm import BaseModel
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger("arbitrium.models.registry")


class ModelProvider(Protocol):
    """Protocol for model providers."""

    @classmethod
    def from_config(
        cls,
        model_key: str,
        config: dict[str, Any],
        response_cache: Any | None = None,
    ) -> Awaitable[BaseModel]: ...


class ProviderRegistry:
    """Registry of model providers for extensible model creation.

    This allows adding new providers without modifying the factory code.
    Production providers are registered at module import time.
    Test providers (like MockModel) are registered in test fixtures.
    """

    _providers: ClassVar[dict[str, type[ModelProvider]]] = {}

    @classmethod
    def register(
        cls, provider_name: str
    ) -> Callable[[type[ModelProvider]], type[ModelProvider]]:
        """Decorator to register a provider.

        Usage:
            @ProviderRegistry.register("openai")
            class OpenAIProvider:
                @classmethod
                def from_config(cls, model_key, config):
                    return LiteLLMModel.from_config(model_key, config)

        Args:
            provider_name: The provider name (e.g., "openai", "anthropic")

        Returns:
            Decorator function
        """

        def decorator(
            provider_class: type[ModelProvider],
        ) -> type[ModelProvider]:
            cls._providers[provider_name] = provider_class
            logger.debug(f"Registered provider: {provider_name}")
            return provider_class

        return decorator

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model_key: str,
        config: dict[str, Any],
        response_cache: Any | None = None,
    ) -> BaseModel:
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Note: Test providers like 'mock' must be registered "
                f"in test fixtures before use."
            )

        return await provider_class.from_config(
            model_key, config, response_cache
        )

    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: The provider name to check

        Returns:
            True if provider is registered, False otherwise
        """
        return provider_name in cls._providers

    @classmethod
    def list_providers(cls) -> list[str]:
        """Get list of all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
