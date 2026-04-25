from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Protocol

from certamen.ports.llm import BaseModel
from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger("certamen.models.registry")


class ModelProvider(Protocol):
    @classmethod
    def from_config(
        cls,
        model_key: str,
        config: dict[str, Any],
        response_cache: Any | None = None,
    ) -> Awaitable[BaseModel]: ...


class ProviderRegistry:
    _providers: ClassVar[dict[str, type[ModelProvider]]] = {}

    @classmethod
    def register(
        cls, provider_name: str
    ) -> Callable[[type[ModelProvider]], type[ModelProvider]]:
        def decorator(
            provider_class: type[ModelProvider],
        ) -> type[ModelProvider]:
            cls._providers[provider_name] = provider_class
            logger.debug("Registered provider: %s", provider_name)
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
        return provider_name in cls._providers

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._providers.keys())
