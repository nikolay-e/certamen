"""Standard model providers for Arbitrium."""

from collections.abc import Callable
from typing import Any, TypeVar

from arbitrium_core.infrastructure.llm.litellm_adapter import LiteLLMModel
from arbitrium_core.infrastructure.llm.registry import ProviderRegistry
from arbitrium_core.ports.llm import BaseModel

__all__ = ["LITELLM_PROVIDERS", "LiteLLMProvider", "ProviderRegistry"]

T = TypeVar("T")


def register_providers(provider_names: list[str]) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        for name in provider_names:
            ProviderRegistry.register(name)(cls)  # type: ignore[arg-type]
        return cls

    return decorator


LITELLM_PROVIDERS = [
    "openai",
    "anthropic",
    "vertex_ai",
    "google",
    "google_ai_studio",
    "bedrock",
    "azure",
    "cohere",
    "replicate",
    "huggingface",
    "ollama",
    "together_ai",
    "anyscale",
    "palm",
    "mistral",
    "xai",
    "groq",
    "perplexity",
    "deepseek",
    "openrouter",
]


@register_providers(LITELLM_PROVIDERS)
class LiteLLMProvider:
    @classmethod
    async def from_config(
        cls,
        model_key: str,
        config: dict[str, Any],
        response_cache: Any | None = None,
    ) -> BaseModel:
        return await LiteLLMModel.from_config(
            model_key, config, response_cache
        )
