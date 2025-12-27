from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from arbitrium_core.ports.cache import CacheProtocol


class ModelResponse:
    def __init__(
        self,
        content: str,
        error: str | None = None,
        error_type: str | None = None,
        provider: str | None = None,
        cost: float = 0.0,
    ):
        self.content = content
        self.error = error
        self.error_type = error_type
        self.provider = provider
        self.cost = cost
        self.is_successful = error is None

    @classmethod
    def create_success(
        cls, content: str, cost: float = 0.0
    ) -> "ModelResponse":
        return cls(content=content, cost=cost)

    @classmethod
    def create_error(
        cls,
        error_message: str,
        error_type: str | None = None,
        provider: str | None = None,
    ) -> "ModelResponse":
        return cls(
            content=f"Error: {error_message}",
            error=error_message,
            error_type=error_type,
            provider=provider,
        )

    def is_error(self) -> bool:
        return self.error is not None


class BaseModel(ABC):
    def __init__(
        self,
        model_key: str,
        model_name: str,
        display_name: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        context_window: int | None = None,
        use_llm_compression: bool = True,
        compression_model: str | None = None,
    ):
        self.model_key = model_key
        self.model_name = model_name
        self.display_name = display_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_llm_compression = use_llm_compression
        self.compression_model = compression_model
        if context_window is None:
            error_message = f"context_window is required for model {model_name}. Please specify context_window in the model configuration."
            raise ValueError(error_message)
        self.context_window = context_window

    @property
    def full_display_name(self) -> str:
        return f"{self.display_name} ({self.model_name})"

    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        pass

    @abstractmethod
    async def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        initial_delay: float | None = None,
        max_delay: float | None = None,
    ) -> ModelResponse:
        pass


class ModelProvider(Protocol):
    @classmethod
    def from_config(
        cls,
        model_key: str,
        config: dict[str, Any],
        response_cache: "CacheProtocol | None" = None,
    ) -> Awaitable[BaseModel]: ...
