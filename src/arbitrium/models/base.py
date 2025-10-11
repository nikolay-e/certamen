"""
Base module for LLM model interactions in Arbitrium Framework.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import litellm

from ..logging import get_contextual_logger
from ..utils.constants import ERROR_PATTERNS
from ..utils.exceptions import ModelResponseError

# Module-level logger for consistent logging
_logger = None


def _get_module_logger() -> Any:
    """Get or create module-level logger."""
    global _logger  # noqa: PLW0603
    if _logger is None:
        _logger = get_contextual_logger("arbitrium.models")
    return _logger


def analyze_error_response(response: Any) -> tuple[bool, str]:
    """Analyzes an error response to determine if it's retryable and what type it is."""
    error_msg = ""
    if hasattr(response, "error"):
        error_msg = str(response.error).lower()
        error_type = getattr(response, "error_type", None)
        if error_type:
            return error_type in ["rate_limit", "timeout", "connection", "service", "overloaded"], error_type
    elif isinstance(response, Exception):
        error_msg = str(response).lower()
        if "notfounderror" in type(response).__name__.lower():
            return False, "not_found"
        if "authenticationerror" in type(response).__name__.lower():
            return False, "authentication"

    if any(p in error_msg for p in ["permission_denied", "service_disabled", "api has not been used"]):
        return False, "permission_denied"

    for error_type, patterns in ERROR_PATTERNS.items():
        if any(p in error_msg for p in patterns):
            return True, error_type

    return False, "general"


# Helper for retry delay calculation
async def _calculate_retry_delay(
    current_delay: float,
    start_time: float,
    total_timeout: float,
    initial_delay: float,
    max_delay: float,
    logger: Any | None = None,
    error_type: str = "general",
    provider: str = "default",
) -> float | None:
    """Calculates the delay for the next retry attempt with jitter."""
    backoff_multipliers = {
        "rate_limit": {"anthropic": 2.5, "default": 2.0},
        "overloaded": {"anthropic": 3.0, "default": 2.5},
        "timeout": 1.5,
        "connection": 1.8,
        "service": 2.0,
        "general": 1.5,
    }

    provider = provider.lower() if provider else "default"
    multiplier_value = backoff_multipliers.get(error_type, backoff_multipliers["general"])

    multiplier: float
    if isinstance(multiplier_value, dict):
        provider_mult = multiplier_value.get(provider, multiplier_value["default"])
        multiplier = float(provider_mult) if provider_mult is not None else 1.5
    elif isinstance(multiplier_value, (int, float)):
        multiplier = float(multiplier_value)
    else:
        multiplier = 1.5  # Default fallback

    jitter_range = 0.05 if error_type in ["rate_limit", "overloaded"] and provider == "anthropic" else 0.1
    jitter_factor = 1.0 + random.uniform(-jitter_range, jitter_range)
    jittered_delay = min(current_delay * multiplier, max_delay) * jitter_factor

    remaining_time = total_timeout - (time.monotonic() - start_time)
    if remaining_time <= 0:
        if logger:
            logger.error("No time left for retry. Stopping retries.")
        return None

    actual_delay = min(jittered_delay, remaining_time)
    min_delay_factor = 0.5 if error_type in ["rate_limit", "overloaded"] else 0.25
    if actual_delay < initial_delay * min_delay_factor:
        if logger:
            logger.error("Not enough time for proper retry delay. Stopping retries.")
        return None

    await asyncio.sleep(actual_delay)
    return min(current_delay * multiplier, max_delay)


class ModelResponse:
    """Represents a response from an LLM model."""

    def __init__(
        self,
        content: str,
        error: str | None = None,
        error_type: str | None = None,
        provider: str | None = None,
        cost: float = 0.0,
    ):
        """
        Initialize a model response.

        Args:
            content: The text content of the response
            error: Error message if the request failed, None otherwise
            error_type: Type of error for better handling (e.g., "rate_limit", "timeout")
            provider: The provider that generated this response/error
            cost: Cost of the API call in USD
        """
        self.content = content
        self.error = error
        self.error_type = error_type
        self.provider = provider
        self.cost = cost
        self.is_successful = error is None

    @classmethod
    def create_success(cls, content: str, cost: float = 0.0) -> "ModelResponse":
        """Create a successful response."""
        return cls(content=content, cost=cost)

    @classmethod
    def create_error(cls, error_message: str, error_type: str | None = None, provider: str | None = None) -> "ModelResponse":
        """Create an error response."""
        return cls(
            content=f"Error: {error_message}",
            error=error_message,
            error_type=error_type,
            provider=provider,
        )

    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return self.error is not None


class BaseModel(ABC):
    """Abstract base class for LLM models."""

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
        compression_model: str = "ollama/qwen:1.8b",
    ):
        """
        Initialize a model.

        Args:
            model_key: Identifier used in config (e.g., 'gpt')
            model_name: Actual model identifier for the API (e.g., 'gpt-4.1')
            display_name: Human-readable name for logs and UI
            provider: Provider name (e.g., 'openai')
            max_tokens: Maximum tokens for the completion/response
            temperature: Temperature setting for response generation
            context_window: Total context window size
            use_llm_compression: Whether to use LLM for prompt compression
            compression_model: Model to use for compression
        """
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
        """Returns display name with model name in parentheses for complete identification."""
        return f"{self.display_name} ({self.model_name})"

    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response to the given prompt.

        Args:
            prompt: The prompt to send to the model

        Returns:
            A ModelResponse containing the response text or error
        """


class LiteLLMModel(BaseModel):
    """Implementation of BaseModel using LiteLLM for API access."""

    def __init__(
        self,
        model_key: str,
        model_name: str,
        display_name: str,
        provider: str,
        temperature: float,
        max_tokens: int = 1024,
        context_window: int | None = None,  # Kept for compatibility with BaseModel
        reasoning: bool = False,
        reasoning_effort: str | None = None,
        model_config: dict[str, Any] | None = None,
        use_llm_compression: bool = True,
        compression_model: str = "ollama/qwen:1.8b",
        system_prompt: str | None = None,
    ):
        """Initialize a LiteLLM-backed model."""
        super().__init__(
            model_key=model_key,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,  # Passed to parent but not used in this implementation
            use_llm_compression=use_llm_compression,
            compression_model=compression_model,
        )
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort  # "low", "medium", "high"
        self.system_prompt = system_prompt  # Optional system prompt for role-playing

        # Check if this model requires temperature=1.0
        # Models that require this should have force_temp_one: true in their YAML config
        # Example: o4-mini, o3-2025-04-16, etc.
        self.requires_temp_one = model_config is not None and hasattr(model_config, "get") and model_config.get("force_temp_one", False)

        # Note: LiteLLM logging is centrally disabled in utils/structured_log.py

    def _try_extract_openai_format(self, response: Any, logger: Any) -> str | None:
        """Try to extract content from OpenAI/LiteLLM format."""
        if not hasattr(response, "choices") or not response.choices:
            return None
        try:
            content: str = response.choices[0].message.content
            if content and content.strip():
                return content
        except (AttributeError, IndexError) as e:
            logger.debug(f"Failed to extract from choices object: {e}")
        return None

    def _try_extract_dict_format(self, response: Any, logger: Any) -> str | None:
        """Try to extract content from dict format."""
        if not isinstance(response, dict):
            return None

        # Try OpenAI dict format
        if "choices" in response:
            try:
                content: str = response["choices"][0]["message"]["content"]
                if content:
                    return content
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Failed to extract from dict choices: {e}")

        # Try Gemini dict format
        if "candidates" in response:
            try:
                content_val: str | None = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                if content_val:
                    return content_val
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Failed to extract from candidates: {e}")

        return None

    def _try_extract_gemini_format(self, response: Any, logger: Any) -> str | None:
        """Try to extract content from Gemini object format."""
        if not hasattr(response, "candidates") or not response.candidates:
            return None
        try:
            content: str = response.candidates[0].content.parts[0].text
            if content:
                return content
        except (AttributeError, IndexError) as e:
            logger.debug(f"Failed to extract from candidates object: {e}")
        return None

    def _try_extract_common_attrs(self, response: Any) -> str | None:
        """Try to extract content from common attribute names."""
        for attr in ["content", "text", "completion", "answer", "response", "output", "result"]:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _extract_response_content(self, response: Any) -> str | None:
        """Extracts content from the LLM response with multiple fallback strategies."""
        logger = _get_module_logger()

        # Try different extraction strategies in order
        strategies: list[Any] = [
            lambda: self._try_extract_openai_format(response, logger),
            lambda: self._try_extract_dict_format(response, logger),
            lambda: self._try_extract_gemini_format(response, logger),
            lambda: self._try_extract_common_attrs(response),
        ]

        for strategy in strategies:
            content: str | None = strategy()
            if content:
                return content

        # Last resort: if response is already a string
        if isinstance(response, str) and response.strip():
            logger.warning("Response was already a string, using directly")
            return response

        return None

    def _extract_response_cost(self, response: Any) -> float:
        """Extracts cost from the LLM response."""
        logger = _get_module_logger()

        # LiteLLM adds cost information to the response object
        if hasattr(response, "_hidden_params") and hasattr(response._hidden_params, "response_cost"):
            cost: float = response._hidden_params.response_cost
            logger.info(f"💰 Cost extracted from _hidden_params.response_cost: ${cost:.4f}")
            return cost
        if hasattr(response, "response_cost"):
            cost_val: float = response.response_cost
            logger.info(f"💰 Cost extracted from response_cost: ${cost_val:.4f}")
            return cost_val

        # Enhanced fallback - try to extract from LiteLLM usage calculation
        if hasattr(response, "usage") and response.usage:
            try:
                # Try LiteLLM's completion_cost function if available
                import litellm

                if hasattr(litellm, "completion_cost"):
                    cost_calc: float = litellm.completion_cost(completion_response=response)
                    if cost_calc and cost_calc > 0:
                        logger.info(f"💰 Cost calculated via litellm.completion_cost: ${cost_calc:.4f}")
                        return cost_calc
            except Exception as e:
                logger.debug(f"Failed to calculate cost via litellm.completion_cost: {e}")

        logger.debug("💰 No cost information found in response, returning 0.0")
        return 0.0

    def _handle_prompt_size_validation(self, prompt: str, logger: Any) -> str | None:
        """Handle prompt size validation - DISABLED to preserve full prompts.

        Users pay for the full context, so we don't truncate anything.
        Large language models can handle large contexts.
        """
        # Always return the full prompt without any truncation or compression
        return prompt

    def _clean_response_content(self, content: str, logger: Any) -> str:
        """Clean response content - preserve markdown formatting.

        User pays for full content including markdown formatting.
        Markdown is useful for readability and structure.
        """
        # Only strip whitespace, preserve all markdown formatting
        return content.strip()

    def _try_extract_content_from_response(self, response: Any, cost: float, logger: Any) -> ModelResponse | None:
        """Try to extract content from response, return ModelResponse if successful."""
        content = self._extract_response_content(response)

        if content and content.strip():
            cleaned_content = self._clean_response_content(content, logger)
            return ModelResponse.create_success(cleaned_content, cost=cost)

        # Try fallback
        logger.warning(f"{self.display_name} returned a response but content extraction failed. Using str() fallback.")
        logger.debug(f"Response object type: {type(response)}, has choices: {hasattr(response, 'choices')}")

        fallback_content = str(response)
        if fallback_content and len(fallback_content.strip()) > 10:
            return ModelResponse.create_success(fallback_content.strip(), cost=cost)

        return None

    async def generate(self, prompt: str) -> ModelResponse:
        """Generates a response to the given prompt using LiteLLM."""
        logger = _get_module_logger()
        if not prompt or not prompt.strip():
            return ModelResponse.create_error("Empty prompt provided")

        # Handle prompt size validation
        validated_prompt = self._handle_prompt_size_validation(prompt, logger)
        if validated_prompt is None:
            method = "LLM compression" if self.use_llm_compression else "truncation"
            return ModelResponse.create_error(f"Prompt too large even after {method}")
        prompt = validated_prompt

        # Prepare parameters
        temperature = 1.0 if self.requires_temp_one else float(self.temperature)

        # Build messages array with optional system prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            logger.debug(f"Using reasoning_effort={self.reasoning_effort} for {self.display_name}")

        try:
            response = await asyncio.wait_for(litellm.acompletion(**params), timeout=600)
            cost = self._extract_response_cost(response)

            # Try to extract and clean content
            model_response = self._try_extract_content_from_response(response, cost, logger)
            if model_response:
                return model_response

            # Only error if truly nothing useful
            raise ModelResponseError(f"Model {self.display_name} returned empty or unusable response", model_key=self.model_key)
        except (
            litellm.exceptions.RateLimitError,
            litellm.exceptions.Timeout,
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.InternalServerError,
            ModelResponseError,
        ) as e:
            error_type, error_message = self._classify_exception(e)
            logger.warning(f"{error_message}. model={self.model_name}, provider={self.provider}")
            return ModelResponse.create_error(error_message, error_type=error_type, provider=self.provider)
        except Exception as e:
            error_type, error_message = self._classify_exception(e)
            logger.error(f"API error with {self.display_name}: {error_message}", exc_info=True)
            return ModelResponse.create_error(error_message, error_type=error_type, provider=self.provider)

    def _classify_exception(self, exc: Exception) -> tuple[str, str]:
        """Classifies an exception into an error type and message."""
        if isinstance(exc, litellm.exceptions.RateLimitError):
            return "rate_limit", f"Rate limit exceeded with {self.display_name}: {exc}"
        if isinstance(exc, (litellm.exceptions.Timeout, asyncio.TimeoutError)):
            return "timeout", f"Request timed out for {self.display_name}: {exc}"
        if isinstance(exc, litellm.exceptions.AuthenticationError):
            return "authentication", f"Authentication failed with {self.display_name}: {exc}"
        if isinstance(exc, litellm.exceptions.ServiceUnavailableError):
            return "service_unavailable", f"Service unavailable for {self.display_name}: {exc}"
        if isinstance(exc, litellm.exceptions.InternalServerError):
            return ("overloaded" if "overload" in str(exc).lower() else "service"), f"Server error with {self.display_name}: {exc}"
        if isinstance(exc, ModelResponseError):
            return "model_response_error", str(exc)
        return "general", f"Unexpected API error with {self.display_name}: {exc}"

    @classmethod
    def _convert_info_to_dict(cls, info: Any, model_name: str, logger: Any) -> dict[str, Any]:
        """Convert model info object to dictionary."""
        if isinstance(info, dict):
            return info
        if hasattr(info, "model_dump"):
            result = info.model_dump()
            if isinstance(result, dict):
                return result
        if hasattr(info, "dict"):
            result = info.dict()
            if isinstance(result, dict):
                return result
        try:
            result = dict(info)
            if isinstance(result, dict):
                return result
        except (TypeError, ValueError):
            pass
        logger.debug(f"Could not convert model info to dict for {model_name}")
        return {}

    @classmethod
    def _get_model_info_from_litellm(cls, model_name: str) -> dict[str, Any]:
        """
        Attempts to get model information from LiteLLM.

        Args:
            model_name: The model name to query

        Returns:
            Dictionary with model info, or empty dict if not found
        """
        logger = _get_module_logger()
        try:
            info = litellm.get_model_info(model_name)
            logger.debug(f"Retrieved model info from LiteLLM for {model_name}: {info}")
            return cls._convert_info_to_dict(info, model_name, logger)
        except Exception as e:
            logger.debug(f"Could not retrieve model info from LiteLLM for {model_name}: {e}")
            return {}

    @classmethod
    def from_config(cls, model_key: str, model_config: dict[str, Any]) -> "LiteLLMModel":
        """
        Create a model instance from configuration.

        Auto-detects max_tokens and context_window from LiteLLM if not provided in config.

        Args:
            model_key: The key used for this model in the config
            model_config: The model configuration dictionary

        Returns:
            A new LiteLLMModel instance
        """
        logger = _get_module_logger()

        # Validate minimal required fields
        required_fields = ["model_name", "provider"]
        for field in required_fields:
            if field not in model_config:
                raise ValueError(f"Required field '{field}' missing in model configuration for {model_key}")

        model_name = model_config["model_name"]

        # Try to get model info from LiteLLM for auto-detection
        litellm_info = cls._get_model_info_from_litellm(model_name)

        # Auto-detect context_window (input tokens limit)
        if "context_window" not in model_config or model_config["context_window"] is None:
            context_window = litellm_info.get("max_input_tokens")
            if context_window:
                logger.info(f"Auto-detected context_window={context_window} for {model_key} from LiteLLM")
                model_config["context_window"] = context_window
            else:
                raise ValueError(
                    f"context_window not provided for {model_key} and could not be auto-detected from LiteLLM. "
                    f"Please specify context_window in config or ensure model is supported by LiteLLM."
                )

        # Auto-detect max_tokens (default output tokens)
        # Cap at 25% of context window to leave room for prompt and safety margin
        if "max_tokens" not in model_config or model_config["max_tokens"] is None:
            max_output_tokens = litellm_info.get("max_output_tokens") or litellm_info.get("max_tokens")
            if max_output_tokens:
                # If max_output_tokens >= context_window, it's likely wrong (same value used for both)
                # Cap at 25% of context window for safety
                context_win = model_config.get("context_window", 128000)
                safe_max_tokens = min(max_output_tokens, int(context_win * 0.25))
                logger.info(
                    f"Auto-detected max_tokens={safe_max_tokens} for {model_key} (from LiteLLM: {max_output_tokens}, capped at 25% of context)"
                )
                model_config["max_tokens"] = safe_max_tokens
            else:
                raise ValueError(
                    f"max_tokens not provided for {model_key} and could not be auto-detected from LiteLLM. "
                    f"Please specify max_tokens in config or ensure model is supported by LiteLLM."
                )

        # Temperature is required in config
        if "temperature" not in model_config:
            raise ValueError(f"temperature is required in model configuration for {model_key}")

        # Check if model supports reasoning_effort
        reasoning_effort = model_config.get("reasoning_effort")
        if reasoning_effort:
            supported_efforts = ["low", "medium", "high"]
            if reasoning_effort not in supported_efforts:
                logger.warning(f"Invalid reasoning_effort '{reasoning_effort}' for {model_key}. Must be one of {supported_efforts}")
                reasoning_effort = None
            else:
                logger.info(f"Using reasoning_effort={reasoning_effort} for {model_key}")

        # Get LLM compression settings from config
        use_llm_compression = model_config.get("llm_compression", True)
        compression_model = model_config.get("compression_model", "ollama/qwen:1.8b")

        # Get system_prompt from config (optional)
        system_prompt = model_config.get("system_prompt")
        if system_prompt:
            logger.info(f"Using system_prompt for {model_key}: {system_prompt[:100]}...")

        return cls(
            model_key=model_key,
            model_name=model_config["model_name"],
            display_name=model_config.get("display_name") or model_config["model_name"],
            provider=model_config["provider"],
            max_tokens=model_config["max_tokens"],
            temperature=float(model_config["temperature"]),
            context_window=model_config["context_window"],
            reasoning=model_config.get("reasoning", False),
            reasoning_effort=reasoning_effort,
            model_config=model_config,
            use_llm_compression=use_llm_compression,
            compression_model=compression_model,
            system_prompt=system_prompt,
        )


def _check_timeout_exceeded(start_time: float, total_timeout: int, logger: Any | None) -> bool:
    """Check if total timeout has been exceeded."""
    if time.monotonic() - start_time > total_timeout:
        if logger:
            logger.error(f"Total timeout ({total_timeout}s) exceeded. Stopping retries.")
        return True
    return False


async def _handle_retry_response(
    response: ModelResponse,
    attempt: int,
    max_attempts: int,
    current_delay: float,
    start_time: float,
    total_timeout: int,
    initial_delay_val: float,
    max_delay_val: float,
    logger: Any | None,
    provider: str,
) -> float | None:
    """Handle a retry response and return next delay, or None to stop retrying."""
    should_retry, error_type = analyze_error_response(response)
    if not should_retry or attempt >= max_attempts:
        return None

    if logger:
        logger.warning(f"Attempt {attempt}/{max_attempts} failed for {provider}. Retrying... Error: {response.error}")

    next_delay = await _calculate_retry_delay(
        current_delay, start_time, total_timeout, initial_delay_val, max_delay_val, logger, error_type, provider
    )
    return next_delay


async def _handle_retry_exception(
    exception: Exception,
    attempt: int,
    max_attempts: int,
    current_delay: float,
    start_time: float,
    total_timeout: int,
    initial_delay_val: float,
    max_delay_val: float,
    logger: Any | None,
    provider: str,
) -> float | None:
    """Handle a retry exception and return next delay, or None to stop retrying."""
    if attempt >= max_attempts:
        return None

    if logger:
        logger.warning(f"Attempt {attempt}/{max_attempts} failed for {provider} with exception. Retrying... Error: {exception}")

    _, error_type = analyze_error_response(exception)
    next_delay = await _calculate_retry_delay(
        current_delay, start_time, total_timeout, initial_delay_val, max_delay_val, logger, error_type, provider
    )
    return next_delay


async def run_with_retry(
    model: BaseModel,
    prompt: str,
    max_attempts: int = 5,
    initial_delay: int | None = None,
    max_delay: int | None = None,
    total_timeout: int = 900,  # 15 minutes total timeout
    logger: Any | None = None,
) -> ModelResponse:
    """Runs a model generation with retry logic."""
    from ..utils.constants import PROVIDER_RETRY_DELAYS as provider_delays

    provider = model.provider.lower() if model.provider else "default"
    provider_config = provider_delays.get(provider, provider_delays["default"])
    initial_delay_val = initial_delay if initial_delay is not None else provider_config["initial"]
    max_delay_val = max_delay if max_delay is not None else provider_config["max"]

    current_delay: float = float(initial_delay_val)
    start_time = time.monotonic()

    for attempt in range(1, max_attempts + 1):
        if _check_timeout_exceeded(start_time, total_timeout, logger):
            return ModelResponse.create_error(f"Exceeded total timeout of {total_timeout}s")

        try:
            response = await model.generate(prompt)
            if not response.is_error():
                return response

            next_delay = await _handle_retry_response(
                response,
                attempt,
                max_attempts,
                current_delay,
                start_time,
                total_timeout,
                initial_delay_val,
                max_delay_val,
                logger,
                provider,
            )
            if next_delay is None:
                return response
            current_delay = next_delay

        except Exception as e:
            next_delay = await _handle_retry_exception(
                e,
                attempt,
                max_attempts,
                current_delay,
                start_time,
                total_timeout,
                initial_delay_val,
                max_delay_val,
                logger,
                provider,
            )
            if next_delay is None:
                _, error_type = analyze_error_response(e)
                return ModelResponse.create_error(str(e), error_type=error_type, provider=provider)
            current_delay = next_delay

    return ModelResponse.create_error("Max attempts reached without a successful response.")
