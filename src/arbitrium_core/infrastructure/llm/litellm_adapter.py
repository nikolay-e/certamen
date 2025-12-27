"""LiteLLM model implementation for Arbitrium Framework."""

import asyncio
from typing import Any

import litellm

from arbitrium_core.domain.errors import (
    ExceptionClassifier,
    ModelResponseError,
)
from arbitrium_core.infrastructure.config.env import get_ollama_base_url
from arbitrium_core.ports.llm import BaseModel, ModelResponse
from arbitrium_core.shared.constants import (
    DEFAULT_MAX_TOKENS,
    calculate_safe_max_tokens,
)
from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.text.json import to_dict

_logger = get_contextual_logger("arbitrium.models")


def _get_module_logger() -> Any:
    return _logger


class LiteLLMModel(BaseModel):
    def __init__(
        self,
        model_key: str,
        model_name: str,
        display_name: str,
        provider: str,
        temperature: float,
        max_tokens: int = 1024,
        context_window: int | None = None,
        reasoning: bool = False,
        reasoning_effort: str | None = None,
        model_config: dict[str, Any] | None = None,
        use_llm_compression: bool = True,
        compression_model: str | None = None,
        system_prompt: str | None = None,
        response_cache: Any | None = None,
        web_search_options: dict[str, Any] | None = None,
    ):
        super().__init__(
            model_key=model_key,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
            use_llm_compression=use_llm_compression,
            compression_model=compression_model,
        )
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt
        self.response_cache = response_cache
        self.web_search_options = web_search_options

        self.requires_temp_one = (
            model_config is not None
            and hasattr(model_config, "get")
            and model_config.get("force_temp_one", False)
        )

        self.api_base = (
            model_config.get("base_url") if model_config is not None else None
        )

    def _extract_response_content(self, response: Any) -> str | None:
        """Extract content from LiteLLM response.

        LiteLLM normalizes all provider responses to OpenAI format,
        so response.choices[0].message.content always works.

        For reasoning models, content may be in reasoning_content field
        when the model uses all tokens for thinking.
        """
        try:
            message = response.choices[0].message

            content: str | None = getattr(message, "content", None)
            if content and content.strip():
                return content

            reasoning_content: str | None = getattr(
                message, "reasoning_content", None
            )
            if reasoning_content and reasoning_content.strip():
                return reasoning_content

        except (AttributeError, IndexError, KeyError):
            pass

        # Fallback for dict access (legacy compatibility)
        if isinstance(response, dict) and "choices" in response:
            try:
                msg = response["choices"][0]["message"]
                content = msg.get("content")
                if content:
                    return content
                reasoning_content = msg.get("reasoning_content")
                if reasoning_content:
                    return reasoning_content
            except (KeyError, IndexError, TypeError):
                pass

        return None

    def _extract_response_cost(self, response: Any) -> float:
        logger = _get_module_logger()

        if hasattr(response, "_hidden_params") and hasattr(
            response._hidden_params, "response_cost"
        ):
            cost: float = response._hidden_params.response_cost
            logger.info(
                f"Cost extracted from _hidden_params.response_cost: ${cost:.4f}"
            )
            return cost
        if hasattr(response, "response_cost"):
            cost_val: float = response.response_cost
            logger.info(f"Cost extracted from response_cost: ${cost_val:.4f}")
            return cost_val

        if hasattr(response, "usage") and response.usage:
            try:
                if hasattr(litellm, "completion_cost"):
                    cost_calc: float = litellm.completion_cost(
                        completion_response=response
                    )
                    if cost_calc and cost_calc > 0:
                        logger.info(
                            f"Cost calculated via litellm.completion_cost: ${cost_calc:.4f}"
                        )
                        return cost_calc
            except Exception as e:
                logger.debug(
                    f"Failed to calculate cost via litellm.completion_cost: {e}"
                )

        logger.debug("No cost information found in response, returning 0.0")
        return 0.0

    def _handle_prompt_size_validation(self, prompt: str) -> str | None:
        return prompt

    def _clean_response_content(self, content: str) -> str:
        return content.strip()

    def _try_extract_content_from_response(
        self, response: Any, cost: float, logger: Any
    ) -> ModelResponse | None:
        """Extract content from LiteLLM response and create ModelResponse."""
        content = self._extract_response_content(response)

        if content and content.strip():
            cleaned_content = self._clean_response_content(content)
            return ModelResponse.create_success(cleaned_content, cost=cost)

        logger.warning(
            f"{self.display_name} content extraction failed. "
            f"Response type: {type(response)}, has choices: {hasattr(response, 'choices')}"
        )
        return None

    def _validate_prompt(
        self, prompt: str
    ) -> tuple[str, ModelResponse | None]:
        if not prompt or not prompt.strip():
            return "", ModelResponse.create_error("Empty prompt provided")

        validated_prompt = self._handle_prompt_size_validation(prompt)
        if validated_prompt is None:
            method = (
                "LLM compression" if self.use_llm_compression else "truncation"
            )
            return "", ModelResponse.create_error(
                f"Prompt too large even after {method}"
            )

        return validated_prompt, None

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _is_gpt5_model(self) -> bool:
        """Check if the model is GPT-5.x which requires max_completion_tokens."""
        model_lower = self.model_name.lower()
        return any(
            pattern in model_lower
            for pattern in ["gpt-5", "gpt5", "gpt-5.1", "gpt-5.2"]
        )

    def _build_completion_params(
        self, messages: list[dict[str, str]], logger: Any
    ) -> dict[str, Any]:
        # Anthropic requires temperature=1 when extended thinking is enabled
        anthropic_with_reasoning = (
            self.provider == "anthropic" and self.reasoning_effort is not None
        )
        temperature = (
            1.0
            if (self.requires_temp_one or anthropic_with_reasoning)
            else float(self.temperature)
        )

        # Anthropic extended thinking requires max_tokens > budget_tokens
        # Minimum 16000 for extended thinking to work properly
        max_tokens = self.max_tokens
        if anthropic_with_reasoning and max_tokens < 16000:
            max_tokens = 16000
            logger.debug(
                f"Increased max_tokens to {max_tokens} for {self.display_name} (extended thinking requirement)"
            )

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        # GPT-5.x models require max_completion_tokens instead of max_tokens
        if self._is_gpt5_model():
            params["max_completion_tokens"] = max_tokens
            logger.debug(
                f"Using max_completion_tokens={max_tokens} for {self.display_name} (GPT-5.x requirement)"
            )
        else:
            params["max_tokens"] = max_tokens

        if self.api_base:
            params["api_base"] = self.api_base
            logger.debug(
                f"Using api_base={self.api_base} for {self.display_name}"
            )

        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            if anthropic_with_reasoning:
                logger.debug(
                    f"Using reasoning_effort={self.reasoning_effort} for {self.display_name} (temperature forced to 1.0)"
                )
            else:
                logger.debug(
                    f"Using reasoning_effort={self.reasoning_effort} for {self.display_name}"
                )

        if self.web_search_options:
            params["web_search_options"] = self.web_search_options
            logger.debug(
                f"Using web_search_options={self.web_search_options} for {self.display_name}"
            )

        return params

    async def _execute_completion(
        self, params: dict[str, Any], logger: Any
    ) -> ModelResponse:
        import time

        from arbitrium_core.shared.constants import DEFAULT_MODEL_TIMEOUT

        logger.debug(
            f"Executing completion for {self.display_name} "
            f"(timeout={DEFAULT_MODEL_TIMEOUT}s)"
        )

        start_time = time.perf_counter()
        response = await asyncio.wait_for(
            litellm.acompletion(**params), timeout=DEFAULT_MODEL_TIMEOUT
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        cost = self._extract_response_cost(response)

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)

        # Extract rate limit headers if available
        rate_limit_info = ""
        if hasattr(response, "_hidden_params"):
            headers = getattr(response._hidden_params, "headers", {})
            if headers:
                rate_limit_headers = {
                    k: v
                    for k, v in headers.items()
                    if "rate-limit" in k.lower() or "ratelimit" in k.lower()
                }
                if rate_limit_headers:
                    rate_limit_info = f" rate_limits={rate_limit_headers!s}"

        # Structured logging with all metrics
        # nosemgrep: python-logger-credential-disclosure (logging token counts, not secrets)
        logger.info(
            "LLM request completed: model=%s provider=%s tokens=%d (prompt=%d completion=%d) latency_ms=%.0f cost=$%.4f%s",
            self.model_key,
            self.provider,
            total_tokens,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            cost,
            rate_limit_info,
        )

        model_response = self._try_extract_content_from_response(
            response, cost, logger
        )
        if not model_response:
            raise ModelResponseError(
                f"Model {self.display_name} returned empty or unusable response",
                model_key=self.model_key,
            )

        logger.log_response(
            model_response.content,
            model=self.display_name,
            model_key=self.model_key,
            cost=model_response.cost,
        )
        return model_response

    def _handle_exception(self, exc: Exception, logger: Any) -> ModelResponse:
        classification = ExceptionClassifier.classify(
            exc, context=self.display_name
        )

        if classification.is_retryable or isinstance(exc, ModelResponseError):
            logger.warning(
                f"{classification.message}. model={self.model_name}, provider={self.provider}"
            )
        else:
            logger.error(
                f"API error with {self.display_name}: {classification.message}",
                exc_info=True,
            )

        return ModelResponse.create_error(
            classification.message,
            error_type=classification.error_type,
            provider=self.provider,
        )

    def _get_effective_max_tokens(self, params: dict[str, Any]) -> int:
        return int(
            params.get("max_tokens")
            or params.get("max_completion_tokens")
            or DEFAULT_MAX_TOKENS
        )

    def _build_cache_fingerprint(
        self, messages: list[dict[str, str]], params: dict[str, Any]
    ) -> str:
        import json

        fingerprint_data = {
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": self._get_effective_max_tokens(params),
            "api_base": self.api_base,
            "reasoning_effort": self.reasoning_effort,
            "provider": self.provider,
            "web_search_options": self.web_search_options,
        }
        return json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=False)

    async def generate(self, prompt: str) -> ModelResponse:
        logger = _get_module_logger()
        logger.debug(f"Starting generation for {self.display_name}")

        validated_prompt, error = self._validate_prompt(prompt)
        if error:
            return error

        messages = self._build_messages(validated_prompt)
        params = self._build_completion_params(messages, logger)

        effective_temp = float(params["temperature"])
        effective_max_tokens = self._get_effective_max_tokens(params)
        cache_fingerprint = self._build_cache_fingerprint(messages, params)

        if self.response_cache:
            cached = await asyncio.to_thread(
                self.response_cache.get,
                self.model_name,
                cache_fingerprint,
                effective_temp,
                effective_max_tokens,
            )
            if cached:
                content, cost = cached
                logger.info(
                    f"Cache hit for {self.display_name}, saved ${cost:.4f}"
                )
                return ModelResponse.create_success(content, cost=cost)

        logger.log_prompt(
            validated_prompt, model=self.display_name, model_key=self.model_key
        )

        try:
            response = await self._execute_completion(params, logger)
            if self.response_cache and not response.is_error():
                await asyncio.to_thread(
                    self.response_cache.set,
                    self.model_name,
                    cache_fingerprint,
                    effective_temp,
                    effective_max_tokens,
                    response.content,
                    response.cost,
                )
            return response
        except Exception as e:
            return self._handle_exception(e, logger)

    async def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        initial_delay: float | None = None,
        max_delay: float | None = None,
    ) -> ModelResponse:
        from arbitrium_core.infrastructure.llm.retry import run_with_retry

        return await run_with_retry(
            model=self,
            prompt=prompt,
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            logger=_get_module_logger(),
        )

    @classmethod
    def _get_model_info_from_litellm(cls, model_name: str) -> dict[str, Any]:
        logger = _get_module_logger()
        try:
            info = litellm.get_model_info(model_name)
            logger.debug(
                f"Retrieved model info from LiteLLM for {model_name}: {info}"
            )
            return to_dict(info)
        except Exception as e:
            logger.debug(
                f"Could not retrieve model info from LiteLLM for {model_name}: {e}"
            )
            return {}

    @classmethod
    def _validate_required_fields(
        cls, model_key: str, model_config: dict[str, Any]
    ) -> None:
        required_fields = ["model_name", "provider"]
        for field in required_fields:
            if field not in model_config:
                raise ValueError(
                    f"Required field '{field}' missing in model configuration for {model_key}"
                )

    @classmethod
    async def _get_ollama_model_info(
        cls, base_url: str, model_name: str, logger: Any
    ) -> dict[str, Any] | None:
        try:
            import httpx

            clean_model = model_name.replace("ollama/", "")
            url = f"{base_url}/api/show"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json={"name": clean_model})
            if response.status_code == 200:
                data: dict[str, Any] = response.json()
                return data
            else:
                logger.warning(
                    f"Failed to fetch Ollama model info: HTTP {response.status_code}"
                )
                return None
        except Exception as e:
            logger.warning(
                f"Could not fetch Ollama model info from {base_url}: {e}"
            )
            return None

    @classmethod
    async def _auto_detect_context_window(
        cls,
        model_key: str,
        model_config: dict[str, Any],
        litellm_info: dict[str, Any],
        logger: Any,
    ) -> None:
        if (
            "context_window" not in model_config
            or model_config["context_window"] is None
        ):
            context_window = litellm_info.get("max_input_tokens")

            if not context_window and model_config.get("provider") == "ollama":
                base_url = model_config.get("base_url")
                if not base_url:
                    base_url = get_ollama_base_url()
                ollama_info = await cls._get_ollama_model_info(
                    base_url, model_config["model_name"], logger
                )
                if ollama_info and "model_info" in ollama_info:
                    params = ollama_info["model_info"]
                    context_window = params.get("num_ctx") or params.get(
                        "context_length"
                    )
                    if context_window:
                        logger.info(
                            f"Auto-detected context_window={context_window} for {model_key} from Ollama API"
                        )

            if context_window:
                if not model_config.get("provider") == "ollama":
                    logger.info(
                        f"Auto-detected context_window={context_window} for {model_key} from LiteLLM"
                    )
                model_config["context_window"] = context_window
            else:
                default_context = 8192
                logger.warning(
                    f"context_window not provided for {model_key} and could not be auto-detected. "
                    f"Using default value: {default_context}"
                )
                model_config["context_window"] = default_context

    @classmethod
    def _auto_detect_max_tokens(
        cls,
        model_key: str,
        model_config: dict[str, Any],
        litellm_info: dict[str, Any],
        logger: Any,
    ) -> None:
        if (
            "max_tokens" not in model_config
            or model_config["max_tokens"] is None
        ):
            max_output_tokens = litellm_info.get(
                "max_output_tokens"
            ) or litellm_info.get("max_tokens")

            context_win = model_config.get("context_window", 8192)
            safe_max_tokens = calculate_safe_max_tokens(context_win)

            if max_output_tokens:
                safe_max_tokens = min(max_output_tokens, safe_max_tokens)
                logger.info(
                    f"Auto-detected max_tokens={safe_max_tokens} for {model_key} (from LiteLLM: {max_output_tokens}, capped at 25% of context)"
                )
            else:
                logger.warning(
                    f"max_tokens not provided for {model_key} and could not be auto-detected. "
                    f"Using default value: {safe_max_tokens} (25% of context window)"
                )
            model_config["max_tokens"] = safe_max_tokens

    @classmethod
    def _validate_temperature(
        cls, model_key: str, model_config: dict[str, Any]
    ) -> None:
        if "temperature" not in model_config:
            raise ValueError(
                f"temperature is required in model configuration for {model_key}"
            )

    @classmethod
    def _validate_and_get_reasoning_effort(
        cls, model_key: str, model_config: dict[str, Any], logger: Any
    ) -> str | None:
        reasoning_effort = model_config.get("reasoning_effort")
        if reasoning_effort:
            supported_efforts = ["low", "medium", "high"]
            if reasoning_effort not in supported_efforts:
                logger.warning(
                    f"Invalid reasoning_effort '{reasoning_effort}' for {model_key}. Must be one of {supported_efforts}"
                )
                return None
            logger.info(
                f"Using reasoning_effort={reasoning_effort} for {model_key}"
            )
        return reasoning_effort

    @classmethod
    def _get_compression_settings(
        cls, model_config: dict[str, Any]
    ) -> tuple[bool, str | None]:
        use_llm_compression = model_config.get("llm_compression", True)
        compression_model = model_config.get("compression_model", None)
        return use_llm_compression, compression_model

    @classmethod
    def _get_and_log_system_prompt(
        cls, model_key: str, model_config: dict[str, Any], logger: Any
    ) -> str | None:
        system_prompt = model_config.get("system_prompt")
        if system_prompt:
            logger.info(
                f"Using system_prompt for {model_key}: {system_prompt[:100]}..."
            )
        return system_prompt

    @classmethod
    async def from_config(
        cls,
        model_key: str,
        model_config: dict[str, Any],
        response_cache: Any | None = None,
    ) -> "LiteLLMModel":
        logger = _get_module_logger()

        cls._validate_required_fields(model_key, model_config)

        model_name = model_config["model_name"]
        litellm_info = cls._get_model_info_from_litellm(model_name)

        await cls._auto_detect_context_window(
            model_key, model_config, litellm_info, logger
        )
        cls._auto_detect_max_tokens(
            model_key, model_config, litellm_info, logger
        )
        cls._validate_temperature(model_key, model_config)

        reasoning_effort = cls._validate_and_get_reasoning_effort(
            model_key, model_config, logger
        )
        use_llm_compression, compression_model = cls._get_compression_settings(
            model_config
        )
        system_prompt = cls._get_and_log_system_prompt(
            model_key, model_config, logger
        )

        web_search_options = model_config.get("web_search_options")
        if web_search_options:
            logger.info(
                f"Web search enabled for {model_key}: {web_search_options}"
            )

        return cls(
            model_key=model_key,
            model_name=model_config["model_name"],
            display_name=model_config.get("display_name")
            or model_config["model_name"],
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
            response_cache=response_cache,
            web_search_options=web_search_options,
        )
