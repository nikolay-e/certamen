"""Retry logic and error analysis for model API calls."""

import asyncio
import random
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arbitrium_core.ports.llm import ModelResponse

from arbitrium_core.shared.constants import (
    ERROR_PATTERNS,
    PERMISSION_ERROR_PATTERNS,
    RETRYABLE_ERROR_TYPES,
)
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


def _is_retryable_error_type(error_type: str) -> bool:
    return error_type in RETRYABLE_ERROR_TYPES


def _check_exception_type(response: Exception) -> tuple[bool, str] | None:
    exception_name = type(response).__name__.lower()

    if "notfounderror" in exception_name:
        return False, "not_found"

    if "authenticationerror" in exception_name:
        return False, "authentication"

    return None


def _check_permission_errors(error_msg: str) -> bool:
    return any(p.lower() in error_msg for p in PERMISSION_ERROR_PATTERNS)


def _match_error_patterns(error_msg: str) -> tuple[bool, str] | None:
    for error_type, patterns in ERROR_PATTERNS.items():
        if any(p.lower() in error_msg for p in patterns):
            return True, error_type
    return None


def analyze_error_response(response: Any) -> tuple[bool, str]:
    error_msg = ""

    if hasattr(response, "error"):
        error_msg = str(response.error).lower()
        error_type = getattr(response, "error_type", None)
        if error_type:
            return _is_retryable_error_type(error_type), error_type

    if isinstance(response, Exception):
        error_msg = str(response).lower()
        exception_result = _check_exception_type(response)
        if exception_result:
            return exception_result

    if _check_permission_errors(error_msg):
        return False, "permission_denied"

    pattern_result = _match_error_patterns(error_msg)
    if pattern_result:
        return pattern_result

    return False, "general"


_BACKOFF_MULTIPLIERS = {
    "rate_limit": {"anthropic": 2.5, "default": 2.0},
    "overloaded": {"anthropic": 3.0, "default": 2.5},
    "timeout": 1.5,
    "connection": 1.8,
    "service": 2.0,
    "general": 1.5,
}


def _get_backoff_multiplier(error_type: str, provider: str) -> float:
    provider = provider.lower() if provider else "default"
    multiplier_value = _BACKOFF_MULTIPLIERS.get(
        error_type, _BACKOFF_MULTIPLIERS["general"]
    )

    if isinstance(multiplier_value, dict):
        provider_mult = multiplier_value.get(
            provider, multiplier_value["default"]
        )
        return float(provider_mult) if provider_mult is not None else 1.5

    if isinstance(multiplier_value, (int, float)):
        return float(multiplier_value)

    return 1.5


def _get_jitter_range(error_type: str, provider: str) -> float:
    if error_type in ["rate_limit", "overloaded"] and provider == "anthropic":
        return 0.05
    return 0.1


def _calculate_jittered_delay(
    current_delay: float,
    max_delay: float,
    multiplier: float,
    jitter_range: float,
) -> float:
    jitter_factor = 1.0 + random.uniform(-jitter_range, jitter_range)
    return min(current_delay * multiplier, max_delay) * jitter_factor


def _check_timeout_remaining(
    start_time: float,
    total_timeout: float,
    logger: Any | None,
) -> float | None:
    elapsed_time = time.monotonic() - start_time
    remaining_time = total_timeout - elapsed_time
    if remaining_time <= 0:
        if logger:
            logger.error(
                "No time left for retry. Elapsed: %.2fs, total timeout: %.2fs. Stopping retries.",
                elapsed_time,
                total_timeout,
            )
        return None
    return remaining_time


def _check_min_delay(
    actual_delay: float,
    initial_delay: float,
    error_type: str,
    logger: Any | None,
) -> bool:
    min_delay_factor = (
        0.5 if error_type in ["rate_limit", "overloaded"] else 0.25
    )
    min_required_delay = initial_delay * min_delay_factor
    if actual_delay < min_required_delay:
        if logger:
            logger.error(
                "Not enough time for proper retry delay. Required: %.2fs (%.0f%% of initial), available: %.2fs. Error type: %s. Stopping retries.",
                min_required_delay,
                min_delay_factor * 100,
                actual_delay,
                error_type,
            )
        return False
    return True


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
    multiplier = _get_backoff_multiplier(error_type, provider)
    jitter_range = _get_jitter_range(error_type, provider)
    jittered_delay = _calculate_jittered_delay(
        current_delay, max_delay, multiplier, jitter_range
    )

    remaining_time = _check_timeout_remaining(
        start_time, total_timeout, logger
    )
    if remaining_time is None:
        return None

    actual_delay = min(jittered_delay, remaining_time)
    if not _check_min_delay(actual_delay, initial_delay, error_type, logger):
        return None

    await asyncio.sleep(actual_delay)
    return min(current_delay * multiplier, max_delay)


def _check_timeout_exceeded(
    start_time: float, total_timeout: int, logger: Any | None
) -> bool:
    elapsed_time = time.monotonic() - start_time
    if elapsed_time > total_timeout:
        if logger:
            logger.error(
                "Total timeout (%ds) exceeded. Elapsed: %.2fs. Stopping retries.",
                total_timeout,
                elapsed_time,
            )
        return True
    return False


def _extract_error_info(error_or_exception: Any) -> tuple[str, str]:
    is_exception = isinstance(error_or_exception, Exception)
    if is_exception:
        error_label = type(error_or_exception).__name__
        error_message = str(error_or_exception)[:200]
    else:
        error_label = "error"
        error_message = str(
            getattr(error_or_exception, "error", error_or_exception)
        )[:200]
    return error_label, error_message


async def _handle_retry_error(
    error_or_exception: Any,
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
    should_retry, error_type = analyze_error_response(error_or_exception)
    elapsed_time = time.monotonic() - start_time
    error_label, error_message = _extract_error_info(error_or_exception)
    is_exception = isinstance(error_or_exception, Exception)

    if not should_retry:
        if logger:
            if is_exception:
                logger.warning(
                    "Attempt %d/%d failed with non-retryable exception for %s. Exception: %s. Error type: %s. Elapsed: %.2fs",
                    attempt,
                    max_attempts,
                    provider,
                    error_label,
                    error_type,
                    elapsed_time,
                    error_classification="non-retryable",
                    error_type_detail=error_type,
                )
            else:
                logger.warning(
                    "Attempt %d/%d failed with non-retryable error for %s. Error type: %s. Elapsed: %.2fs",
                    attempt,
                    max_attempts,
                    provider,
                    error_type,
                    elapsed_time,
                    error_classification="non-retryable",
                    error_type_detail=error_type,
                )
        return None

    if attempt >= max_attempts:
        if logger:
            if is_exception:
                logger.error(
                    "Attempt %d/%d failed for %s with exception. Max attempts reached. Exception: %s. Error type: %s. Total elapsed: %.2fs",
                    attempt,
                    max_attempts,
                    provider,
                    error_label,
                    error_type,
                    elapsed_time,
                    error_classification="retryable",
                    error_type_detail=error_type,
                )
            else:
                logger.error(
                    "Attempt %d/%d failed for %s. Max attempts reached. Error type: %s. Total elapsed: %.2fs",
                    attempt,
                    max_attempts,
                    provider,
                    error_type,
                    elapsed_time,
                    error_classification="retryable",
                    error_type_detail=error_type,
                )
        return None

    multiplier = _get_backoff_multiplier(error_type, provider)
    jitter_range = _get_jitter_range(error_type, provider)
    next_delay_base = min(current_delay * multiplier, max_delay_val)
    jitter_factor = 1.0 + random.uniform(-jitter_range, jitter_range)
    planned_delay = next_delay_base * jitter_factor

    if logger:
        if is_exception:
            logger.warning(
                "Attempt %d/%d failed for %s with exception. Retrying after %.2fs delay (base: %.2fs, multiplier: %.2fx, jitter: %.1f%%). Exception: %s. Error type: %s. Elapsed: %.2fs. Error: %s",
                attempt,
                max_attempts,
                provider,
                planned_delay,
                next_delay_base,
                multiplier,
                (jitter_factor - 1.0) * 100,
                error_label,
                error_type,
                elapsed_time,
                error_message,
                error_classification="retryable",
                error_type_detail=error_type,
                retry_delay=planned_delay,
            )
        else:
            logger.warning(
                "Attempt %d/%d failed for %s. Retrying after %.2fs delay (base: %.2fs, multiplier: %.2fx, jitter: %.1f%%). Error type: %s. Elapsed: %.2fs. Error: %s",
                attempt,
                max_attempts,
                provider,
                planned_delay,
                next_delay_base,
                multiplier,
                (jitter_factor - 1.0) * 100,
                error_type,
                elapsed_time,
                error_message,
                error_classification="retryable",
                error_type_detail=error_type,
                retry_delay=planned_delay,
            )

    return await _calculate_retry_delay(
        current_delay,
        start_time,
        total_timeout,
        initial_delay_val,
        max_delay_val,
        logger,
        error_type,
        provider,
    )


async def run_with_retry(
    model: Any,
    prompt: str,
    max_attempts: int = 5,
    initial_delay: float | None = None,
    max_delay: float | None = None,
    total_timeout: int = 3600,
    logger: Any | None = None,
) -> "ModelResponse":
    from arbitrium_core.ports.llm import ModelResponse
    from arbitrium_core.shared.constants import (
        PROVIDER_RETRY_DELAYS as provider_delays,
    )

    provider = model.provider.lower() if model.provider else "default"
    provider_config = provider_delays.get(provider, provider_delays["default"])
    initial_delay_val = (
        initial_delay
        if initial_delay is not None
        else provider_config["initial"]
    )
    max_delay_val = (
        max_delay if max_delay is not None else provider_config["max"]
    )

    current_delay: float = float(initial_delay_val)
    start_time = time.monotonic()

    if logger:
        logger.debug(
            "Starting retry loop for %s. Max attempts: %d, initial delay: %.2fs, max delay: %.2fs, total timeout: %ds",
            provider,
            max_attempts,
            initial_delay_val,
            max_delay_val,
            total_timeout,
        )

    for attempt in range(1, max_attempts + 1):
        if _check_timeout_exceeded(start_time, total_timeout, logger):
            return ModelResponse.create_error(
                f"Exceeded total timeout of {total_timeout}s"
            )

        try:
            response = await model.generate(prompt)
            if not response.is_error():
                elapsed_time = time.monotonic() - start_time
                if logger and attempt > 1:
                    logger.info(
                        "Retry successful for %s on attempt %d/%d. Total elapsed: %.2fs",
                        provider,
                        attempt,
                        max_attempts,
                        elapsed_time,
                        successful_attempt=attempt,
                    )
                return response  # type: ignore[no-any-return]

            next_delay = await _handle_retry_error(
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
                return response  # type: ignore[no-any-return]
            current_delay = next_delay

        except Exception as e:
            next_delay = await _handle_retry_error(
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
                return ModelResponse.create_error(
                    str(e), error_type=error_type, provider=provider
                )
            current_delay = next_delay

    elapsed_time = time.monotonic() - start_time
    if logger:
        logger.error(
            "All retry attempts exhausted for %s. Max attempts: %d. Total elapsed: %.2fs",
            provider,
            max_attempts,
            elapsed_time,
        )
    return ModelResponse.create_error(
        "Max attempts reached without a successful response."
    )
