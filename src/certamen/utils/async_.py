"""
Asynchronous utility functions for Certamen Framework.
"""

import asyncio
import atexit
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from certamen.logging import get_contextual_logger

from .constants import DEFAULT_INPUT_TIMEOUT, DEFAULT_THREAD_POOL_WORKERS

# Executor for running blocking I/O operations
_executor = ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_WORKERS)

# Module-level logger
logger = get_contextual_logger("certamen.utils.async_")


# Register the executor to be cleaned up on exit
def _shutdown_executor() -> None:
    if not _executor._shutdown:
        _executor.shutdown(wait=True)


atexit.register(_shutdown_executor)


def _validate_default_value(
    default: str,
    validation_func: Callable[[str], bool] | None,
    min_length: int,
    max_length: int | None,
    logger: Any,
) -> str:
    if not default:
        return default

    is_valid = True
    if validation_func is not None and not validation_func(default):
        logger.warning(
            f"Default value '{default}' does not pass validation function."
        )
        is_valid = False

    if min_length > 0 and len(default) < min_length:
        logger.warning(
            f"Default value length ({len(default)}) is less than min_length ({min_length})."
        )
        is_valid = False

    if max_length is not None and len(default) > max_length:
        logger.warning(
            f"Default value length ({len(default)}) is greater than max_length ({max_length})."
        )
        is_valid = False

    if not is_valid:
        logger.error(
            "Invalid default value provided. Using empty string as fallback."
        )
        return ""
    return default


def _try_set_future_result(input_future: Any, value: str, logger: Any) -> bool:
    try:
        if not input_future.done():
            input_future.set_result(value)
            return True
    except asyncio.InvalidStateError:
        logger.debug("Input future already done when trying to set result")
    return False


def _try_set_future_exception(
    input_future: Any, exception: Exception, logger: Any
) -> None:
    """Try to set future exception."""
    try:
        if not input_future.done():
            input_future.set_exception(exception)
    except asyncio.InvalidStateError:
        logger.debug(
            f"Input future already done when exception occurred: {exception!s}"
        )


def _check_input_validation(
    user_input: str,
    min_length: int,
    max_length: int | None,
    validation_func: Callable[[str], bool] | None,
    validation_message: str,
) -> bool:
    """Check if user input passes all validation criteria."""
    # Check minimum length
    if min_length > 0 and len(user_input) < min_length:
        print(f"Input must be at least {min_length} characters long.")
        return False

    # Check maximum length
    if max_length is not None and len(user_input) > max_length:
        print(f"Input must be at most {max_length} characters long.")
        return False

    # Apply custom validation function if provided
    if validation_func is not None and not validation_func(user_input):
        print(validation_message)
        return False

    return True


async def _get_input_with_validation(
    input_future: Any,
    prompt: str,
    min_length: int,
    max_length: int | None,
    validation_func: Callable[[str], bool] | None,
    validation_message: str,
    logger: Any,
) -> None:
    """Get user input with validation in a loop."""
    while True:
        try:
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(_executor, input, prompt)

            # Check if input passes validation criteria
            is_valid = _check_input_validation(
                user_input,
                min_length,
                max_length,
                validation_func,
                validation_message,
            )

            # If input is valid, try to set result
            if is_valid:
                if _try_set_future_result(input_future, user_input, logger):
                    return
                return
            else:
                # Check if timeout occurred during validation
                if input_future.done():
                    logger.debug("Input future done during validation retry")
                    return
                continue

        except Exception as e:
            from .exceptions import InputError

            _try_set_future_exception(
                input_future,
                InputError(f"Error getting user input: {e!s}"),
                logger,
            )
            return


async def _handle_input_timeout(
    input_future: Any, timeout: int, default: str, logger: Any
) -> None:
    """Handle input timeout by setting default value."""
    await asyncio.sleep(timeout)

    if _try_set_future_result(input_future, default, logger):
        logger.warning(
            f"Input timed out after {timeout} seconds. Using default value."
        )
        print(f"\nInput timed out. Using default: '{default}'")


def _check_non_interactive_environment(
    prompt: str, default: str, logger: Any
) -> str | None:
    """Check if environment is non-interactive and return default if so."""
    if not sys.stdin.isatty():
        logger.warning(
            "Non-interactive environment detected (stdin is not a TTY). Using default input value."
        )
        print(f"{prompt} [Non-interactive mode, using default: '{default}']")
        return default
    return None


async def async_input(
    prompt: str = "",
    default: str = "",
    timeout: int = DEFAULT_INPUT_TIMEOUT,
    validation_func: Callable[[str], bool] | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    validation_message: str = "Input validation failed. Please try again.",
) -> str:
    """
    Non-blocking version of the built-in input() function with
    proper handling for non-interactive environments and input validation.

    Args:
        prompt: The prompt to display to the user
        default: Default value to return in non-interactive environments or on timeout
        timeout: Maximum time to wait for input in seconds
        validation_func: Optional function to validate user input
        min_length: Minimum length for valid input (0 means no minimum)
        max_length: Maximum length for valid input (None means no maximum)
        validation_message: Message to display if validation fails

    Returns:
        The text entered by the user or the default value
    """
    # Check if stdin is a TTY
    non_interactive_result = _check_non_interactive_environment(
        prompt, default, logger
    )
    if non_interactive_result is not None:
        return non_interactive_result

    # Validate default value for consistency
    default = _validate_default_value(
        default, validation_func, min_length, max_length, logger
    )

    # Create a future for the input operation
    loop = asyncio.get_running_loop()
    input_future = loop.create_future()

    # Start input task
    input_task = asyncio.create_task(
        _get_input_with_validation(
            input_future,
            prompt,
            min_length,
            max_length,
            validation_func,
            validation_message,
            logger,
        )
    )
    # Keep reference to prevent garbage collection
    input_task.add_done_callback(lambda _: None)

    # Start timeout task if needed
    if timeout > 0:
        timeout_task = asyncio.create_task(
            _handle_input_timeout(input_future, timeout, default, logger)
        )
        # Keep reference to prevent garbage collection
        timeout_task.add_done_callback(lambda _: None)

    # Wait for either valid input or timeout
    try:
        result: str = await input_future
        return result
    except Exception as e:
        logger.error(f"Input error: {e!s}")
        return default
