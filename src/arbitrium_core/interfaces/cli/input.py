import asyncio
import atexit
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from arbitrium_core.shared.constants import (
    DEFAULT_INPUT_TIMEOUT,
    DEFAULT_THREAD_POOL_WORKERS,
)
from arbitrium_core.shared.logging import get_contextual_logger

_executor = ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_WORKERS)

logger = get_contextual_logger("arbitrium.interfaces.cli.input")


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
    if min_length > 0 and len(user_input) < min_length:
        print(f"Input must be at least {min_length} characters long.")
        return False

    if max_length is not None and len(user_input) > max_length:
        print(f"Input must be at most {max_length} characters long.")
        return False

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
    while True:
        try:
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(_executor, input, prompt)

            is_valid = _check_input_validation(
                user_input,
                min_length,
                max_length,
                validation_func,
                validation_message,
            )

            if is_valid:
                _try_set_future_result(input_future, user_input, logger)
                return
            else:
                if input_future.done():
                    logger.debug("Input future done during validation retry")
                    return
                continue

        except Exception as e:
            from arbitrium_core.domain.errors import InputError

            _try_set_future_exception(
                input_future,
                InputError(f"Error getting user input: {e!s}"),
                logger,
            )
            return


async def _handle_input_timeout(
    input_future: Any, timeout: int, default: str, logger: Any
) -> None:
    await asyncio.sleep(timeout)

    if _try_set_future_result(input_future, default, logger):
        logger.warning(
            f"Input timed out after {timeout} seconds. Using default value."
        )
        print(f"\nInput timed out. Using default: '{default}'")


def _check_non_interactive_environment(
    prompt: str, default: str, logger: Any
) -> str | None:
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
    Async input with timeout and validation.

    Note: Uses blocking input() in a thread pool. When timeout triggers,
    the default value is returned but the underlying input() call remains
    blocking until the user presses Enter. This is a known limitation of
    using blocking input() in a thread - the terminal may appear "stuck"
    on the old prompt until Enter is pressed.
    """
    non_interactive_result = _check_non_interactive_environment(
        prompt, default, logger
    )
    if non_interactive_result is not None:
        return non_interactive_result

    default = _validate_default_value(
        default, validation_func, min_length, max_length, logger
    )

    loop = asyncio.get_running_loop()
    input_future = loop.create_future()

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
    input_task.add_done_callback(lambda _: None)

    if timeout > 0:
        timeout_task = asyncio.create_task(
            _handle_input_timeout(input_future, timeout, default, logger)
        )
        timeout_task.add_done_callback(lambda _: None)

    try:
        result: str = await input_future
        return result
    except Exception as e:
        logger.error(f"Input error: {e!s}")
        return default
