"""
Structured logging wrapper for Arbitrium Framework.

This module provides consistent structured logging functions
that follow a single unified approach.
"""

import logging
import os
import sys
from pathlib import Path
from typing import ClassVar

from colorama import Fore, Style

from arbitrium.utils.constants import DEFAULT_LOG_CACHE_SIZE


class DuplicateFilter(logging.Filter):
    """Filter to prevent duplicate log messages."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_messages: set[str] = set()
        self.max_cache_size = DEFAULT_LOG_CACHE_SIZE


class ColorFormatter(logging.Formatter):
    """A logging formatter that adds color to log output and handles special display types."""

    # ANSI color codes for log levels
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    # Model colors (imported from display.py)
    MODEL_COLORS: ClassVar[dict[str, str]] = {}

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, include_module: bool = True) -> None:
        """Initialize the formatter with format string."""
        super().__init__(fmt, datefmt)
        self.include_module = include_module
        # Import model colors from display module
        from arbitrium.utils.display import MODEL_COLORS

        self.MODEL_COLORS = MODEL_COLORS  # type: ignore[misc]

    def format(self, record: logging.LogRecord) -> str:
        """Add color and special formatting to the record."""
        # Check for special display types
        display_type = getattr(record, "display_type", None)

        # More robust color capability detection
        should_colorize = self._should_use_color()

        if display_type:
            return self._format_display(record, display_type, should_colorize)

        level_name = record.levelname

        # Get the original formatted message first
        original_msg = super().format(record)

        # Build context parts from record attributes (set by ContextFilter)
        from arbitrium.logging.structured import build_context_parts

        context_parts = build_context_parts(record)

        # Add module info if enabled
        if self.include_module and hasattr(record, "module"):
            context_parts.append(f"{record.module}")

        # Insert context after [LEVEL] if present
        if context_parts and "[" in original_msg:
            # Find the end of the level marker
            level_end = original_msg.find("]") + 1
            context_prefix = "[" + "] [".join(context_parts) + "] "
            original_msg = original_msg[:level_end] + " " + context_prefix + original_msg[level_end:].lstrip()
        elif context_parts:
            context_prefix = "[" + "] [".join(context_parts) + "] "
            original_msg = context_prefix + original_msg

        if should_colorize:
            color = self.COLORS.get(level_name, "")
            reset = Style.RESET_ALL
            return f"{color}{original_msg}{reset}"
        else:
            # Remove any existing ANSI color codes to ensure clean output in non-color environments
            from arbitrium.utils.terminal import strip_ansi_codes

            return strip_ansi_codes(original_msg)

    def _format_display(self, record: logging.LogRecord, display_type: str, should_colorize: bool) -> str:
        """Format special display types (headers, model responses, etc)."""
        message = record.getMessage()

        if display_type == "section_header":
            # Format: \n--- Message ---
            if should_colorize:
                return f"\n{Fore.CYAN}--- {message} ---{Style.RESET_ALL}"
            return f"\n--- {message} ---"

        elif display_type == "header":
            # Format: \n========== Message ==========
            border = "=" * 50
            if should_colorize:
                return f"\n{Fore.CYAN}{border}\n{message}\n{border}{Style.RESET_ALL}"
            return f"\n{border}\n{message}\n{border}"

        elif display_type == "model_response":
            # Format model response with color
            model_name = getattr(record, "model_name", "Unknown")
            color = self.MODEL_COLORS.get(model_name, Fore.WHITE) if should_colorize else ""
            reset = Style.RESET_ALL if should_colorize else ""

            lines = [f"\n{color}Model: {model_name}{reset}", f"{color}Response:{reset}"]
            for line in message.split("\n"):
                lines.append(f"{color}{line}{reset}")
            lines.append(f"{color}{'-' * 20}{reset}")
            return "\n".join(lines)

        elif display_type == "colored_text":
            # Format text with specific color
            color_name = getattr(record, "color", None)
            if should_colorize and color_name:
                color = self.MODEL_COLORS.get(color_name, Fore.WHITE)
                return f"{color}{message}{Style.RESET_ALL}"
            return message

        # Fallback to regular formatting
        result: str = super().format(record)
        return result

    def _should_use_color(self) -> bool:
        """
        Determine if color should be used based on terminal capabilities and environment.

        Returns:
            Boolean indicating whether colors should be used.
        """
        # Use the centralized terminal utility function for color detection
        from arbitrium.utils.terminal import should_use_color

        return should_use_color()


def _validate_log_file_path(log_file: str | None) -> str | None:
    """Validate and prepare log file path, return None if invalid."""
    if not log_file:
        return None

    try:
        log_path = Path(log_file)
        log_dir = log_path.parent

        # Ensure the log directory exists and is writable
        if log_dir:
            # Create directory if it doesn't exist
            if not log_dir.exists():
                os.makedirs(log_dir, exist_ok=True)

            # Check if directory is writable
            if not os.access(log_dir, os.W_OK):
                print(f"Warning: Log directory {log_dir} is not writable. Logs will not be saved to file.")
                return None
        return log_file
    except OSError as e:
        print(f"Error with log file path: {e!s}")
        return None
    except Exception as e:
        print(f"Unexpected error with log file: {e!s}")
        return None


def _create_file_handler(
    log_file: str,
    file_format: str,
    file_duplicate_filter: DuplicateFilter,
    context_filter: logging.Filter,
    json_format: bool = False,
    include_module: bool = True,
) -> logging.FileHandler | None:
    """Create and test file handler, return None if it fails."""
    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Save all logs to file regardless of level

        # Choose formatter based on format preference
        if json_format:
            from arbitrium.logging.structured import JSONFormatter

            file_handler.setFormatter(JSONFormatter())
        elif include_module:
            from arbitrium.logging.structured import StructuredFormatter

            file_handler.setFormatter(StructuredFormatter(file_format, include_module=include_module))
        else:
            file_handler.setFormatter(logging.Formatter(file_format))

        file_handler.addFilter(file_duplicate_filter)
        file_handler.addFilter(context_filter)

        # Test the file handler by writing a dummy log message
        try:
            test_record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Testing file handler",
                args=(),
                exc_info=None,
            )
            file_handler.emit(test_record)
        except Exception as test_error:
            raise OSError(f"Error writing to log file: {test_error!s}") from test_error

        return file_handler
    except OSError as file_error:
        print(f"❌ CRITICAL ERROR: Cannot open log file {log_file}: {file_error!s}")
        print("❌ MODEL RESPONSES WILL NOT BE SAVED TO FILE!")
        print("❌ This means expensive API calls could be lost!")
        print("Continuing with console logging only.")
        return None
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Unexpected error setting up file logging: {e!s}")
        print("❌ MODEL RESPONSES WILL NOT BE SAVED TO FILE!")
        print("❌ This means expensive API calls could be lost!")
        print("Continuing with console logging only.")
        return None


def _configure_litellm() -> None:
    """Configure LiteLLM to suppress verbose logging."""
    # Set environment variables for complete LiteLLM suppression BEFORE import
    os.environ["LITELLM_LOG"] = "CRITICAL"
    os.environ["LITELLM_VERBOSE"] = "False"
    os.environ["LITELLM_DROP_PARAMS"] = "True"
    os.environ["LITELLM_SUPPRESS_DEBUG"] = "True"
    os.environ["LITELLM_SUCCESS_CALLBACK"] = "[]"
    os.environ["LITELLM_FAILURE_CALLBACK"] = "[]"

    # Globally configure LiteLLM before any third-party logging setup
    try:
        import litellm

        # Configure LiteLLM suppression globally using the comprehensive approach
        litellm.suppress_debug_info = True
        litellm.drop_params = True
        litellm.set_verbose = False

        # Suppress LiteLLM's own logger
        litellm_logger = logging.getLogger("LiteLLM")
        litellm_logger.setLevel(logging.CRITICAL)
        litellm_logger.propagate = False
    except ImportError:
        pass  # LiteLLM not installed, skip configuration


def setup_logging(
    log_file: str | None = None,
    level: int = logging.INFO,
    debug: bool = False,
    verbose: bool = False,
    enable_file_logging: bool = True,
    json_format: bool = False,
    include_module: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file. If None and enable_file_logging is True, creates timestamped log.
        level: The logging level
        debug: If True, sets level to DEBUG regardless of level parameter
        verbose: If True, shows all INFO messages. If False, shows only major phase headers.
        enable_file_logging: If True and log_file is None, creates a timestamped log file in current directory.
        json_format: If True, use JSON format for file logging (console stays human-readable).
        include_module: If True, include module/file information in logs.

    Returns:
        Configured logger
    """
    # Generate timestamped log file if not specified but file logging is enabled
    if log_file is None and enable_file_logging:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"arbitrium_{timestamp}_logs.log"

    # Validate log file path
    log_file = _validate_log_file_path(log_file)

    # Set level to DEBUG if debug flag is set
    if debug:
        level = logging.DEBUG

    # Configure root logger
    # IMPORTANT: Always set root logger to DEBUG so file handler can capture everything
    # Individual handlers control what they output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Import ContextFilter - always needed for context injection
    from arbitrium.logging.structured import ContextFilter

    # Import structured formatters if needed
    if json_format or include_module:
        pass

    # Format strings
    console_format = "[%(levelname)s] %(message)s"  # Cleaner console output
    file_format = "%(asctime)s [%(levelname)s] %(message)s"  # Full timestamp for files

    # Create separate duplicate filters for console and file
    console_duplicate_filter = DuplicateFilter()
    file_duplicate_filter = DuplicateFilter()

    # Create context filter for structured logging
    context_filter = ContextFilter()

    # Determine console level - quiet mode for clean output unless verbose
    console_level = level if verbose or debug else logging.WARNING

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    # Always use ColorFormatter for console (human-readable)
    console_handler.setFormatter(ColorFormatter(console_format, include_module=include_module))
    console_handler.addFilter(console_duplicate_filter)
    console_handler.addFilter(context_filter)

    # Initialize file handler early for potential exceptions
    file_handler = None

    # Add file handler if specified
    if log_file:
        file_handler = _create_file_handler(
            log_file,
            file_format,
            file_duplicate_filter,
            context_filter,
            json_format=json_format,
            include_module=include_module,
        )

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
        # Log first message to verify file handler works
        root_logger.info(f"Log file initialized at {log_file}")

    # Configure LiteLLM suppression
    _configure_litellm()

    # Create a specific logger for Arbitrium Framework
    # IMPORTANT: Set to DEBUG so all messages reach handlers, handlers control what gets output
    logger = logging.getLogger("arbitrium")
    logger.setLevel(logging.DEBUG)
    # Ensure the arbitrium logger propagates messages to the root logger
    logger.propagate = True

    # Set appropriate levels for third-party loggers
    # Avoid completely silencing them, but keep reasonable levels
    third_party_loggers = {
        # Core API request libraries - allow warnings
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        # LLM-related libraries - silence completely in debug mode to avoid spam
        "litellm": logging.ERROR,
        "openai": logging.WARNING,
        "anthropic": logging.WARNING,
        "google": logging.WARNING,
        "vertexai": logging.WARNING,
        # Other system libraries
        "asyncio": logging.WARNING,
    }

    # Apply the configured log levels and ensure propagation
    for logger_name, log_level in third_party_loggers.items():
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(log_level)
        # For LiteLLM, disable propagation completely to prevent debug spam
        if logger_name == "litellm":
            third_party_logger.propagate = False
        else:
            # Ensure other third-party loggers propagate messages to the root logger
            third_party_logger.propagate = True

    # Return the configured logger
    return logger
