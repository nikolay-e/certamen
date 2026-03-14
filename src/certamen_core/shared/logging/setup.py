import logging
import os
import re
import sys
from pathlib import Path
from typing import ClassVar

from colorama import Fore, Style

from certamen_core.shared.constants import DEFAULT_LOG_CACHE_SIZE


class SensitiveDataFilter(logging.Filter):
    PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        # OpenAI API keys: sk-... (48-51 chars)
        (re.compile(r"\bsk-[A-Za-z0-9]{20,50}\b"), "[REDACTED_OPENAI_KEY]"),
        # Anthropic API keys: sk-ant-...
        (
            re.compile(r"\bsk-ant-[A-Za-z0-9-]{20,100}\b"),
            "[REDACTED_ANTHROPIC_KEY]",
        ),
        # Google API keys: AIza...
        (
            re.compile(r"\bAIza[A-Za-z0-9_-]{30,40}\b"),
            "[REDACTED_GOOGLE_KEY]",
        ),
        # xAI API keys: xai-...
        (
            re.compile(r"\bxai-[A-Za-z0-9_-]{20,100}\b"),
            "[REDACTED_XAI_KEY]",
        ),
        # Generic API keys in common formats
        (
            re.compile(
                r"\b[A-Za-z0-9]{32,64}\b(?=.*(?:key|token|secret))", re.I
            ),
            "[REDACTED_KEY]",
        ),
        # JWT tokens (header.payload.signature)
        (
            re.compile(
                r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
            ),
            "[REDACTED_JWT]",
        ),
        # Bearer tokens in headers
        (
            re.compile(r"(?i)bearer\s+[A-Za-z0-9_.+-]+"),
            "Bearer [REDACTED_TOKEN]",
        ),
        # Password patterns in key=value, key:value, "key":"value"
        (
            re.compile(
                r"(?i)(password|passwd|pwd|secret|api_key|apikey|auth_token|access_token|refresh_token)"
                r'[\s]*[=:]\s*["\']?([^"\'\s,\}]{4,})["\']?'
            ),
            r"\1=[REDACTED]",
        ),
        # Database URLs with credentials (e.g. postgresql://...:...@host)
        (  # pragma: allowlist secret
            re.compile(
                r"(?i)(postgresql|postgres|mysql|mongodb|redis)://[^:]+:([^@]+)@"
            ),
            r"\1://[user]:[REDACTED]@",
        ),
        # Basic auth in URLs (e.g. https://...:...@host)
        (  # pragma: allowlist secret
            re.compile(r"(https?://)[^:]+:([^@]+)@"),
            r"\1[user]:[REDACTED]@",
        ),
    ]

    def _sanitize_message(self, msg: str) -> str:
        for pattern, replacement in self.PATTERNS:
            msg = pattern.sub(replacement, msg)
        return msg

    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize the main message
        if record.msg and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)

        # Sanitize message arguments if they are strings
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (
                        self._sanitize_message(str(v))
                        if isinstance(v, str)
                        else v
                    )
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    (
                        self._sanitize_message(str(arg))
                        if isinstance(arg, str)
                        else arg
                    )
                    for arg in record.args
                )

        return True


class DuplicateFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.seen_messages: dict[str, None] = {}
        self.max_cache_size = DEFAULT_LOG_CACHE_SIZE

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        key = f"{record.levelname}:{msg[:200]}"

        if key in self.seen_messages:
            return False

        self.seen_messages[key] = None

        if len(self.seen_messages) > self.max_cache_size:
            # Evict oldest half (dict preserves insertion order since Python 3.7)
            keys_to_keep = list(self.seen_messages.keys())[
                self.max_cache_size // 2 :
            ]
            self.seen_messages = dict.fromkeys(keys_to_keep)

        return True


class ColorFormatter(logging.Formatter):
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

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        include_module: bool = True,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.include_module = include_module
        # Import model colors from terminal module
        from certamen_core.shared.terminal import MODEL_COLORS

        self.MODEL_COLORS = MODEL_COLORS  # type: ignore[misc]

    def format(self, record: logging.LogRecord) -> str:
        display_type = getattr(record, "display_type", None)

        # More robust color capability detection
        should_colorize = self._should_use_color()

        if display_type:
            return self._format_display(record, display_type, should_colorize)

        level_name = record.levelname

        # Get the original formatted message first
        original_msg = super().format(record)

        # Build context parts from record attributes (set by ContextFilter)
        from certamen_core.shared.logging.structured import (
            build_context_parts,
        )

        context_parts = build_context_parts(record)

        # Add module info if enabled
        if self.include_module and hasattr(record, "module"):
            context_parts.append(f"{record.module}")

        # Insert context after [LEVEL] if present
        if context_parts and "[" in original_msg:
            # Find the end of the level marker
            level_end = original_msg.find("]") + 1
            context_prefix = "[" + "] [".join(context_parts) + "] "
            original_msg = (
                original_msg[:level_end]
                + " "
                + context_prefix
                + original_msg[level_end:].lstrip()
            )
        elif context_parts:
            context_prefix = "[" + "] [".join(context_parts) + "] "
            original_msg = context_prefix + original_msg

        if should_colorize:
            color = self.COLORS.get(level_name, "")
            reset = Style.RESET_ALL
            return f"{color}{original_msg}{reset}"
        else:
            # Remove any existing ANSI color codes to ensure clean output in non-color environments
            from certamen_core.shared.terminal import strip_ansi_codes

            return strip_ansi_codes(original_msg)

    def _format_section_header(
        self, message: str, should_colorize: bool
    ) -> str:
        if should_colorize:
            return f"\n{Fore.CYAN}--- {message} ---{Style.RESET_ALL}"
        return f"\n--- {message} ---"

    def _format_header(self, message: str, should_colorize: bool) -> str:
        border = "=" * 50
        if should_colorize:
            return (
                f"\n{Fore.CYAN}{border}\n{message}\n{border}{Style.RESET_ALL}"
            )
        return f"\n{border}\n{message}\n{border}"

    def _format_model_response(
        self, record: logging.LogRecord, message: str, should_colorize: bool
    ) -> str:
        model_name = getattr(record, "model_name", "Unknown")
        color = (
            self.MODEL_COLORS.get(model_name, Fore.WHITE)
            if should_colorize
            else ""
        )
        reset = Style.RESET_ALL if should_colorize else ""

        lines = [
            f"\n{color}Model: {model_name}{reset}",
            f"{color}Response:{reset}",
        ]
        for line in message.split("\n"):
            lines.append(f"{color}{line}{reset}")
        lines.append(f"{color}{'-' * 20}{reset}")
        return "\n".join(lines)

    def _format_colored_text(
        self, record: logging.LogRecord, message: str, should_colorize: bool
    ) -> str:
        ansi_color = getattr(record, "ansi_color", None)
        if should_colorize and ansi_color:
            return f"{ansi_color}{message}{Style.RESET_ALL}"
        elif not should_colorize:
            from certamen_core.shared.terminal import strip_ansi_codes

            return strip_ansi_codes(message)
        return message

    def _format_display(
        self,
        record: logging.LogRecord,
        display_type: str,
        should_colorize: bool,
    ) -> str:
        message = record.getMessage()

        if display_type == "section_header":
            return self._format_section_header(message, should_colorize)
        elif display_type == "header":
            return self._format_header(message, should_colorize)
        elif display_type == "model_response":
            return self._format_model_response(
                record, message, should_colorize
            )
        elif display_type == "colored_text":
            return self._format_colored_text(record, message, should_colorize)

        # Fallback to regular formatting
        result: str = super().format(record)
        return result

    def _should_use_color(self) -> bool:
        from certamen_core.shared.terminal import should_use_color

        return should_use_color()


def _validate_log_file_path(log_file: str | None) -> str | None:
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
                print(
                    f"Warning: Log directory {log_dir} is not writable. Logs will not be saved to file."
                )
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
    _file_format: str,
    file_duplicate_filter: DuplicateFilter,
    context_filter: logging.Filter,
    sensitive_data_filter: SensitiveDataFilter | None = None,
    _include_module: bool = True,
) -> logging.FileHandler | None:
    try:
        file_handler = logging.FileHandler(
            log_file, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(
            logging.DEBUG
        )  # Save all logs to file regardless of level

        # Always use JSON format for file logging (structured, parseable)
        from certamen_core.shared.logging.structured import JSONFormatter

        file_handler.setFormatter(JSONFormatter())

        if sensitive_data_filter:
            file_handler.addFilter(
                sensitive_data_filter
            )  # Must be first to sanitize before other filters
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
            raise OSError(
                f"Error writing to log file: {test_error!s}"
            ) from test_error

        return file_handler
    except OSError as file_error:
        print(
            f"CRITICAL ERROR: Cannot open log file {log_file}: {file_error!s}"
        )
        print("MODEL RESPONSES WILL NOT BE SAVED TO FILE!")
        print("This means expensive API calls could be lost!")
        print("Continuing with console logging only.")
        return None
    except Exception as e:
        print(
            f"CRITICAL ERROR: Unexpected error setting up file logging: {e!s}"
        )
        print("MODEL RESPONSES WILL NOT BE SAVED TO FILE!")
        print("This means expensive API calls could be lost!")
        print("Continuing with console logging only.")
        return None


def _configure_litellm() -> None:
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
        litellm.suppress_debug_info = True  # type: ignore[attr-defined]
        litellm.drop_params = True  # type: ignore[attr-defined]
        litellm.set_verbose = False  # type: ignore[attr-defined]

        # Suppress LiteLLM's own logger
        litellm_logger = logging.getLogger("LiteLLM")
        litellm_logger.setLevel(logging.CRITICAL)
        litellm_logger.propagate = False
    except ImportError:
        pass  # LiteLLM not installed, skip configuration


def _get_log_level_from_env() -> int | None:
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if not env_level:
        return None
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(env_level)


def _resolve_log_file_path(
    log_file: str | None, enable_file_logging: bool, log_dir: str | None
) -> str | None:
    if log_file is not None or not enable_file_logging:
        return log_file
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"certamen_{timestamp}_logs.log"
    if log_dir:
        resolved_dir = Path(log_dir).resolve()
        os.makedirs(resolved_dir, exist_ok=True)
        return os.path.join(str(resolved_dir), filename)
    return filename


def _resolve_log_level(level: int, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    env_level = _get_log_level_from_env()
    return env_level if env_level is not None else level


def _resolve_console_level(level: int, verbose: bool, debug: bool) -> int:
    if verbose or debug:
        return level
    if _get_log_level_from_env() is not None:
        return level
    return logging.INFO


def _reset_root_logger() -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    return root_logger


def _configure_third_party_loggers() -> None:
    third_party_loggers = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "litellm": logging.ERROR,
        "openai": logging.WARNING,
        "anthropic": logging.WARNING,
        "google": logging.WARNING,
        "vertexai": logging.WARNING,
        "asyncio": logging.WARNING,
    }
    for logger_name, log_level in third_party_loggers.items():
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(log_level)
        third_party_logger.propagate = logger_name != "litellm"


def setup_logging(
    log_file: str | None = None,
    level: int = logging.INFO,
    debug: bool = False,
    verbose: bool = False,
    enable_file_logging: bool = True,
    include_module: bool = True,
    log_dir: str | None = None,
) -> logging.Logger:
    log_file = _resolve_log_file_path(log_file, enable_file_logging, log_dir)
    log_file = _validate_log_file_path(log_file)
    level = _resolve_log_level(level, debug)

    root_logger = _reset_root_logger()

    from certamen_core.shared.logging.structured import ContextFilter

    console_format = "[%(levelname)s] %(message)s"
    file_format = "%(asctime)s [%(levelname)s] %(message)s"

    console_duplicate_filter = DuplicateFilter()
    file_duplicate_filter = DuplicateFilter()
    context_filter = ContextFilter()
    sensitive_data_filter = SensitiveDataFilter()

    console_level = _resolve_console_level(level, verbose, debug)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        ColorFormatter(console_format, include_module=include_module)
    )
    console_handler.addFilter(sensitive_data_filter)
    console_handler.addFilter(console_duplicate_filter)
    console_handler.addFilter(context_filter)

    file_handler = None
    if log_file:
        file_handler = _create_file_handler(
            log_file,
            file_format,
            file_duplicate_filter,
            context_filter,
            sensitive_data_filter=sensitive_data_filter,
            _include_module=include_module,
        )

    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
        root_logger.info("Log file initialized at %s", log_file)

    _configure_litellm()

    logger = logging.getLogger("certamen")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    _configure_third_party_loggers()

    return logger
