"""Structured logging with correlation IDs and JSON support for Arbitrium Framework."""

import contextvars
import json
import logging
import uuid
from datetime import datetime

# Context variables for correlation IDs
run_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
task_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)
phase_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("phase", default=None)
model_context: contextvars.ContextVar[str | None] = contextvars.ContextVar("model", default=None)


def generate_run_id() -> str:
    """Generate a unique run ID for tracing an entire tournament."""
    return str(uuid.uuid4())[:8]


def generate_task_id() -> str:
    """Generate a unique task ID for tracing a single model interaction."""
    return str(uuid.uuid4())[:8]


def set_run_id(run_id: str) -> None:
    """Set the current run ID in context."""
    run_id_context.set(run_id)


def set_task_id(task_id: str) -> None:
    """Set the current task ID in context."""
    task_id_context.set(task_id)


def set_phase(phase: str) -> None:
    """Set the current phase in context."""
    phase_context.set(phase)


def set_model(model: str) -> None:
    """Set the current model in context."""
    model_context.set(model)


def get_context() -> dict[str, str]:
    """Get all current context values."""
    context = {}

    run_id = run_id_context.get()
    if run_id:
        context["run_id"] = run_id

    task_id = task_id_context.get()
    if task_id:
        context["task_id"] = task_id

    phase = phase_context.get()
    if phase:
        context["phase"] = phase

    model = model_context.get()
    if model:
        context["model"] = model

    return context


def clear_task_context() -> None:
    """Clear task-specific context (task_id)."""
    task_id_context.set(None)


def clear_all_context() -> None:
    """Clear all context variables."""
    run_id_context.set(None)
    task_id_context.set(None)
    phase_context.set(None)
    model_context.set(None)


def build_context_parts(record: logging.LogRecord) -> list[str]:
    """
    Build context parts from log record attributes.

    This function extracts context attributes set by ContextFilter and
    formats them as a list of strings for display.

    Args:
        record: LogRecord with context attributes set by ContextFilter

    Returns:
        List of formatted context strings (e.g., ["run:abc123", "task:def456"])
    """
    context_parts = []

    # Check for context attributes (set by ContextFilter)
    if hasattr(record, "run_id") and record.run_id:
        context_parts.append(f"run:{record.run_id}")

    if hasattr(record, "task_id") and record.task_id:
        context_parts.append(f"task:{record.task_id}")

    if hasattr(record, "phase") and record.phase:
        context_parts.append(f"phase:{record.phase}")

    if hasattr(record, "model") and record.model:
        context_parts.append(f"model:{record.model}")

    return context_parts


class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Build the base log entry
        from typing import Any

        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add module/file information
        if record.pathname:
            log_entry["module"] = record.module
            log_entry["file"] = record.filename
            log_entry["line"] = record.lineno

        # Add correlation IDs and context from context vars
        context = get_context()
        log_entry.update(context)

        # Add any extra attributes passed via 'extra' parameter
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """
    Filter that adds context as LogRecord attributes.

    This filter reads from contextvars and sets them as LogRecord attributes
    so they can be used in format strings.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context attributes to the log record."""
        context = get_context()
        record.run_id = context.get("run_id", "")
        record.task_id = context.get("task_id", "")
        record.phase = context.get("phase", "")
        record.model = context.get("model", "")
        return True


class StructuredFormatter(logging.Formatter):
    """
    Enhanced text formatter with correlation IDs and module information.

    Format: timestamp [level] [run_id:abc123] [task_id:def456] [module] message

    Note: This formatter requires ContextFilter to be added to the handler
    for proper context injection.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, include_module: bool = True) -> None:
        """Initialize the formatter."""
        # Build format string with context attributes
        if fmt is None:
            # Build dynamic format based on available context
            fmt = "%(asctime)s [%(levelname)s]"
            if include_module:
                fmt += " [%(module)s]"
            fmt += " %(message)s"

        super().__init__(fmt, datefmt)
        self.include_module = include_module

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with correlation IDs and context."""
        # Build context parts from record attributes (set by ContextFilter)
        context_parts = build_context_parts(record)

        # Format the original message
        original_msg = super().format(record)

        # Insert context after [LEVEL] if present
        if context_parts and "[" in original_msg:
            # Find the end of the level marker
            level_end = original_msg.find("]") + 1
            context_prefix = "[" + "] [".join(context_parts) + "] "
            return original_msg[:level_end] + " " + context_prefix + original_msg[level_end:].lstrip()
        elif context_parts:
            context_prefix = "[" + "] [".join(context_parts) + "] "
            return context_prefix + original_msg
        else:
            return original_msg


class ContextualLogger:
    """
    Logger with automatic correlation ID and context management.

    Usage:
        logger = ContextualLogger("arbitrium")

        # Set run context
        logger.set_run("abc123")

        # Set task context
        with logger.task_context("def456", phase="INITIAL", model="gpt-4"):
            logger.info("Processing task")  # Will include run_id, task_id, phase, model
    """

    def __init__(self, name: str = "arbitrium"):
        """Initialize the contextual logger."""
        self.logger = logging.getLogger(name)
        self._run_id: str | None = None

    def set_run(self, run_id: str | None = None) -> str:
        """
        Set the run ID for this tournament.

        Args:
            run_id: Optional run ID. If not provided, generates a new one.

        Returns:
            The run ID that was set.
        """
        if run_id is None:
            run_id = generate_run_id()
        self._run_id = run_id
        set_run_id(run_id)
        return run_id

    def get_run_id(self) -> str | None:
        """Get the current run ID."""
        return self._run_id

    class TaskContext:
        """Context manager for task-level logging."""

        def __init__(
            self,
            task_id: str | None = None,
            phase: str | None = None,
            model: str | None = None,
        ):
            """Initialize task context."""
            self.task_id = task_id or generate_task_id()
            self.phase = phase
            self.model = model

        def __enter__(self) -> "ContextualLogger.TaskContext":
            """Enter the context."""
            set_task_id(self.task_id)
            if self.phase:
                set_phase(self.phase)
            if self.model:
                set_model(self.model)
            return self

        def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
            """Exit the context and clear task-specific data."""
            clear_task_context()
            if self.phase:
                phase_context.set(None)
            if self.model:
                model_context.set(None)

    def task_context(
        self,
        task_id: str | None = None,
        phase: str | None = None,
        model: str | None = None,
    ) -> TaskContext:
        """
        Create a task context manager.

        Args:
            task_id: Optional task ID. If not provided, generates a new one.
            phase: Optional phase name (e.g., "INITIAL", "IMPROVEMENT").
            model: Optional model name (e.g., "gpt-4", "gemma-2b").

        Returns:
            TaskContext instance for use with 'with' statement.
        """
        return self.TaskContext(task_id=task_id, phase=phase, model=model)

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug message with context."""
        self.logger.debug(message, extra=kwargs, stacklevel=2)

    def info(self, message: str, **kwargs: object) -> None:
        """Log info message with context."""
        self.logger.info(message, extra=kwargs, stacklevel=2)

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning message with context."""
        self.logger.warning(message, extra=kwargs, stacklevel=2)

    def error(self, message: str, exc_info: bool = False, **kwargs: object) -> None:
        """Log error message with context."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs, stacklevel=2)

    def critical(self, message: str, exc_info: bool = True, **kwargs: object) -> None:
        """Log critical message with context."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs, stacklevel=2)


def get_contextual_logger(name: str = "arbitrium") -> ContextualLogger:
    """
    Get a contextual logger instance.

    NOTE: You must call setup_logging() before using this logger!

    Args:
        name: Logger name (default: "arbitrium")

    Returns:
        ContextualLogger instance with correlation ID support.
    """
    return ContextualLogger(name)
