import contextvars
import json
import logging
import uuid
from datetime import datetime

from arbitrium_core.shared.json_utils import sanitize_for_json

# Context variables for correlation IDs
run_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "run_id", default=None
)
task_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "task_id", default=None
)
phase_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "phase", default=None
)
model_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "model", default=None
)


def generate_tournament_run_id() -> str:
    return str(uuid.uuid4())[:8]


def generate_unique_task_id() -> str:
    return str(uuid.uuid4())[:8]


def set_run_id(run_id: str) -> None:
    run_id_context.set(run_id)


def set_task_id(task_id: str) -> None:
    task_id_context.set(task_id)


def set_phase(phase: str) -> None:
    phase_context.set(phase)


def set_model(model: str) -> None:
    model_context.set(model)


def get_context() -> dict[str, str]:
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
    task_id_context.set(None)


def build_context_parts(record: logging.LogRecord) -> list[str]:
    context_parts = []

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
    def _sanitize_value(self, value: object, max_length: int = 500) -> object:
        return sanitize_for_json(
            value,
            max_length=max_length,
            truncate_suffix=(
                f"... (truncated, {len(str(value))} total chars)"
                if isinstance(value, str)
                else "... (truncated)"
            ),
        )

    def format(self, record: logging.LogRecord) -> str:
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
                    # Sanitize value to prevent JSON issues and bloat
                    log_entry[key] = self._sanitize_value(value)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Use ensure_ascii=False for unicode support, default=str for non-serializable objects
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = get_context()
        record.run_id = context.get("run_id", "")
        record.task_id = context.get("task_id", "")
        record.phase = context.get("phase", "")
        record.model = context.get("model", "")
        return True


class ContextualLogger:
    def __init__(self, name: str = "arbitrium"):
        self.logger = logging.getLogger(name)
        self._run_id: str | None = None

    def set_run(self, run_id: str | None = None) -> str:
        if run_id is None:
            run_id = generate_tournament_run_id()
        self._run_id = run_id
        set_run_id(run_id)
        return run_id

    class TaskContext:
        def __init__(
            self,
            task_id: str | None = None,
            phase: str | None = None,
            model: str | None = None,
        ):
            self.task_id = task_id or generate_unique_task_id()
            self.phase = phase
            self.model = model

        def __enter__(self) -> "ContextualLogger.TaskContext":
            set_task_id(self.task_id)
            if self.phase:
                set_phase(self.phase)
            if self.model:
                set_model(self.model)
            return self

        def __exit__(
            self, _exc_type: object, _exc_val: object, _exc_tb: object
        ) -> None:
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
        return self.TaskContext(task_id=task_id, phase=phase, model=model)

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        self.logger.debug(message, *args, extra=extra, stacklevel=2)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        self.logger.info(message, *args, extra=extra, stacklevel=2)

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        exc_info = bool(kwargs.get("exc_info", False))
        self.logger.warning(
            message, *args, exc_info=exc_info, extra=extra, stacklevel=2
        )

    def error(
        self,
        message: str,
        *args: object,
        exc_info: bool = False,
        **kwargs: object,
    ) -> None:
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        self.logger.error(
            message, *args, exc_info=exc_info, extra=extra, stacklevel=2
        )

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        self.logger.exception(message, *args, extra=extra, stacklevel=2)

    def critical(
        self,
        message: str,
        *args: object,
        exc_info: bool = True,
        **kwargs: object,
    ) -> None:
        extra = {k: v for k, v in kwargs.items() if k != "exc_info"}
        self.logger.critical(
            message, *args, exc_info=exc_info, extra=extra, stacklevel=2
        )

    def log_prompt(
        self, prompt: str, model: str | None = None, **kwargs: object
    ) -> None:
        extra = {
            "event_type": "prompt",
            "prompt": prompt,
            "prompt_length": len(prompt),
            **kwargs,
        }
        if model:
            extra["model"] = model

        self.logger.debug(
            f"Prompt ({len(prompt)} chars)"
            + (f" to {model}" if model else ""),
            extra=extra,
            stacklevel=2,
        )

    def log_response(
        self,
        response: str,
        model: str | None = None,
        cost: float | None = None,
        **kwargs: object,
    ) -> None:
        extra = {
            "event_type": "response",
            "response": response,
            "response_length": len(response),
            **kwargs,
        }
        if model:
            extra["model"] = model
        if cost is not None:
            extra["cost"] = cost

        cost_str = f", ${cost:.4f}" if cost is not None else ""
        self.logger.debug(
            f"Response ({len(response)} chars)"
            + (f" from {model}" if model else "")
            + cost_str,
            extra=extra,
            stacklevel=2,
        )


def get_contextual_logger(name: str = "arbitrium") -> ContextualLogger:
    return ContextualLogger(name)
