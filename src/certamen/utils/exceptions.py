"""
Exception classes for Certamen Framework.

This module provides a structured hierarchy of custom exceptions
to enable more granular error handling throughout the application.
"""


class CertamenError(Exception):
    """Base exception class for all Certamen Framework-specific errors."""

    def __init__(self, message: str, *args: object, **kwargs: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(CertamenError):
    """Raised when there are issues with the configuration."""


class FatalError(CertamenError):
    """Raised for fatal errors that should terminate the application."""


class APIError(CertamenError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.provider = provider
        self.status_code = status_code

        # Enhance the message with provider and status code if available
        enhanced_message = message
        if provider:
            enhanced_message = f"[{provider}] {enhanced_message}"
        if status_code:
            enhanced_message = f"{enhanced_message} (Status: {status_code})"

        super().__init__(enhanced_message, *args)


class RateLimitError(APIError):
    """Raised when hitting rate limits on API calls."""


class AuthenticationError(APIError):
    """Raised when API authentication fails."""


class ModelError(CertamenError):
    """Base class for model-related errors."""

    def __init__(
        self,
        message: str,
        model_key: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.model_key = model_key
        enhanced_message = message

        if model_key:
            enhanced_message = f"[{model_key}] {enhanced_message}"

        super().__init__(enhanced_message, *args)


class ModelResponseError(ModelError):
    """Raised when a model's response is invalid or problematic."""


class FileSystemError(CertamenError):
    """Base class for filesystem-related errors."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.file_path = file_path
        enhanced_message = message

        if file_path:
            enhanced_message = f"[{file_path}] {enhanced_message}"

        super().__init__(enhanced_message, *args)


class InputError(CertamenError):
    """Raised when there's an issue with user input."""


class BudgetExceededError(CertamenError):
    """Raised when tournament exceeds the configured budget."""

    def __init__(
        self,
        message: str,
        spent: float,
        budget: float,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.spent = spent
        self.budget = budget
        enhanced_message = f"Budget exceeded: spent ${spent:.4f} >= limit ${budget:.4f}. {message}"
        super().__init__(enhanced_message, *args)


class TournamentTimeoutError(CertamenError):
    """Raised when tournament exceeds the configured time limit."""

    def __init__(
        self,
        message: str,
        elapsed: float,
        timeout: float,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.elapsed = elapsed
        self.timeout = timeout
        enhanced_message = f"Tournament timeout: {elapsed:.1f}s >= limit {timeout:.1f}s. {message}"
        super().__init__(enhanced_message, *args)


class ErrorClassification:
    """Result of exception classification."""

    __slots__ = ("error_type", "is_retryable", "message")

    def __init__(
        self, error_type: str, message: str, is_retryable: bool
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.is_retryable = is_retryable


RETRYABLE_ERROR_TYPES = frozenset(
    {
        "rate_limit",
        "timeout",
        "connection",
        "service",
        "overloaded",
    }
)


class ExceptionClassifier:
    """Unified exception classifier for LLM API errors.

    Combines isinstance-based type checking with pattern matching for
    consistent error classification across the codebase.
    """

    _EXCEPTION_TYPE_MAP: dict[str, tuple[str, bool]] = {
        "ratelimiterror": ("rate_limit", True),
        "timeout": ("timeout", True),
        "timeouterror": ("timeout", True),
        "asynciotimeouterror": ("timeout", True),
        "authenticationerror": ("authentication", False),
        "notfounderror": ("not_found", False),
        "serviceunavailableerror": ("service_unavailable", True),
        "internalservererror": ("service", True),
        "connectionerror": ("connection", True),
        "modelresponseerror": ("model_response_error", False),
    }

    _PERMISSION_PATTERNS = (
        "permission_denied",
        "service_disabled",
        "api has not been used",
    )

    @classmethod
    def classify(
        cls,
        exc: Exception,
        context: str = "",
    ) -> ErrorClassification:
        """Classify an exception and determine if it's retryable.

        Args:
            exc: The exception to classify
            context: Optional context string (e.g., model display name)

        Returns:
            ErrorClassification with error_type, message, and is_retryable
        """
        exc_name = type(exc).__name__.lower()
        error_msg = str(exc).lower()

        for type_pattern, (
            mapped_type,
            is_retryable,
        ) in cls._EXCEPTION_TYPE_MAP.items():
            if type_pattern in exc_name:
                error_type = (
                    "overloaded"
                    if mapped_type == "service" and "overload" in error_msg
                    else mapped_type
                )
                msg = cls._format_message(error_type, exc, context)
                return ErrorClassification(error_type, msg, is_retryable)

        if any(p in error_msg for p in cls._PERMISSION_PATTERNS):
            msg = cls._format_message("permission_denied", exc, context)
            return ErrorClassification("permission_denied", msg, False)

        from .constants import ERROR_PATTERNS

        for error_type, patterns in ERROR_PATTERNS.items():
            if any(p in error_msg for p in patterns):
                is_retryable = error_type in RETRYABLE_ERROR_TYPES
                msg = cls._format_message(error_type, exc, context)
                return ErrorClassification(error_type, msg, is_retryable)

        msg = cls._format_message("general", exc, context)
        return ErrorClassification("general", msg, False)

    @classmethod
    def _format_message(
        cls, error_type: str, exc: Exception, context: str
    ) -> str:
        """Format error message with context."""
        type_labels = {
            "rate_limit": "Rate limit exceeded",
            "timeout": "Request timed out",
            "authentication": "Authentication failed",
            "not_found": "Resource not found",
            "service_unavailable": "Service unavailable",
            "service": "Server error",
            "overloaded": "Server overloaded",
            "connection": "Connection error",
            "permission_denied": "Permission denied",
            "model_response_error": "Invalid model response",
            "general": "Unexpected error",
        }
        label = type_labels.get(error_type, "Error")
        if context:
            return f"{label} with {context}: {exc}"
        return f"{label}: {exc}"
