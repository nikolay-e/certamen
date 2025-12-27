from arbitrium_core.shared.constants import RETRYABLE_ERROR_TYPES
from arbitrium_core.shared.errors import (
    ArbitriumError,
    ConfigurationError,
    FileSystemError,
)

__all__ = [
    "RETRYABLE_ERROR_TYPES",
    "APIError",
    "ArbitriumError",
    "AuthenticationError",
    "BudgetExceededError",
    "ConfigurationError",
    "ErrorClassification",
    "ExceptionClassifier",
    "FatalError",
    "FileSystemError",
    "GraphValidationError",
    "InputError",
    "ModelError",
    "ModelResponseError",
    "RateLimitError",
    "TournamentTimeoutError",
]


class FatalError(ArbitriumError):
    pass


class APIError(ArbitriumError):
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

        enhanced_message = message
        if provider:
            enhanced_message = f"[{provider}] {enhanced_message}"
        if status_code:
            enhanced_message = f"{enhanced_message} (Status: {status_code})"

        super().__init__(enhanced_message, *args)


class RateLimitError(APIError):
    pass


class AuthenticationError(APIError):
    pass


class ModelError(ArbitriumError):
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
    pass


class InputError(ArbitriumError):
    pass


class BudgetExceededError(ArbitriumError):
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


class TournamentTimeoutError(ArbitriumError):
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


class GraphValidationError(ArbitriumError):
    """Exception for workflow graph validation errors."""

    def __init__(
        self,
        message: str,
        node_id: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.node_id = node_id
        enhanced_message = message
        if node_id:
            enhanced_message = f"[Node: {node_id}] {enhanced_message}"
        super().__init__(enhanced_message, *args)


class ErrorClassification:
    __slots__ = ("error_type", "is_retryable", "message")

    def __init__(
        self, error_type: str, message: str, is_retryable: bool
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.is_retryable = is_retryable


class ExceptionClassifier:
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

    @classmethod
    def classify(
        cls,
        exc: Exception,
        context: str = "",
    ) -> ErrorClassification:
        exc_name = type(exc).__name__.lower()
        error_msg = str(exc).lower()

        if (
            "unable to complete request" in error_msg
            and "max_output_tokens" in error_msg
        ):
            msg = cls._format_message("token_limit", exc, context)
            return ErrorClassification("token_limit", msg, False)

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

        from arbitrium_core.shared.constants import (
            ERROR_PATTERNS,
            PERMISSION_ERROR_PATTERNS,
        )

        if any(p in error_msg for p in PERMISSION_ERROR_PATTERNS):
            msg = cls._format_message("permission_denied", exc, context)
            return ErrorClassification("permission_denied", msg, False)

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
        type_labels = {
            "rate_limit": "Rate limit exceeded",
            "timeout": "Request timed out",
            "authentication": "Authentication failed",
            "not_found": "Resource not found",
            "service_unavailable": "Service unavailable",
            "service": "Server error",
            "overloaded": "Server overloaded",
            "connection": "Connection error",
            "token_limit": "Output token limit reached",
            "permission_denied": "Permission denied",
            "model_response_error": "Invalid model response",
            "general": "Unexpected error",
        }
        label = type_labels.get(error_type, "Error")
        if context:
            return f"{label} with {context}: {exc}"
        return f"{label}: {exc}"
