"""Base exceptions for Arbitrium Framework.

This module contains the base exception classes used throughout the framework.
Domain-specific exceptions should extend these in the domain layer.
"""


class ArbitriumError(Exception):
    """Base exception for all Arbitrium errors."""

    def __init__(self, message: str, *args: object, **kwargs: object) -> None:
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(ArbitriumError):
    """Exception for configuration errors."""

    pass


class FileSystemError(ArbitriumError):
    """Exception for file system operations."""

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
