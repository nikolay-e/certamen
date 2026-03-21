import os


def get_bool_env(key: str, default: str = "false") -> bool:
    """Parse boolean environment variable.

    Accepts: '1', 'true', 'yes' (case-insensitive) as True
    Everything else is False

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Parsed boolean value
    """
    return os.getenv(key, default).lower() in {"1", "true", "yes"}


def get_int_env(key: str, default: str) -> int:
    """Parse integer environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set (as string)

    Returns:
        Parsed integer value

    Raises:
        ValueError: If value cannot be converted to int
    """
    return int(os.getenv(key, default))


def get_str_env(key: str, default: str = "") -> str:
    """Get string environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_ollama_base_url() -> str:
    """Get and validate OLLAMA_BASE_URL from environment.

    Returns:
        OLLAMA_BASE_URL value

    Raises:
        RuntimeError: If OLLAMA_BASE_URL is not set
    """
    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "OLLAMA_BASE_URL environment variable is required but not set.\n"
            "Options:\n"
            "  - Local Ollama: http://localhost:11434\n"
            "  - Docker Compose ollama service: http://ollama:11434\n"
            "  - Host machine from Docker: http://host.docker.internal:11434"
        )
    return base_url


def get_comma_separated_env(key: str, default: str = "") -> set[str]:
    """Parse comma-separated environment variable into a set.

    Filters out empty strings after splitting.

    Args:
        key: Environment variable name
        default: Default comma-separated value

    Returns:
        Set of non-empty strings
    """
    value = os.getenv(key, default)
    return set(filter(None, value.split(",")))
