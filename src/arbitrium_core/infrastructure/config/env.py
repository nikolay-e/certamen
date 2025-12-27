import os

from arbitrium_core.shared.constants import DEFAULT_OLLAMA_URL


def get_bool_env(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() in {"1", "true", "yes"}


def get_int_env(key: str, default: str) -> int:
    return int(os.getenv(key, default))


def get_str_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_ollama_base_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "OLLAMA_BASE_URL environment variable is required but not set.\n"
            "Options:\n"
            f"  - Local Ollama: {DEFAULT_OLLAMA_URL}\n"
            "  - Docker Compose ollama service: http://ollama:11434\n"
            "  - Host machine from Docker: http://host.docker.internal:11434"
        )
    return base_url


def get_comma_separated_env(key: str, default: str = "") -> set[str]:
    value = os.getenv(key, default)
    return set(filter(None, value.split(",")))
