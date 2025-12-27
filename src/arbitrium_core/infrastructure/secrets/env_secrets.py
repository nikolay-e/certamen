import os
import subprocess

from arbitrium_core.shared.constants import DEFAULT_SUBPROCESS_TIMEOUT
from arbitrium_core.shared.errors import ConfigurationError
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger("arbitrium.utils.secrets")


def _validate_config_structure(config: dict[str, object] | None) -> None:
    if config is None:
        raise ConfigurationError(
            "Secret configuration not provided. Config must be passed to get_secret_config."
        )

    if not isinstance(config, dict):
        raise ConfigurationError("Invalid config type. Expected dict.")


def _extract_secrets_config(config: dict[str, object]) -> dict[str, object]:
    secrets_config: object = config.get("secrets", {})

    if not isinstance(secrets_config, dict):
        raise ConfigurationError("Invalid secrets configuration type.")

    if not secrets_config:
        raise ConfigurationError(
            "No 'secrets' section found in configuration file. Please add secrets configuration."
        )

    return secrets_config


def _extract_providers_config(
    secrets_config: dict[str, object],
) -> dict[str, object]:
    providers: object = secrets_config.get("providers", {})

    if not isinstance(providers, dict):
        raise ConfigurationError("Invalid providers configuration type.")

    if not providers:
        raise ConfigurationError(
            "No providers found in secrets configuration. Please add provider configurations."
        )

    return providers


def _validate_provider_config(provider: str, config_data: object) -> None:
    if not isinstance(config_data, dict):
        raise ConfigurationError(
            f"Invalid secret configuration for provider '{provider}': must be a dictionary"
        )

    if "env_var" not in config_data:
        raise ConfigurationError(
            f"Missing 'env_var' in secret configuration for provider '{provider}'"
        )

    if "op_path" not in config_data:
        raise ConfigurationError(
            f"Missing 'op_path' in secret configuration for provider '{provider}'"
        )


def _build_secret_config(
    providers: dict[str, object],
) -> dict[str, tuple[str, str]]:
    secret_config: dict[str, tuple[str, str]] = {}

    for provider, config_data in providers.items():
        _validate_provider_config(provider, config_data)
        assert isinstance(
            config_data, dict
        )  # Validated by _validate_provider_config
        secret_config[provider] = (
            str(config_data["env_var"]),
            str(config_data["op_path"]),
        )

    return secret_config


def get_secret_config(config: dict[str, object]) -> dict[str, tuple[str, str]]:
    _validate_config_structure(config)
    secrets_config = _extract_secrets_config(config)
    providers = _extract_providers_config(secrets_config)
    return _build_secret_config(providers)


def _ensure_op_cli_is_available() -> None:
    try:
        # 'op account list' is a lightweight command to check for an active session.
        subprocess.run(
            ["op", "account", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=DEFAULT_SUBPROCESS_TIMEOUT,
        )
    except FileNotFoundError as err:
        raise ConfigurationError(
            "1Password CLI ('op') is not installed. Please install it to continue."
        ) from err
    except subprocess.CalledProcessError as err:
        raise ConfigurationError(
            "You are not signed in to the 1Password CLI. Please run 'op signin' to continue."
        ) from err
    except subprocess.TimeoutExpired as err:
        raise ConfigurationError(
            "The 1Password CLI timed out. It may be busy or unresponsive."
        ) from err


def _get_missing_providers(
    secret_config: dict[str, tuple[str, str]], required_providers: list[str]
) -> list[str]:
    providers_to_fetch = []
    for provider in required_providers:
        provider_lower = provider.lower()
        if provider_lower in secret_config:
            env_var, _ = secret_config[provider_lower]
            if not os.getenv(env_var):
                providers_to_fetch.append(provider_lower)
    return providers_to_fetch


def _handle_missing_op_cli(
    secret_config: dict[str, tuple[str, str]],
    providers_to_fetch: list[str],
    error: ConfigurationError,
) -> None:
    logger.error(
        f"1Password CLI not available, and secrets are missing. Error: {error}"
    )
    missing_vars = [secret_config[p][0] for p in providers_to_fetch]
    raise ConfigurationError(
        f"Secrets not found in environment variables ({', '.join(missing_vars)}) and 1Password CLI is not available. "
        "Please set them manually or configure 1Password."
    ) from error


def _fetch_secret_from_1password(env_var: str, op_path: str) -> None:
    try:
        result = subprocess.run(
            ["op", "read", op_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=DEFAULT_SUBPROCESS_TIMEOUT,
        )
        secret_value = result.stdout.strip()
        if not secret_value:
            raise ConfigurationError(
                f"Secret value for '{op_path}' is empty in 1Password."
            )

        os.environ[env_var] = secret_value
        logger.info(f"Successfully loaded and set '{env_var}' from 1Password.")

    except subprocess.CalledProcessError as e:
        logger.error(f"1Password CLI error output: {e.stderr.strip()}")
        error_message = (
            f"Failed to read secret '{op_path}' from 1Password. "
            f"Please ensure the secret exists and you have access permissions. "
            f"Exit code: {e.returncode}"
        )
        raise ConfigurationError(error_message) from e
    except subprocess.TimeoutExpired as err:
        raise ConfigurationError(
            f"Timeout while fetching secret '{op_path}' from 1Password."
        ) from err


def load_secrets(
    config: dict[str, object], required_providers: list[str]
) -> None:
    secret_config = get_secret_config(config)
    providers_to_fetch = _get_missing_providers(
        secret_config, required_providers
    )

    if not providers_to_fetch:
        logger.info("All required secrets are present in the environment.")
        return

    logger.info(
        f"Missing secrets for: {', '.join(providers_to_fetch)}. Attempting to fetch from 1Password."
    )

    try:
        _ensure_op_cli_is_available()
    except ConfigurationError as e:
        _handle_missing_op_cli(secret_config, providers_to_fetch, e)

    for provider in providers_to_fetch:
        env_var, op_path = secret_config[provider]
        _fetch_secret_from_1password(env_var, op_path)
