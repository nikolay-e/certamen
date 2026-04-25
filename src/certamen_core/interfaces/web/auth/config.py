from certamen_core.infrastructure.config.env import (
    get_bool_env,
    get_int_env,
    get_str_env,
)
from certamen_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)

DB_HOST = get_str_env("CERTAMEN_DB_HOST", "localhost")
DB_PORT = get_int_env("CERTAMEN_DB_PORT", "5432")
DB_NAME = get_str_env("CERTAMEN_DB_NAME", "certamen")
DB_USER = get_str_env("CERTAMEN_DB_USER", "certamen")
DB_PASSWORD = get_str_env("CERTAMEN_DB_PASSWORD")
DB_POOL_MIN_SIZE = get_int_env("CERTAMEN_DB_POOL_MIN_SIZE", "2")
DB_POOL_MAX_SIZE = get_int_env("CERTAMEN_DB_POOL_MAX_SIZE", "10")

JWT_SECRET = get_str_env("CERTAMEN_JWT_SECRET")
JWT_ACCESS_TOKEN_EXPIRES_MINUTES = get_int_env(
    "CERTAMEN_JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "15"
)
JWT_REFRESH_TOKEN_EXPIRES_DAYS = get_int_env(
    "CERTAMEN_JWT_REFRESH_TOKEN_EXPIRES_DAYS", "30"
)

SKIP_AUTH = get_bool_env("CERTAMEN_SKIP_AUTH")
SKIP_DB_INIT = get_bool_env("SKIP_DB_INIT")
IS_PRODUCTION = (
    get_str_env("CERTAMEN_ENV", "development").lower() == "production"
)

if not SKIP_AUTH and (not JWT_SECRET or len(JWT_SECRET) < 32):
    raise RuntimeError(
        "CERTAMEN_JWT_SECRET must be set and be at least 32 characters. "
        "Generate with: openssl rand -base64 32"
    )

if not SKIP_DB_INIT and not DB_PASSWORD:
    raise RuntimeError(
        "CERTAMEN_DB_PASSWORD must be set when database initialization is enabled. "
        "Set SKIP_DB_INIT=true to bypass this check (not recommended for production)."
    )

if SKIP_AUTH and IS_PRODUCTION:
    raise RuntimeError(
        "CERTAMEN_SKIP_AUTH cannot be enabled in production environment. "
        "Set CERTAMEN_ENV=development or remove CERTAMEN_SKIP_AUTH."
    )

if SKIP_AUTH:
    logger.warning(
        "WARNING: Authentication is DISABLED (CERTAMEN_SKIP_AUTH=true). "
        "This is INSECURE and should only be used for development/testing."
    )
