import hashlib
from typing import Any

from aiohttp import web
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError

from certamen.api.schemas import (
    RefreshTokenRequest,
    UserLogin,
    UserRegistration,
)
from certamen.gui.auth.database import execute_write_transaction, query_db
from certamen.gui.auth.security import (
    DUMMY_HASH_FOR_TIMING_ATTACK_PREVENTION,
    auth_rate_limiter,
    get_current_user_from_request,
    hash_password,
    verify_password,
    verify_refresh_token,
)
from certamen.gui.auth.token_utils import (
    build_token_response,
    create_auth_tokens_and_save,
)
from certamen.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


async def parse_request_json(
    request: web.Request,
    schema_class: type[PydanticBaseModel],
    error_context: str,
) -> tuple[Any | None, web.Response | None]:
    try:
        data = await request.json()
        return schema_class(**data), None
    except (ValueError, ValidationError) as e:
        logger.warning("%s validation error: %s", error_context, e)
        return None, error_response(
            f"Invalid {error_context.lower()} data", 400
        )


def error_response(detail: str, status: int = 400) -> web.Response:
    return web.json_response({"detail": detail}, status=status)


async def register_user(request: web.Request) -> web.Response:
    user_data, error = await parse_request_json(
        request, UserRegistration, "Registration"
    )
    if error:
        return error
    assert user_data is not None

    logger.info("Starting registration for user: %s", user_data.username)
    try:
        existing_user = query_db(
            "SELECT id FROM users WHERE username = %s",
            (user_data.username,),
            one=True,
        )
        if existing_user:
            return error_response("Username already exists", 400)

        hashed_password = hash_password(user_data.password)
        result = execute_write_transaction(
            "INSERT INTO users (username, password) VALUES (%s, %s) "
            "RETURNING id, is_admin",
            (user_data.username, hashed_password),
            fetch_results=True,
            one=True,
        )

        if not result:
            return error_response("Failed to create user", 500)

        user_id = result["id"]
        is_admin = result.get("is_admin", False)
        logger.info(
            "Successfully created user %s with id %s",
            user_data.username,
            user_id,
        )

        access_token, refresh_token = create_auth_tokens_and_save(
            user_id, user_data.username, is_admin
        )
        return build_token_response(
            access_token,
            refresh_token,
            user_id,
            user_data.username,
            is_admin,
            status=201,
        )

    except Exception as e:
        logger.exception("Registration error: %s", e)
        return error_response("Registration failed", 500)


def _get_client_ip(request: web.Request) -> str:
    """Extract client IP from request, respecting X-Forwarded-For."""
    peername: tuple[str, int] | None = (
        request.transport.get_extra_info("peername")
        if request.transport
        else None
    )
    remote_ip: str | None = peername[0] if peername else None
    forwarded: str | None = request.headers.get("X-Forwarded-For")
    if forwarded:
        first_ip: str = forwarded.split(",")[0].strip()
        return first_ip
    return remote_ip or str(request.remote or "unknown")


async def login_user(request: web.Request) -> web.Response:
    user_data, error = await parse_request_json(request, UserLogin, "Login")
    if error:
        return error
    assert user_data is not None

    client_ip = _get_client_ip(request)
    logger.info(
        "Login attempt for user: %s from %s", user_data.username, client_ip
    )

    # Check rate limit before processing
    allowed, rate_limit_reason = await auth_rate_limiter.check_rate_limit(
        client_ip, user_data.username
    )
    if not allowed:
        logger.warning(
            "Login blocked by rate limit for %s from %s: %s",
            user_data.username,
            client_ip,
            rate_limit_reason,
        )
        return error_response(rate_limit_reason or "Rate limit exceeded", 429)

    try:
        user = query_db(
            "SELECT id, username, password, is_admin FROM users "
            "WHERE username = %s",
            (user_data.username,),
            one=True,
        )

        # Always run bcrypt to prevent timing attacks (user enumeration)
        # If user doesn't exist, verify against dummy hash to take same time
        password_hash = (
            user["password"]
            if user
            else DUMMY_HASH_FOR_TIMING_ATTACK_PREVENTION
        )
        password_valid = verify_password(user_data.password, password_hash)

        if not user or not password_valid:
            # Record failed attempt for account lockout
            await auth_rate_limiter.record_failed_attempt(user_data.username)
            logger.warning(
                "Invalid login attempt for user: %s from %s",
                user_data.username,
                client_ip,
            )
            return error_response("Invalid username or password", 401)

        # Clear failed attempts on successful login
        await auth_rate_limiter.record_successful_login(user_data.username)
        is_admin = user.get("is_admin", False)
        logger.info(
            "Successful login for user: %s from %s",
            user_data.username,
            client_ip,
        )

        access_token, refresh_token = create_auth_tokens_and_save(
            user["id"], user["username"], is_admin
        )
        return build_token_response(
            access_token, refresh_token, user["id"], user["username"], is_admin
        )

    except Exception as e:
        logger.exception("Login error: %s", e)
        return error_response("Login failed", 500)


async def refresh_access_token(request: web.Request) -> web.Response:
    refresh_request, error = await parse_request_json(
        request, RefreshTokenRequest, "Refresh token"
    )
    if error:
        return error
    assert refresh_request is not None

    logger.info("Access token refresh attempt")
    try:
        user_data = verify_refresh_token(refresh_request.refresh_token)

        user = query_db(
            "SELECT id, username, is_admin FROM users WHERE id = %s",
            (user_data["user_id"],),
            one=True,
        )

        if not user:
            return error_response("User not found", 401)

        is_admin = user.get("is_admin", False)

        # Revoke old refresh token
        old_token_hash = hashlib.sha256(
            refresh_request.refresh_token.encode()
        ).hexdigest()
        execute_write_transaction(
            "UPDATE refresh_tokens SET revoked_at = NOW() WHERE token_hash = %s",
            (old_token_hash,),
        )

        # nosemgrep: python-logger-credential-disclosure (logging username, not secret)
        logger.info("Access token refreshed for user: %s", user["username"])

        access_token, refresh_token = create_auth_tokens_and_save(
            user["id"], user["username"], is_admin
        )
        return build_token_response(
            access_token, refresh_token, user["id"], user["username"], is_admin
        )

    except web.HTTPUnauthorized as e:
        return error_response(e.reason, 401)
    except Exception as e:
        # nosemgrep: python-logger-credential-disclosure (logging error, not secret)
        logger.exception("Token refresh error: %s", e)
        return error_response("Token refresh failed", 500)


async def delete_account(request: web.Request) -> web.Response:
    try:
        current_user = await get_current_user_from_request(request)
    except web.HTTPUnauthorized as e:
        return error_response(e.reason, 401)

    logger.info(
        "Account deletion request for user: %s", current_user["username"]
    )
    try:
        result = execute_write_transaction(
            "DELETE FROM users WHERE id = %s",
            (current_user["user_id"],),
        )

        if result == 0:
            logger.warning(
                "Account deletion failed - user not found: %s",
                current_user["username"],
            )
            return error_response("User not found", 404)

        logger.info(
            "Account successfully deleted for user: %s",
            current_user["username"],
        )
        return web.json_response({"message": "Account deleted successfully"})

    except Exception as e:
        logger.exception("Account deletion error: %s", e)
        return error_response("Failed to delete account", 500)


def register_auth_routes(app: web.Application) -> None:
    app.router.add_post("/api/auth/register", register_user)
    app.router.add_post("/api/auth/login", login_user)
    app.router.add_post("/api/auth/refresh", refresh_access_token)
    app.router.add_delete("/api/auth/delete-account", delete_account)
