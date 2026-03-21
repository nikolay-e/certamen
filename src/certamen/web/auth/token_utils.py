from aiohttp import web

from certamen.api.schemas import TokenResponse, UserResponse
from certamen.gui.auth.database import execute_write_transaction
from certamen.gui.auth.security import (
    create_access_token,
    create_refresh_token,
)


def create_auth_tokens_and_save(
    user_id: int, username: str, is_admin: bool
) -> tuple[str, str]:
    """Create access and refresh tokens for user and save refresh token to database.

    Args:
        user_id: User ID
        username: Username
        is_admin: Whether user is admin

    Returns:
        Tuple of (access_token, refresh_token)
    """
    access_token = create_access_token(
        data={
            "userId": user_id,
            "sub": username,
            "isAdmin": is_admin,
        }
    )

    refresh_token, token_hash, expires_at = create_refresh_token()
    execute_write_transaction(
        "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (%s, %s, %s)",
        (user_id, token_hash, expires_at),
    )

    return access_token, refresh_token


def build_token_response(
    access_token: str,
    refresh_token: str,
    user_id: int,
    username: str,
    is_admin: bool,
    status: int = 200,
) -> web.Response:
    """Build standardized token response.

    Args:
        access_token: JWT access token
        refresh_token: JWT refresh token
        user_id: User ID
        username: Username
        is_admin: Whether user is admin
        status: HTTP status code (default: 200)

    Returns:
        JSON response with token data
    """
    response = TokenResponse(
        token=access_token,
        refresh_token=refresh_token,
        expires_in="15m",
        user=UserResponse(
            id=user_id,
            username=username,
            is_admin=is_admin,
        ),
    )
    return web.json_response(response.model_dump(), status=status)
