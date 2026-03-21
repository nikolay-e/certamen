import asyncio
import datetime
import hashlib
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import Any

import bcrypt
import jwt
from aiohttp import web

from certamen.gui.auth.config import (
    JWT_ACCESS_TOKEN_EXPIRES_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRES_DAYS,
    JWT_SECRET,
    SKIP_AUTH,
)
from certamen.gui.auth.database import query_db
from certamen.logging import get_contextual_logger

logger = get_contextual_logger(__name__)

# Auth rate limiting configuration
LOGIN_RATE_LIMIT_PER_IP = 10  # max attempts per IP per minute
LOGIN_RATE_LIMIT_PER_USERNAME = 5  # max attempts per username per minute
ACCOUNT_LOCKOUT_THRESHOLD = 5  # failed attempts before lockout
ACCOUNT_LOCKOUT_DURATION_MINUTES = 15


class AuthRateLimiter:
    """Rate limiter for authentication endpoints to prevent brute force attacks."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Track login attempts by IP
        self._ip_attempts: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=LOGIN_RATE_LIMIT_PER_IP * 2)
        )
        # Track login attempts by username
        self._username_attempts: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=LOGIN_RATE_LIMIT_PER_USERNAME * 2)
        )
        # Track failed attempts for account lockout
        self._failed_attempts: dict[str, list[float]] = defaultdict(list)
        # Track locked accounts
        self._locked_accounts: dict[str, float] = {}

    async def check_rate_limit(
        self, client_ip: str, username: str
    ) -> tuple[bool, str | None]:
        """Check if login attempt is allowed.

        Returns:
            (allowed, reason) - True if allowed, or (False, reason) if blocked
        """
        now = time.time()
        async with self._lock:
            # Check account lockout first
            if username in self._locked_accounts:
                lockout_until = self._locked_accounts[username]
                if now < lockout_until:
                    remaining = int(lockout_until - now)
                    return (
                        False,
                        f"Account locked. Try again in {remaining} seconds",
                    )
                else:
                    # Lockout expired
                    del self._locked_accounts[username]
                    self._failed_attempts[username].clear()

            # Check IP rate limit
            ip_timestamps = self._ip_attempts[client_ip]
            ip_recent = sum(1 for ts in ip_timestamps if now - ts < 60)
            if ip_recent >= LOGIN_RATE_LIMIT_PER_IP:
                logger.warning("IP rate limit exceeded: %s", client_ip)
                return False, "Too many login attempts from this IP"

            # Check username rate limit
            username_timestamps = self._username_attempts[username]
            username_recent = sum(
                1 for ts in username_timestamps if now - ts < 60
            )
            if username_recent >= LOGIN_RATE_LIMIT_PER_USERNAME:
                logger.warning("Username rate limit exceeded: %s", username)
                return False, "Too many login attempts for this account"

            # Record this attempt
            ip_timestamps.append(now)
            username_timestamps.append(now)

            return True, None

    async def record_failed_attempt(self, username: str) -> None:
        """Record a failed login attempt and potentially lock the account."""
        now = time.time()
        async with self._lock:
            # Remove old failed attempts (older than lockout duration)
            window = ACCOUNT_LOCKOUT_DURATION_MINUTES * 60
            self._failed_attempts[username] = [
                ts
                for ts in self._failed_attempts[username]
                if now - ts < window
            ]
            self._failed_attempts[username].append(now)

            # Check if should lock account
            if (
                len(self._failed_attempts[username])
                >= ACCOUNT_LOCKOUT_THRESHOLD
            ):
                lockout_until = now + (ACCOUNT_LOCKOUT_DURATION_MINUTES * 60)
                self._locked_accounts[username] = lockout_until
                logger.warning(
                    "Account locked due to failed attempts: %s (until %s)",
                    username,
                    datetime.datetime.fromtimestamp(lockout_until).isoformat(),
                )

    async def record_successful_login(self, username: str) -> None:
        """Clear failed attempts after successful login."""
        async with self._lock:
            self._failed_attempts.pop(username, None)
            self._locked_accounts.pop(username, None)

    async def cleanup_stale_entries(self) -> None:
        """Remove stale entries to prevent memory leaks."""
        now = time.time()
        window = 300  # 5 minutes
        async with self._lock:
            # Cleanup IP attempts
            stale_ips = [
                ip
                for ip, timestamps in self._ip_attempts.items()
                if not timestamps or (now - max(timestamps)) > window
            ]
            for ip in stale_ips:
                del self._ip_attempts[ip]

            # Cleanup username attempts
            stale_usernames = [
                username
                for username, timestamps in self._username_attempts.items()
                if not timestamps or (now - max(timestamps)) > window
            ]
            for username in stale_usernames:
                del self._username_attempts[username]

            # Cleanup expired lockouts
            expired_lockouts = [
                username
                for username, lockout_until in self._locked_accounts.items()
                if now > lockout_until
            ]
            for username in expired_lockouts:
                del self._locked_accounts[username]
                self._failed_attempts.pop(username, None)


# Global rate limiter instance
auth_rate_limiter = AuthRateLimiter()


# Pre-computed bcrypt hash for timing attack prevention
# This hash is used when user doesn't exist to ensure constant-time response
# The actual password doesn't matter - we just need to run bcrypt
DUMMY_HASH_FOR_TIMING_ATTACK_PREVENTION = (
    "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.5qLB6kGGGGGGGG"
)


def hash_password(password: str) -> str:
    hashed: bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    result: bool = bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )
    return result


def create_access_token(data: dict[str, Any]) -> str:
    to_encode = data.copy()
    now = datetime.datetime.now(datetime.UTC)
    expire = now + datetime.timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRES_MINUTES)
    to_encode.update(
        {"exp": expire, "iat": now, "nbf": now, "jti": str(uuid.uuid4())}
    )
    encoded: str = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded


def create_refresh_token() -> tuple[str, str, datetime.datetime]:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
        days=JWT_REFRESH_TOKEN_EXPIRES_DAYS
    )
    return token, token_hash, expires_at


def verify_refresh_token(token: str) -> dict[str, Any]:
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    token_data = query_db(
        """
        SELECT user_id, expires_at, revoked_at
        FROM refresh_tokens
        WHERE token_hash = %s
        """,
        (token_hash,),
        one=True,
    )

    if not token_data:
        raise web.HTTPUnauthorized(reason="Invalid refresh token")

    if token_data["revoked_at"] is not None:
        raise web.HTTPUnauthorized(reason="Refresh token has been revoked")

    if token_data["expires_at"] < datetime.datetime.now(datetime.UTC):
        raise web.HTTPUnauthorized(reason="Refresh token has expired")

    return {"user_id": token_data["user_id"]}


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("userId")
        username = payload.get("sub")
        is_admin = payload.get("isAdmin", False)

        if user_id is None or username is None:
            raise web.HTTPUnauthorized(reason="Invalid token payload")

        return {"user_id": user_id, "username": username, "is_admin": is_admin}

    except jwt.ExpiredSignatureError as e:
        raise web.HTTPUnauthorized(reason="Token has expired") from e
    except jwt.PyJWTError as e:
        raise web.HTTPUnauthorized(reason="Invalid token") from e


async def get_current_user_from_request(
    request: web.Request,
) -> dict[str, Any]:
    if SKIP_AUTH:
        return {
            "user_id": 0,
            "username": "dev",
            "is_admin": True,
        }

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise web.HTTPUnauthorized(reason="Missing authorization header")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise web.HTTPUnauthorized(
            reason="Invalid authorization header format"
        )

    token = parts[1]
    return verify_access_token(token)


async def require_admin(request: web.Request) -> dict[str, Any]:
    current_user = await get_current_user_from_request(request)

    user = query_db(
        "SELECT is_admin FROM users WHERE id = %s",
        (current_user["user_id"],),
        one=True,
    )

    if not user or not user.get("is_admin"):
        raise web.HTTPForbidden(reason="Admin access required")

    return current_user
