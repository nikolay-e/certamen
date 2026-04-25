from collections.abc import Awaitable, Callable

from aiohttp import web

from certamen_core.interfaces.web.auth.config import SKIP_AUTH
from certamen_core.interfaces.web.auth.security import (
    get_current_user_from_request,
)
from certamen_core.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)

PROTECTED_PATHS = [
    "/api/execute",
    "/api/validate",
    "/ws",
]

PUBLIC_PATHS = [
    "/api/auth/register",
    "/api/auth/login",
    "/api/auth/refresh",
    "/health",
    "/api/models",
    "/api/nodes",
    "/",
    "/assets",
]


def is_path_protected(path: str) -> bool:
    for public_path in PUBLIC_PATHS:
        if path.startswith(public_path):
            return False

    for protected_path in PROTECTED_PATHS:
        if path.startswith(protected_path):
            return True

    return False


@web.middleware  # type: ignore[misc]
async def auth_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    if SKIP_AUTH:
        logger.debug("Auth check skipped (SKIP_AUTH=true)")
        request["user"] = {
            "user_id": 0,
            "username": "dev",
            "is_admin": True,
        }
        return await handler(request)

    if not is_path_protected(request.path):
        return await handler(request)

    try:
        user = await get_current_user_from_request(request)
        request["user"] = user
        logger.debug("Authenticated user: %s", user["username"])
        return await handler(request)
    except web.HTTPUnauthorized as e:
        logger.warning(
            "Unauthorized access attempt to %s: %s", request.path, e.reason
        )
        return web.json_response(
            {"detail": e.reason},
            status=401,
        )
    except Exception as e:
        logger.exception("Auth middleware error: %s", e)
        return web.json_response(
            {"detail": "Authentication error"},
            status=500,
        )
