from typing import Any

from aiohttp import web

from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


async def send_ws_error(
    ws: web.WebSocketResponse,
    message: str,
    code: str | None = None,
    **extra_fields: Any,
) -> bool:
    error_payload: dict[str, Any] = {"type": "error", "message": message}
    if code:
        error_payload["code"] = code
    error_payload.update(extra_fields)
    try:
        await ws.send_json(error_payload)
        return True
    except ConnectionResetError:
        logger.warning(
            "Cannot send error to client (connection reset): %s", message
        )
        return False
    except Exception as e:
        logger.warning("Failed to send error to client: %s", e)
        return False


async def send_ws_json(
    ws: web.WebSocketResponse,
    payload: dict[str, Any],
) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except ConnectionResetError:
        logger.warning(
            "Cannot send to client (connection reset): type=%s",
            payload.get("type"),
        )
        return False
    except Exception as e:
        logger.warning("Failed to send to client: %s", e)
        return False
