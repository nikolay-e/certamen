import asyncio
import json
import os
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from weakref import WeakSet

from aiohttp import web

from certamen.application.execution.async_executor import AsyncExecutor
from certamen.application.workflow.nodes import (
    register_all as _register_all_nodes,
)
from certamen.application.workflow.registry import registry
from certamen.infrastructure.config.defaults import get_defaults
from certamen.infrastructure.config.env import (
    get_comma_separated_env,
    get_int_env,
)
from certamen.interfaces.web.websocket_utils import (
    send_ws_error,
    send_ws_json,
)
from certamen.shared.logging import get_contextual_logger

_register_all_nodes()

logger = get_contextual_logger(__name__)

MAX_WEBSOCKET_CONNECTIONS = get_int_env(
    "CERTAMEN_MAX_WEBSOCKET_CONNECTIONS", "1000"
)
MAX_CONNECTIONS_PER_IP = get_int_env("CERTAMEN_MAX_CONNECTIONS_PER_IP", "10")
WEBSOCKET_MESSAGE_RATE_LIMIT_PER_SECOND = get_int_env(
    "CERTAMEN_WEBSOCKET_MESSAGE_RATE_LIMIT_PER_SECOND", "10"
)
WEBSOCKET_MESSAGE_RATE_LIMIT_PER_MINUTE = get_int_env(
    "CERTAMEN_WEBSOCKET_MESSAGE_RATE_LIMIT_PER_MINUTE", "100"
)
ALLOWED_ORIGINS = get_comma_separated_env(
    "CERTAMEN_ALLOWED_ORIGINS", "http://localhost:8765"
)
TRUSTED_PROXIES = get_comma_separated_env("CERTAMEN_TRUSTED_PROXIES", "")


class GUIServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):  # noqa: S104
        self.host = host
        self.port = port
        self.app = web.Application()
        self.clients: WeakSet[web.WebSocketResponse] = WeakSet()
        self.connections_per_ip: dict[str, int] = defaultdict(int)
        self.rate_limit_tracker: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=WEBSOCKET_MESSAGE_RATE_LIMIT_PER_MINUTE)
        )
        self._gui_dir: Path | None = None
        self.executor = AsyncExecutor(
            broadcast_fn=self.broadcast,
            config=get_defaults(),
        )
        self.connection_start_times: dict[int, float] = {}
        self.messages_received: dict[int, int] = defaultdict(int)
        self.messages_sent: dict[int, int] = defaultdict(int)
        self.broadcast_success_count: int = 0
        self.broadcast_failure_count: int = 0
        self.total_messages_received: int = 0
        self.last_rate_log_time: float = time.time()
        self.last_rate_log_count: int = 0
        # Locks for thread-safe connection tracking (prevents race conditions)
        self._clients_lock = asyncio.Lock()
        self._connections_lock = asyncio.Lock()
        self._rate_limit_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._setup_auth()
        self._setup_routes()
        self.app.on_startup.append(self._on_startup)
        self.app.on_shutdown.append(self._on_shutdown)

    def _setup_auth(self) -> None:
        from certamen.interfaces.web.auth import (
            auth_middleware,
            register_auth_routes,
        )

        self.app.middlewares.append(self._security_headers_middleware)
        self.app.middlewares.append(auth_middleware)
        register_auth_routes(self.app)

    @web.middleware  # type: ignore[misc]
    async def _security_headers_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        response = await handler(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';"
        )
        return response

    async def _on_startup(self, _app: web.Application) -> None:
        # Start periodic cleanup task for rate limiter memory management
        self._cleanup_task = asyncio.create_task(
            self._periodic_rate_limit_cleanup(), name="rate_limit_cleanup"
        )
        logger.info("Server startup complete")

    async def _periodic_rate_limit_cleanup(self) -> None:
        """Periodically cleanup stale entries from rate_limit_tracker to prevent memory leaks."""
        cleanup_interval = 60  # seconds
        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_stale_rate_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Rate limit cleanup error: %s", e)

    async def _cleanup_stale_rate_limits(self) -> None:
        """Remove entries with no recent activity from rate_limit_tracker."""
        now = time.time()
        window = 60  # Keep entries with activity in last 60 seconds
        async with self._rate_limit_lock:
            stale_keys = [
                ip
                for ip, timestamps in self.rate_limit_tracker.items()
                if not timestamps or (now - max(timestamps)) > window
            ]
            for key in stale_keys:
                del self.rate_limit_tracker[key]
            if stale_keys:
                logger.debug(
                    "Cleaned up %d stale rate limiter entries", len(stale_keys)
                )

    async def _on_shutdown(self, _app: web.Application) -> None:
        from certamen.interfaces.web.auth.database import cleanup_db_pool

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all WebSocket connections gracefully
        async with self._clients_lock:
            clients_snapshot = list(self.clients)
        if clients_snapshot:
            await asyncio.gather(
                *[
                    ws.close(code=1001, message=b"Server shutdown")
                    for ws in clients_snapshot
                ],
                return_exceptions=True,
            )

        cleanup_db_pool()
        logger.info("Server shutdown complete")

    def _check_origin(self, request: web.Request) -> bool:
        origin = request.headers.get("Origin", "")
        if not origin:
            return False
        return origin in ALLOWED_ORIGINS

    def _get_client_ip(self, request: web.Request) -> str:
        peername = (
            request.transport.get_extra_info("peername")
            if request.transport
            else None
        )
        remote_ip: str | None = peername[0] if peername else None

        if remote_ip and TRUSTED_PROXIES and remote_ip in TRUSTED_PROXIES:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip: str = forwarded.split(",")[0].strip()
                logger.debug(
                    "Trusting X-Forwarded-For from proxy %s: %s",
                    remote_ip,
                    client_ip,
                )
                return client_ip

        return str(request.remote or "unknown")

    async def _check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        async with self._rate_limit_lock:
            timestamps = self.rate_limit_tracker[client_ip]

            messages_per_second = sum(1 for ts in timestamps if now - ts < 1)
            messages_per_minute = len(timestamps)

            if (
                messages_per_second >= WEBSOCKET_MESSAGE_RATE_LIMIT_PER_SECOND
                or messages_per_minute
                >= WEBSOCKET_MESSAGE_RATE_LIMIT_PER_MINUTE
            ):
                return False

            timestamps.append(now)
            return True

    def _setup_routes(self) -> None:
        self.app.router.add_get("/ws", self.websocket_handler)
        self.app.router.add_get("/api/models", self.get_models)
        self.app.router.add_get(
            "/api/models/{provider}", self.get_provider_models
        )
        self.app.router.add_get("/api/nodes", self.get_nodes)
        self.app.router.add_post("/api/validate", self.validate_workflow)
        self.app.router.add_post("/api/execute", self.execute_workflow)
        self.app.router.add_get("/health", self.health_check)

        # Try multiple paths for gui/dist (handles both dev and installed package)
        gui_dir = None
        candidate_paths = [
            Path(os.environ.get("CERTAMEN_GUI_DIR", "")) / "dist",
            Path("/app/gui/dist"),  # Docker container path
            Path(__file__).parent.parent.parent.parent
            / "gui"
            / "dist",  # Dev path
        ]
        for path in candidate_paths:
            if path.exists() and (path / "index.html").exists():
                gui_dir = path
                break

        if gui_dir:
            self.app.router.add_get("/", self._serve_index)
            self.app.router.add_static(
                "/assets", gui_dir / "assets", name="assets"
            )
            self._gui_dir = gui_dir

    async def _serve_index(
        self, request: web.Request
    ) -> web.FileResponse | web.Response:
        if self._gui_dir is None:
            return web.Response(text="GUI not configured")
        index_path = self._gui_dir / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="GUI not built. Run: cd gui && npm run build")

    async def broadcast(self, message: str) -> None:
        # Take snapshot of clients under lock to avoid race conditions
        async with self._clients_lock:
            if not self.clients:
                return
            clients_list = list(self.clients)

        # Log broadcast message type (parse first to get type)
        try:
            msg_data = json.loads(message)
            msg_type = msg_data.get("type", "unknown")
            logger.debug(
                "Broadcasting to %d client(s): type=%s",
                len(clients_list),
                msg_type,
            )
        except json.JSONDecodeError:
            logger.debug(
                "Broadcasting raw message to %d client(s)", len(clients_list)
            )
        results = await asyncio.gather(
            *[client.send_str(message) for client in clients_list],
            return_exceptions=True,
        )
        success_count = 0
        failure_count = 0
        for client, result in zip(clients_list, results, strict=True):
            if isinstance(result, Exception):
                failure_count += 1
                self.broadcast_failure_count += 1
                peer = (
                    client._req.transport.get_extra_info("peername")
                    if client._req and client._req.transport
                    else None
                )
                client_id = f"{peer[0]}:{peer[1]}" if peer else "unknown"
                logger.warning("Broadcast failed to %s: %s", client_id, result)
            else:
                success_count += 1
                self.broadcast_success_count += 1
                ws_id = id(client)
                self.messages_sent[ws_id] += 1

        if failure_count > 0:
            logger.info(
                "Broadcast complete: success=%d failure=%d total_success=%d total_failure=%d",
                success_count,
                failure_count,
                self.broadcast_success_count,
                self.broadcast_failure_count,
            )

    async def websocket_handler(
        self, request: web.Request
    ) -> web.WebSocketResponse:
        from certamen.interfaces.web.auth.config import SKIP_AUTH
        from certamen.interfaces.web.auth.security import (
            verify_access_token,
        )

        client_ip = self._get_client_ip(request)

        if not SKIP_AUTH and not self._check_origin(request):
            logger.warning(
                "WebSocket connection rejected: invalid origin from %s",
                client_ip,
            )
            return web.Response(status=403, text="Origin not allowed")

        if len(self.clients) >= MAX_WEBSOCKET_CONNECTIONS:
            logger.warning(
                "WebSocket connection rejected: max connections reached (%d)",
                MAX_WEBSOCKET_CONNECTIONS,
            )
            return web.Response(
                status=503, text="Maximum WebSocket connections reached"
            )

        if self.connections_per_ip[client_ip] >= MAX_CONNECTIONS_PER_IP:
            logger.warning(
                "WebSocket connection rejected: max connections per IP (%s)",
                client_ip,
            )
            return web.Response(
                status=429, text="Too many connections from this IP"
            )

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        if not SKIP_AUTH:
            try:
                auth_msg = await asyncio.wait_for(
                    ws.receive_json(), timeout=10.0
                )
                if auth_msg.get("type") != "auth":
                    raise ValueError("First message must be auth")

                token = auth_msg.get("token")
                if not token:
                    raise ValueError("Missing token")

                user_info = verify_access_token(token)
                user_id = user_info.get("user_id")
                if not user_id:
                    raise ValueError("Invalid token")

                await send_ws_json(ws, {"type": "auth_success"})
                logger.info("WebSocket authenticated for user %s", user_id)

            except TimeoutError:
                logger.warning("WebSocket auth timeout from %s", client_ip)
                await ws.close(code=4001, message=b"Authentication timeout")
                return ws
            except Exception as e:
                logger.warning(
                    "WebSocket auth failed from %s: %s", client_ip, e
                )
                await ws.close(code=4003, message=b"Authentication failed")
                return ws

        async with self._clients_lock:
            self.clients.add(ws)
        async with self._connections_lock:
            self.connections_per_ip[client_ip] += 1
        ws_id = id(ws)
        self.connection_start_times[ws_id] = time.time()
        async with self._clients_lock:
            active_connections = len(self.clients)
        logger.info(
            "WebSocket connected: client=%s active_connections=%d",
            client_ip,
            active_connections,
        )

        disconnect_reason = "normal"
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    self.messages_received[ws_id] += 1
                    self.total_messages_received += 1

                    if self.total_messages_received % 100 == 0:
                        elapsed = time.time() - self.last_rate_log_time
                        messages_in_period = (
                            self.total_messages_received
                            - self.last_rate_log_count
                        )
                        rate_per_minute = (
                            (messages_in_period / elapsed) * 60
                            if elapsed > 0
                            else 0
                        )
                        logger.info(
                            "Message rate: total=%d period_messages=%d period_seconds=%.1f rate_per_minute=%.1f active_connections=%d",
                            self.total_messages_received,
                            messages_in_period,
                            elapsed,
                            rate_per_minute,
                            len(self.clients),
                        )
                        self.last_rate_log_time = time.time()
                        self.last_rate_log_count = self.total_messages_received

                    if not await self._check_rate_limit(client_ip):
                        logger.warning("Rate limit exceeded for %s", client_ip)
                        await send_ws_error(
                            ws, "Rate limit exceeded", code="RATE_LIMIT"
                        )
                        continue

                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(ws, data)
                    except json.JSONDecodeError as e:
                        logger.error("Invalid JSON from %s: %s", client_ip, e)
                        await send_ws_error(
                            ws, "Invalid JSON format", code="INVALID_JSON"
                        )
                    except Exception as e:
                        logger.exception(
                            "Error handling message from %s: %s",
                            client_ip,
                            e,
                        )
                        await send_ws_error(
                            ws, f"Internal error: {e}", code="INTERNAL_ERROR"
                        )
                elif msg.type == web.WSMsgType.ERROR:
                    disconnect_reason = f"error: {ws.exception()}"
                    logger.error(
                        "WebSocket error from %s: %s",
                        client_ip,
                        ws.exception(),
                    )
                elif msg.type == web.WSMsgType.CLOSE:
                    disconnect_reason = f"client closed (code={ws.close_code})"
        except asyncio.CancelledError:
            disconnect_reason = "cancelled"
            raise
        except Exception as e:
            disconnect_reason = f"exception: {e}"
            logger.exception("WebSocket loop error from %s: %s", client_ip, e)
        finally:
            async with self._clients_lock:
                self.clients.discard(ws)
            async with self._connections_lock:
                self.connections_per_ip[client_ip] -= 1
                if self.connections_per_ip[client_ip] <= 0:
                    del self.connections_per_ip[client_ip]

            duration = time.time() - self.connection_start_times.get(
                ws_id, time.time()
            )
            messages_sent = self.messages_sent.get(ws_id, 0)
            messages_received = self.messages_received.get(ws_id, 0)
            async with self._clients_lock:
                active_connections = len(self.clients)

            logger.info(
                "WebSocket disconnected: client=%s duration_s=%.1f messages_sent=%d messages_received=%d active_connections=%d reason=%s",
                client_ip,
                duration,
                messages_sent,
                messages_received,
                active_connections,
                disconnect_reason,
            )

            self.connection_start_times.pop(ws_id, None)
            self.messages_sent.pop(ws_id, None)
            self.messages_received.pop(ws_id, None)

        return ws

    def _validate_message(self, message: dict[str, Any]) -> tuple[bool, str]:
        msg_type = message.get("type", "")

        if not msg_type or not isinstance(msg_type, str):
            return False, "Missing or invalid message type"

        if msg_type in {"validate", "execute"}:
            nodes = message.get("nodes", [])
            edges = message.get("edges", [])

            if not isinstance(nodes, list):
                return False, "nodes must be an array"

            if not isinstance(edges, list):
                return False, "edges must be an array"

            if len(nodes) > 1000:
                return False, "Too many nodes (max 1000)"

            if len(edges) > 10000:
                return False, "Too many edges (max 10000)"

        return True, ""

    async def _handle_message(
        self,
        ws: web.WebSocketResponse,
        message: dict[str, Any],
    ) -> None:
        valid, error = self._validate_message(message)
        if not valid:
            logger.warning("WebSocket message validation failed: %s", error)
            await send_ws_error(ws, error, code="VALIDATION_ERROR")
            return

        msg_type = message.get("type", "")
        logger.debug("WebSocket message received: type=%s", msg_type)

        if msg_type == "get_models":
            from certamen.application.workflow.nodes.llm import (
                get_models_by_provider,
            )

            # Get Ollama models dynamically from local Ollama server
            ollama_models = await get_models_by_provider("ollama")

            # Build response with real installed models
            models_data = {}
            if ollama_models:
                for model_name in ollama_models:
                    # Extract display name from full model name (e.g., "ollama/llama3:8b" -> "llama3:8b")
                    display_name = model_name.replace("ollama/", "")
                    models_data[model_name] = {
                        "display_name": display_name,
                        "provider": "ollama",
                    }
                logger.info(
                    "Sending %d Ollama models to client: %s...",
                    len(models_data),
                    list(models_data.keys())[:5],
                )
            else:
                # No Ollama models available - return empty dict
                logger.warning(
                    "No Ollama models found. Ensure Ollama is running and OLLAMA_BASE_URL is set."
                )
                models_data = {}

            await send_ws_json(ws, {"type": "models", "data": models_data})

        elif msg_type == "get_nodes":
            await send_ws_json(
                ws, {"type": "nodes", "data": registry.list_by_category()}
            )

        elif msg_type == "get_node_definition":
            node_type = message.get("node_type")
            if not node_type:
                await send_ws_error(
                    ws, "node_type is required", code="MISSING_PARAMETER"
                )
                return

            node_class = registry.get(node_type)
            if not node_class:
                await send_ws_error(
                    ws,
                    f"Node type '{node_type}' not found",
                    code="NODE_NOT_FOUND",
                )
                return

            await send_ws_json(
                ws,
                {
                    "type": "node_definition",
                    "data": node_class(node_id="schema").get_schema(),
                },
            )

        elif msg_type == "validate":
            nodes = message.get("nodes", [])
            edges = message.get("edges", [])
            result = self.executor.validate(nodes, edges)
            await send_ws_json(ws, {"type": "validated", **result})

        elif msg_type == "execute":
            nodes = message.get("nodes", [])
            edges = message.get("edges", [])
            logger.info(
                "WebSocket execute request: %d nodes, %d edges",
                len(nodes),
                len(edges),
            )
            logger.debug(
                "Execute workflow nodes: %s",
                [n.get("id") + "(" + n.get("type", "") + ")" for n in nodes],
            )
            result = await self.executor.execute(nodes, edges)
            await send_ws_json(ws, {"type": "execution_result", **result})

        elif msg_type == "cancel_execution":
            cancelled = self.executor.cancel()
            await send_ws_json(
                ws,
                {
                    "type": "cancel_acknowledged",
                    "cancelled": cancelled,
                },
            )

        elif msg_type == "cancel":
            execution_id = message.get("execution_id")
            await send_ws_json(
                ws, {"type": "cancelled", "execution_id": execution_id}
            )

        else:
            logger.warning("Unknown message type: %s", msg_type)
            await send_ws_error(ws, f"Unknown message type: {msg_type}")

    async def get_models(self, request: web.Request) -> web.Response:
        """Get available models - queries Ollama dynamically."""
        from certamen.application.workflow.nodes.llm import (
            get_models_by_provider,
        )

        ollama_models = await get_models_by_provider("ollama")
        models_data = {}
        for model_name in ollama_models:
            display_name = model_name.replace("ollama/", "")
            models_data[model_name] = {
                "display_name": display_name,
                "provider": "ollama",
                "model_name": model_name,
            }
        return web.json_response(models_data)

    async def get_provider_models(self, request: web.Request) -> web.Response:
        from certamen.application.workflow.nodes.llm import (
            get_models_by_provider,
        )

        provider = request.match_info.get("provider", "")
        models = await get_models_by_provider(provider)
        return web.json_response({"provider": provider, "models": models})

    async def get_nodes(self, request: web.Request) -> web.Response:
        return web.json_response(registry.list_by_category())

    async def validate_workflow(self, request: web.Request) -> web.Response:
        data = await request.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        result = self.executor.validate(nodes, edges)
        return web.json_response(result)

    async def execute_workflow(self, request: web.Request) -> web.Response:
        data = await request.json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        validation_result = self.executor.validate(nodes, edges)
        if not validation_result["valid"]:
            return web.json_response(
                {
                    "error": "Workflow validation failed",
                    "errors": validation_result["errors"],
                    "warnings": validation_result.get("warnings", []),
                },
                status=400,
            )

        result = await self.executor.execute(nodes, edges)
        return web.json_response(result)

    async def health_check(self, request: web.Request) -> web.Response:
        total_messages_sent = sum(self.messages_sent.values())
        total_messages_received = sum(self.messages_received.values())
        active_connections = len(self.clients)

        return web.json_response(
            {
                "status": "healthy",
                "models_discovery": "dynamic",
                "nodes_available": len(registry.list_nodes()),
                "websocket": {
                    "active_connections": active_connections,
                    "total_messages_sent": total_messages_sent,
                    "total_messages_received": total_messages_received,
                    "broadcast_success": self.broadcast_success_count,
                    "broadcast_failure": self.broadcast_failure_count,
                },
            }
        )

    def run(self) -> None:
        web.run_app(self.app, host=self.host, port=self.port)


async def run_gui_server(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 8765,
) -> None:
    server = GUIServer(host=host, port=port)
    runner = web.AppRunner(server.app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info("GUI server running at http://%s:%d", host, port)
    logger.info("WebSocket endpoint: ws://%s:%d/ws", host, port)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
