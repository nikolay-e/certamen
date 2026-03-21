import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

from certamen_core.tournament import EventHandler


class WebSocketEventBridge(EventHandler):
    def __init__(
        self,
        broadcast_fn: Callable[[str], Coroutine[Any, Any, None]],
    ):
        self._broadcast = broadcast_fn
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._background_tasks: set[asyncio.Task[None]] = set()

    def publish(self, event_name: str, data: dict[str, Any]) -> None:
        message = {
            "type": "tournament_event",
            "event": event_name,
            "data": data,
            "timestamp": time.time(),
        }
        task = asyncio.create_task(self._broadcast(json.dumps(message)))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        message = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        await self._broadcast(json.dumps(message))
