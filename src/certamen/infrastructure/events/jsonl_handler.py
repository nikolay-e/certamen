import json
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from certamen.ports.tournament import EventHandler

EVENT_SCHEMA_VERSION = 1


def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    return f"{timestamp}_{suffix}"


class JsonlEventHandler(EventHandler):
    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._seq = 0
        self._lock = threading.Lock()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self._fh = self.events_path.open("a", encoding="utf-8")

    def publish(self, event_name: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "seq": self._seq,
                "ts": time.time(),
                "run_id": self.run_id,
                "event_type": event_name,
                "payload": data,
            }
            self._fh.write(json.dumps(event, default=_safe_json) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JsonlEventHandler":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _safe_json(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
