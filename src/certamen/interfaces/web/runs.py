import asyncio
import json
import os
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from certamen.shared.logging import get_contextual_logger

logger = get_contextual_logger(__name__)


def _runs_root() -> Path:
    base = Path(os.environ.get("CERTAMEN_OUTPUTS_DIR", "outputs"))
    return base / "runs"


def _read_events_from_offset(
    events_path: Path, from_seq: int = 0
) -> tuple[list[dict[str, Any]], int]:
    if not events_path.exists():
        return [], 0
    events: list[dict[str, Any]] = []
    size = events_path.stat().st_size
    with events_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("seq", 0) > from_seq:
                events.append(event)
    return events, size


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    summary: dict[str, Any] = {
        "run_id": run_dir.name,
        "status": "unknown",
        "question": None,
        "champion": None,
        "total_cost": 0.0,
        "model_count": 0,
        "event_count": 0,
        "started_at": None,
        "ended_at": None,
    }
    if not events_path.exists():
        summary["status"] = "missing"
        return summary

    started = False
    ended = False
    for raw in events_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary["event_count"] += 1
        et = event.get("event_type", "")
        p = event.get("payload", {})
        ts = event.get("ts", 0)
        if et == "tournament_started":
            started = True
            summary["question"] = p.get("question")
            summary["model_count"] = len(p.get("models", []))
            summary["started_at"] = ts
        elif et == "llm_response":
            summary["total_cost"] += float(p.get("cost") or 0.0)
        elif et == "tournament_ended":
            ended = True
            summary["champion"] = p.get("champion")
            summary["ended_at"] = ts
            if p.get("total_cost") is not None:
                summary["total_cost"] = float(p["total_cost"])

    if ended:
        summary["status"] = "completed"
    elif started:
        summary["status"] = "running"
    else:
        summary["status"] = "empty"
    return summary


async def list_runs(_request: web.Request) -> web.Response:
    root = _runs_root()
    if not root.exists():
        return web.json_response({"runs": []})
    runs = []
    for run_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if run_dir.is_dir():
            try:
                runs.append(_summarize_run(run_dir))
            except Exception as e:
                logger.warning("Failed to summarize run %s: %s", run_dir, e)
    return web.json_response({"runs": runs})


async def get_run(request: web.Request) -> web.Response:
    run_id = request.match_info["run_id"]
    run_dir = _runs_root() / run_id
    if not run_dir.is_dir():
        return web.json_response({"error": "run not found"}, status=404)
    return web.json_response(_summarize_run(run_dir))


async def get_run_events(request: web.Request) -> web.Response:
    run_id = request.match_info["run_id"]
    run_dir = _runs_root() / run_id
    if not run_dir.is_dir():
        return web.json_response({"error": "run not found"}, status=404)

    from_seq = int(request.query.get("from_seq", "0"))
    limit = int(request.query.get("limit", "10000"))

    events, _ = _read_events_from_offset(run_dir / "events.jsonl", from_seq)
    return web.json_response({"events": events[:limit]})


async def attach_run_websocket(
    request: web.Request,
) -> web.WebSocketResponse:
    run_id = request.match_info["run_id"]
    run_dir = _runs_root() / run_id
    events_path = run_dir / "events.jsonl"

    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)

    if not run_dir.is_dir():
        await ws.send_json({"type": "error", "message": "run not found"})
        await ws.close()
        return ws

    from_seq = int(request.query.get("from_seq", "0"))
    last_seq = from_seq
    last_size = 0

    # Initial replay
    events, last_size = _read_events_from_offset(events_path, last_seq)
    for event in events:
        try:
            await ws.send_json({"type": "event", "event": event})
            last_seq = event.get("seq", last_seq)
        except ConnectionResetError:
            return ws

    # Live tail loop
    poll_interval = 0.2
    idle_timeout = 600  # 10 min without changes → close
    idle_seconds = 0.0

    async def consumer() -> None:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break

    consumer_task = asyncio.create_task(consumer())

    try:
        while not ws.closed:
            await asyncio.sleep(poll_interval)
            if not events_path.exists():
                continue
            try:
                size = events_path.stat().st_size
            except OSError:
                continue
            if size == last_size:
                idle_seconds += poll_interval
                if idle_seconds >= idle_timeout:
                    break
                continue
            idle_seconds = 0
            new_events, new_size = _read_events_from_offset(
                events_path, last_seq
            )
            last_size = new_size
            for event in new_events:
                try:
                    await ws.send_json({"type": "event", "event": event})
                    last_seq = event.get("seq", last_seq)
                except ConnectionResetError:
                    break
            # Auto-close on tournament_ended
            if any(
                e.get("event_type") == "tournament_ended" for e in new_events
            ):
                await ws.send_json({"type": "ended", "last_seq": last_seq})
                break
    finally:
        consumer_task.cancel()
        if not ws.closed:
            await ws.close()
    return ws


def register_runs_routes(app: web.Application) -> None:
    app.router.add_get("/api/runs", list_runs)
    app.router.add_get("/api/runs/{run_id}", get_run)
    app.router.add_get("/api/runs/{run_id}/events", get_run_events)
    app.router.add_get("/api/runs/{run_id}/attach", attach_run_websocket)
