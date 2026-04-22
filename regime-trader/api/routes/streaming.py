"""Server-Sent Events stream for dashboard + agent listeners (Phase B5)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.dependencies import get_service
from api.services import PlatformEvent, PlatformService

router = APIRouter(tags=["streaming"])


async def _event_stream(request: Request, service: PlatformService) -> AsyncIterator[bytes]:
    queue: asyncio.Queue[PlatformEvent] = asyncio.Queue(maxsize=200)
    loop = asyncio.get_event_loop()

    def _listener(event: PlatformEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    service.register_listener(_listener)
    try:
        # Emit an initial comment frame so clients (and tests) observe the
        # connection is live immediately, even when the event queue is empty.
        yield b": connected\n\n"
        # Replay recent events so late connectors see prior state.
        for cached in service.recent_events(limit=20):
            yield f"data: {json.dumps(cached)}\n\n".encode("utf-8")
        while True:
            if await request.is_disconnected():
                break
            try:
                # Short poll so we observe client disconnects promptly. The
                # keep-alive frame keeps the connection alive for long-lived
                # SSE listeners when the event queue stays idle.
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                yield b": keep-alive\n\n"
                continue
            payload = {
                "event": event.event,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
            }
            yield f"event: {event.event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
    finally:
        if _listener in service.listeners:
            service.listeners.remove(_listener)


@router.get("/events/stream")
async def stream_events(request: Request, service: PlatformService = Depends(get_service)) -> StreamingResponse:
    return StreamingResponse(_event_stream(request, service), media_type="text/event-stream")


@router.get("/events/recent")
def recent_events(service: PlatformService = Depends(get_service)) -> dict:
    return {"events": service.recent_events(limit=100), "generated_at": datetime.now(timezone.utc).isoformat()}
