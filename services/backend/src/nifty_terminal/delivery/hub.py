"""Bounded in-process fan-out for WebSocket market-state delivery."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from nifty_terminal.delivery.models import MarketStateView


@dataclass(frozen=True, slots=True)
class SequencedMarketState:
    sequence: int
    sent_at: datetime
    view: MarketStateView


class MarketStateHub:
    def __init__(self, *, subscriber_queue_size: int = 4) -> None:
        self._queue_size = subscriber_queue_size
        self._subscribers: dict[str, set[asyncio.Queue[SequencedMarketState]]] = {}
        self._sequence: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def publish(self, view: MarketStateView) -> SequencedMarketState:
        instrument_id = view.snapshot.instrument_id
        async with self._lock:
            sequence = self._sequence.get(instrument_id, 0) + 1
            self._sequence[instrument_id] = sequence
            message = SequencedMarketState(
                sequence=sequence,
                sent_at=datetime.now(timezone.utc),
                view=view,
            )
            for queue in tuple(self._subscribers.get(instrument_id, set())):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(message)
            return message

    @asynccontextmanager
    async def subscribe(
        self, instrument_id: str
    ) -> AsyncIterator[asyncio.Queue[SequencedMarketState]]:
        queue: asyncio.Queue[SequencedMarketState] = asyncio.Queue(self._queue_size)
        async with self._lock:
            self._subscribers.setdefault(instrument_id, set()).add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(instrument_id)
                if subscribers is not None:
                    subscribers.discard(queue)
                    if not subscribers:
                        self._subscribers.pop(instrument_id, None)
