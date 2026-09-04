"""Append-only canonical event ledger interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from nifty_terminal.domain.market_event import CanonicalMarketEvent


class EventLedger(Protocol):
    def append(self, event: CanonicalMarketEvent) -> None: ...

    def get(self, event_id: str) -> CanonicalMarketEvent | None: ...

    def __iter__(self) -> Iterator[CanonicalMarketEvent]: ...

    def __len__(self) -> int: ...


class InMemoryEventLedger:
    """Test/local ledger. A durable database adapter will implement the same contract."""

    def __init__(self) -> None:
        self._events: list[CanonicalMarketEvent] = []
        self._by_id: dict[str, CanonicalMarketEvent] = {}

    def append(self, event: CanonicalMarketEvent) -> None:
        if event.event_id in self._by_id:
            raise ValueError(f"Event already exists: {event.event_id}")
        self._events.append(event)
        self._by_id[event.event_id] = event

    def get(self, event_id: str) -> CanonicalMarketEvent | None:
        return self._by_id.get(event_id)

    def __iter__(self) -> Iterator[CanonicalMarketEvent]:
        return iter(tuple(self._events))

    def __len__(self) -> int:
        return len(self._events)
