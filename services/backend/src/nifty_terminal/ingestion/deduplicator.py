"""Deterministic event deduplication."""

from nifty_terminal.domain.market_event import CanonicalMarketEvent


class EventDeduplicator:
    def __init__(self) -> None:
        self._event_ids: set[str] = set()

    def is_duplicate(self, event: CanonicalMarketEvent) -> bool:
        return event.event_id in self._event_ids

    def mark(self, event: CanonicalMarketEvent) -> None:
        self._event_ids.add(event.event_id)

    def __len__(self) -> int:
        return len(self._event_ids)
