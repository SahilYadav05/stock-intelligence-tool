"""Replaceable latest-view read model."""

from __future__ import annotations

from threading import RLock

from nifty_terminal.delivery.models import MarketStateView


class InMemoryMarketStateReadModel:
    """Process-local Step 4 store; production persistence arrives in later steps."""

    def __init__(self) -> None:
        self._latest: dict[str, MarketStateView] = {}
        self._lock = RLock()

    def put(self, view: MarketStateView) -> None:
        with self._lock:
            current = self._latest.get(view.snapshot.instrument_id)
            if current is not None and view.published_at < current.published_at:
                raise ValueError("An older market-state view cannot replace a newer view")
            self._latest[view.snapshot.instrument_id] = view

    def get(self, instrument_id: str) -> MarketStateView | None:
        with self._lock:
            return self._latest.get(instrument_id)
