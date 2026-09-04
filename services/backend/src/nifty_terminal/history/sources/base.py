"""Provider-neutral historical-data source interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nifty_terminal.history.models import HistoricalBatch, HistoricalRequest


class HistoricalDataSource(ABC):
    @abstractmethod
    def fetch(self, request: HistoricalRequest) -> HistoricalBatch:
        """Return provider-exported finalized bars with unmodified provenance."""
