"""Replaceable persistence interface for historical research datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nifty_terminal.domain.candle import Candle, Timeframe
from nifty_terminal.features.models import PriceFeatureRow
from nifty_terminal.history.models import HistoricalBatch, HistoricalQualityReport, QualityStatus


class HistoricalRepository(ABC):
    @abstractmethod
    def load_dataset_quality_status(self, *, dataset_id: str) -> QualityStatus | None:
        """Return the immutable dataset verdict, or None when it does not exist."""

    @abstractmethod
    def save_dataset(
        self,
        *,
        dataset_id: str,
        batch: HistoricalBatch,
        quality: HistoricalQualityReport,
        candles: tuple[Candle, ...],
    ) -> bool:
        """Persist a new immutable dataset, returning False when already present."""

    @abstractmethod
    def load_latest_candles(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        timeframe: Timeframe,
    ) -> tuple[Candle, ...]:
        """Load the latest revision of every bucket in chronological order."""

    @abstractmethod
    def save_feature_rows(
        self,
        *,
        dataset_id: str,
        rows: tuple[PriceFeatureRow, ...],
    ) -> int:
        """Persist immutable feature rows and return the inserted count."""
