"""Historical source adapters."""

from nifty_terminal.history.sources.base import HistoricalDataSource
from nifty_terminal.history.sources.angelone_source import (
    AcquiredHistoricalDataSource,
    AngelOneHistoricalAcquirer,
)
from nifty_terminal.history.sources.csv_source import CsvHistoricalDataSource

__all__ = [
    "AcquiredHistoricalDataSource",
    "AngelOneHistoricalAcquirer",
    "CsvHistoricalDataSource",
    "HistoricalDataSource",
]
