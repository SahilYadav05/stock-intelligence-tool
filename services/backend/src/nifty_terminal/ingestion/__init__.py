"""Canonical event normalization, validation, deduplication, and storage."""

from nifty_terminal.ingestion.ledger import InMemoryEventLedger
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer, NormalizationError
from nifty_terminal.ingestion.pipeline import IngestionOutcome, IngestionPipeline, IngestionStatus
from nifty_terminal.ingestion.sequence import SequenceTracker
from nifty_terminal.ingestion.validator import MarketEventValidator, ValidationDecision

__all__ = [
    "InMemoryEventLedger",
    "IngestionOutcome",
    "IngestionPipeline",
    "IngestionStatus",
    "MarketEventNormalizer",
    "MarketEventValidator",
    "NormalizationError",
    "SequenceTracker",
    "ValidationDecision",
]
