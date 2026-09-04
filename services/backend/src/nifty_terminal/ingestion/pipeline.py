"""End-to-end provider-neutral ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nifty_terminal.domain.market_event import CanonicalMarketEvent, RawMarketEvent
from nifty_terminal.ingestion.deduplicator import EventDeduplicator
from nifty_terminal.ingestion.ledger import EventLedger
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer, NormalizationError
from nifty_terminal.ingestion.sequence import SequenceTracker
from nifty_terminal.ingestion.validator import MarketEventValidator, ValidationIssue
from nifty_terminal.providers.base import ProviderAdapter


class IngestionStatus(StrEnum):
    STORED = "STORED"
    DUPLICATE = "DUPLICATE"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    status: IngestionStatus
    event: CanonicalMarketEvent | None
    issues: tuple[ValidationIssue, ...] = ()
    detail: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        *,
        normalizer: MarketEventNormalizer,
        validator: MarketEventValidator,
        ledger: EventLedger,
        deduplicator: EventDeduplicator | None = None,
        sequence_tracker: SequenceTracker | None = None,
    ) -> None:
        self._normalizer = normalizer
        self._validator = validator
        self._ledger = ledger
        self._deduplicator = deduplicator or EventDeduplicator()
        self._sequence_tracker = sequence_tracker or SequenceTracker()

    def process(self, raw: RawMarketEvent) -> IngestionOutcome:
        try:
            event = self._normalizer.normalize(raw)
        except NormalizationError as error:
            return IngestionOutcome(
                status=IngestionStatus.QUARANTINED,
                event=None,
                detail=str(error),
            )

        decision = self._validator.validate(event)
        if not decision.accepted:
            return IngestionOutcome(
                status=IngestionStatus.QUARANTINED,
                event=event,
                issues=decision.issues,
                detail="Canonical validation failed.",
            )

        if self._deduplicator.is_duplicate(event):
            return IngestionOutcome(
                status=IngestionStatus.DUPLICATE,
                event=event,
                issues=decision.issues,
                detail="Duplicate event was not appended.",
            )

        sequence_issue = self._sequence_tracker.validate(event)
        if sequence_issue is not None:
            return IngestionOutcome(
                status=IngestionStatus.QUARANTINED,
                event=event,
                issues=(*decision.issues, sequence_issue),
                detail="Provider sequence continuity failed.",
            )

        self._ledger.append(event)
        self._deduplicator.mark(event)
        self._sequence_tracker.mark(event)
        return IngestionOutcome(
            status=IngestionStatus.STORED,
            event=event,
            issues=decision.issues,
        )

    async def consume(self, adapter: ProviderAdapter) -> tuple[IngestionOutcome, ...]:
        outcomes: list[IngestionOutcome] = []
        await adapter.connect()
        try:
            async for raw in adapter.stream():
                outcomes.append(self.process(raw))
        finally:
            await adapter.disconnect()
        return tuple(outcomes)
