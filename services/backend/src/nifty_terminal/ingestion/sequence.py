"""Sequence continuity checks for providers that document contiguous numbering."""

from nifty_terminal.domain.enums import ValidationSeverity
from nifty_terminal.domain.market_event import CanonicalMarketEvent
from nifty_terminal.ingestion.validator import ValidationIssue


class SequenceTracker:
    def __init__(self) -> None:
        self._last_by_scope: dict[tuple[str, str, str], int] = {}

    def validate(self, event: CanonicalMarketEvent) -> ValidationIssue | None:
        if not event.provider_sequence_is_contiguous:
            return None
        if event.provider_sequence is None or event.provider_sequence_scope is None:
            return ValidationIssue(
                code="SEQUENCE_METADATA_INCOMPLETE",
                severity=ValidationSeverity.ERROR,
                detail="Contiguous sequence metadata is incomplete.",
            )

        key = (event.provider, event.instrument_id, event.provider_sequence_scope)
        previous = self._last_by_scope.get(key)
        if previous is None:
            return None
        if event.provider_sequence < previous:
            return ValidationIssue(
                code="SEQUENCE_OUT_OF_ORDER",
                severity=ValidationSeverity.ERROR,
                detail=(
                    f"Provider sequence moved backwards from {previous} "
                    f"to {event.provider_sequence}."
                ),
            )
        if event.provider_sequence > previous + 1:
            return ValidationIssue(
                code="SEQUENCE_GAP",
                severity=ValidationSeverity.ERROR,
                detail=(
                    f"Expected provider sequence {previous + 1}, "
                    f"received {event.provider_sequence}."
                ),
            )
        return None

    def mark(self, event: CanonicalMarketEvent) -> None:
        if (
            event.provider_sequence_is_contiguous
            and event.provider_sequence is not None
            and event.provider_sequence_scope is not None
        ):
            key = (event.provider, event.instrument_id, event.provider_sequence_scope)
            self._last_by_scope[key] = event.provider_sequence
