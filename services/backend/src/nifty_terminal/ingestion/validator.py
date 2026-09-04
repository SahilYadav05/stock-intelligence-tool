"""Fail-safe canonical market-event validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from nifty_terminal.domain.enums import (
    EventQualityCode,
    MarketEventType,
    ValidationSeverity,
)
from nifty_terminal.domain.instruments import InstrumentRegistry
from nifty_terminal.domain.market_event import CanonicalMarketEvent


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: ValidationSeverity
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    accepted: bool
    event: CanonicalMarketEvent
    issues: tuple[ValidationIssue, ...]


class MarketEventValidator:
    def __init__(
        self,
        registry: InstrumentRegistry,
        *,
        future_tolerance: timedelta = timedelta(seconds=2),
        delayed_after: timedelta = timedelta(seconds=3),
    ) -> None:
        self._registry = registry
        self._future_tolerance = future_tolerance
        self._delayed_after = delayed_after

    def validate(self, event: CanonicalMarketEvent) -> ValidationDecision:
        issues: list[ValidationIssue] = []
        instrument = self._registry.get(event.instrument_id)

        if event.provider_sequence is not None and event.provider_sequence < 0:
            issues.append(_error("NEGATIVE_SEQUENCE", "Provider sequence cannot be negative."))

        if event.provider_sequence is not None and not event.provider_sequence_scope:
            issues.append(
                _error(
                    "SEQUENCE_SCOPE_REQUIRED",
                    "A provider sequence requires an explicit documented scope.",
                )
            )
        if event.provider_sequence is None and event.provider_sequence_scope is not None:
            issues.append(
                _error(
                    "UNEXPECTED_SEQUENCE_SCOPE",
                    "Sequence scope cannot be set when no sequence is present.",
                )
            )
        if event.provider_sequence_is_contiguous and event.provider_sequence is None:
            issues.append(
                _error(
                    "CONTIGUOUS_SEQUENCE_REQUIRED",
                    "Contiguous sequence semantics require a provider sequence.",
                )
            )

        if event.normalized_event_time > event.server_arrival_time + self._future_tolerance:
            issues.append(
                _error(
                    "EVENT_FROM_FUTURE",
                    "Event time exceeds the permitted clock-skew tolerance.",
                )
            )

        arrival_delay = event.server_arrival_time - event.normalized_event_time
        if arrival_delay > self._delayed_after:
            issues.append(
                _warning(
                    EventQualityCode.LATE_ARRIVAL.value,
                    f"Event arrived {arrival_delay.total_seconds():.3f}s after event time.",
                )
            )

        if event.event_type is not MarketEventType.HEARTBEAT:
            if event.price is None:
                issues.append(_error("PRICE_REQUIRED", "Market observation requires a price."))
            elif event.price <= Decimal("0"):
                issues.append(_error("PRICE_NOT_POSITIVE", "Price must be positive."))

        for field_name, value in (
            ("last_quantity", event.last_quantity),
            ("cumulative_volume", event.cumulative_volume),
        ):
            if value is not None and value < Decimal("0"):
                issues.append(
                    _error("NEGATIVE_QUANTITY", f"{field_name} cannot be negative.")
                )

        if not instrument.volume_supported and (
            event.last_quantity is not None or event.cumulative_volume is not None
        ):
            issues.append(
                _error(
                    "VOLUME_NOT_SUPPORTED",
                    f"{instrument.display_name} does not have legitimate spot-index volume.",
                )
            )

        if event.bid_price is not None and event.bid_price <= Decimal("0"):
            issues.append(_error("BID_NOT_POSITIVE", "Bid price must be positive."))
        if event.ask_price is not None and event.ask_price <= Decimal("0"):
            issues.append(_error("ASK_NOT_POSITIVE", "Ask price must be positive."))
        if (
            event.bid_price is not None
            and event.ask_price is not None
            and event.bid_price > event.ask_price
        ):
            issues.append(_error("CROSSED_QUOTE", "Bid price cannot exceed ask price."))

        if event.event_type is MarketEventType.CORRECTION:
            if event.supersedes_event_id is None:
                issues.append(
                    _error(
                        "CORRECTION_TARGET_REQUIRED",
                        "Correction event must identify the superseded event.",
                    )
                )
        elif event.supersedes_event_id is not None:
            issues.append(
                _error(
                    "UNEXPECTED_CORRECTION_TARGET",
                    "Only correction events may identify a superseded event.",
                )
            )

        accepted = not any(item.severity is ValidationSeverity.ERROR for item in issues)
        return ValidationDecision(accepted=accepted, event=event, issues=tuple(issues))


def _error(code: str, detail: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=ValidationSeverity.ERROR, detail=detail)


def _warning(code: str, detail: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=ValidationSeverity.WARNING, detail=detail)
