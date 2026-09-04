"""Immutable, chart-synchronized analysis view for the professional dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from nifty_terminal.signals.models import SignalDecision


class ContextStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class NewsStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NO_MATERIAL_EVENT = "NO_MATERIAL_EVENT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class NewsContextItem:
    event_id: str
    headline: str
    source: str
    published_at: datetime
    received_at: datetime
    impact: str

    def to_contract(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "headline": self.headline,
            "source": self.source,
            "published_at": _time(self.published_at),
            "received_at": _time(self.received_at),
            "impact": self.impact,
        }


@dataclass(frozen=True, slots=True)
class HistoricalSignalMarker:
    signal_id: str
    occurred_at: datetime
    direction: str
    price: Decimal
    status: str

    def to_contract(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "occurred_at": _time(self.occurred_at),
            "direction": self.direction,
            "price": _decimal(self.price),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class AnalysisView:
    schema_version: int
    analysis_id: str
    snapshot_id: str
    candle_revision_checksum: str
    instrument_id: str
    decision_time: datetime
    generated_at: datetime
    data_as_of: datetime
    signal: SignalDecision
    model_version: str
    calibration_version: str
    feature_version: str
    market_context_status: ContextStatus
    regime: str | None
    trend: str | None
    momentum: str | None
    volatility: str | None
    support_levels: tuple[Decimal, ...]
    resistance_levels: tuple[Decimal, ...]
    reasons: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    news_status: NewsStatus
    news_items: tuple[NewsContextItem, ...]
    historical_analog_count: int | None
    historical_analog_summary: str | None
    historical_signals: tuple[HistoricalSignalMarker, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("AnalysisView schema_version must be 1")
        for value in (self.decision_time, self.generated_at, self.data_as_of):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Analysis timestamps must be timezone-aware")
        if self.signal.snapshot_id != self.snapshot_id:
            raise ValueError("Signal and analysis must reference the same snapshot")
        if self.signal.input_revision_checksum != self.candle_revision_checksum:
            raise ValueError("Signal and analysis candle revisions must match")
        if self.signal.instrument_id != self.instrument_id:
            raise ValueError("Signal and analysis instruments must match")
        if self.signal.decision_time != self.decision_time:
            raise ValueError("Signal and analysis decision times must match")
        if self.data_as_of > self.generated_at:
            raise ValueError("Analysis cannot use data arriving after generation")
        if self.market_context_status is ContextStatus.UNAVAILABLE and any(
            value is not None
            for value in (self.regime, self.trend, self.momentum, self.volatility)
        ):
            raise ValueError("Unavailable market context cannot contain classifications")
        if self.news_status is NewsStatus.UNAVAILABLE and self.news_items:
            raise ValueError("Unavailable news context cannot contain news items")
        if self.historical_analog_count is not None and self.historical_analog_count < 0:
            raise ValueError("Historical analog count cannot be negative")

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "snapshot_id": self.snapshot_id,
            "candle_revision_checksum": self.candle_revision_checksum,
            "instrument_id": self.instrument_id,
            "decision_time": _time(self.decision_time),
            "generated_at": _time(self.generated_at),
            "data_as_of": _time(self.data_as_of),
            "signal": self.signal.to_contract(),
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "feature_version": self.feature_version,
            "market_context": {
                "status": self.market_context_status.value,
                "regime": self.regime,
                "trend": self.trend,
                "momentum": self.momentum,
                "volatility": self.volatility,
                "support_levels": [_decimal(item) for item in self.support_levels],
                "resistance_levels": [_decimal(item) for item in self.resistance_levels],
            },
            "reasons": list(self.reasons),
            "contradictory_evidence": list(self.contradictory_evidence),
            "news": {
                "status": self.news_status.value,
                "items": [item.to_contract() for item in self.news_items],
            },
            "historical_analogs": {
                "count": self.historical_analog_count,
                "summary": self.historical_analog_summary,
            },
            "historical_signals": [item.to_contract() for item in self.historical_signals],
        }


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _decimal(value: Decimal) -> str:
    return format(value, "f")
