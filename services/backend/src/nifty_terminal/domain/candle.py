"""Immutable canonical candle records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"

    @property
    def minutes(self) -> int:
        return {self.M1: 1, self.M5: 5, self.M15: 15, self.H1: 60}[self]


class CandleStatus(StrEnum):
    DEVELOPING = "DEVELOPING"
    FINALIZED = "FINALIZED"


class CandleSource(StrEnum):
    PROVISIONAL_EVENTS = "PROVISIONAL_EVENTS"
    AUTHORITATIVE_MINUTE = "AUTHORITATIVE_MINUTE"
    AGGREGATED = "AGGREGATED"


@dataclass(frozen=True, slots=True)
class Candle:
    schema_version: int
    candle_id: str
    instrument_id: str
    timeframe: Timeframe
    opens_at: datetime
    closes_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    status: CandleStatus
    revision: int
    source: CandleSource
    provider: str
    source_revision: int
    finalized_at: datetime | None
    component_candle_ids: tuple[str, ...]
    source_watermark: str
    supersedes_candle_id: str | None = None

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candle_id": self.candle_id,
            "instrument_id": self.instrument_id,
            "timeframe": self.timeframe.value,
            "opens_at": _datetime_text(self.opens_at),
            "closes_at": _datetime_text(self.closes_at),
            "open": _decimal_text(self.open),
            "high": _decimal_text(self.high),
            "low": _decimal_text(self.low),
            "close": _decimal_text(self.close),
            "volume": _decimal_text(self.volume),
            "status": self.status.value,
            "revision": self.revision,
            "source": self.source.value,
            "provider": self.provider,
            "source_revision": self.source_revision,
            "finalized_at": _datetime_text(self.finalized_at),
            "component_candle_ids": list(self.component_candle_ids),
            "source_watermark": self.source_watermark,
            "supersedes_candle_id": self.supersedes_candle_id,
        }


@dataclass(frozen=True, slots=True)
class FinalizedMinuteBarInput:
    provider_bar_id: str
    provider: str
    instrument_id: str
    opens_at: datetime
    closes_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    provider_revision: int
    finalized_at: datetime
    source_watermark: str


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _datetime_text(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None
