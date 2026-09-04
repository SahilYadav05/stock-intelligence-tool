"""Immutable raw and canonical market-event records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

from nifty_terminal.domain.enums import EventQualityCode, MarketEventType, TimestampSource


@dataclass(frozen=True, slots=True)
class RawMarketEvent:
    """Provider-shaped observation before canonical normalization."""

    provider: str
    provider_instrument_id: str
    event_type: MarketEventType
    server_arrival_time: datetime
    connection_epoch: str
    raw_payload: Mapping[str, Any] = field(default_factory=dict)
    provider_event_time: datetime | None = None
    provider_send_time: datetime | None = None
    timestamp_source: TimestampSource = TimestampSource.PROVIDER
    provider_sequence: int | None = None
    provider_sequence_scope: str | None = None
    provider_sequence_is_contiguous: bool = False
    price: Decimal | None = None
    last_quantity: Decimal | None = None
    cumulative_volume: Decimal | None = None
    bid_price: Decimal | None = None
    ask_price: Decimal | None = None
    supersedes_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class CanonicalMarketEvent:
    """Provider-neutral, immutable observation accepted by the ingestion boundary."""

    schema_version: int
    event_id: str
    instrument_id: str
    event_type: MarketEventType
    provider: str
    provider_instrument_id: str
    connection_epoch: str
    provider_sequence: int | None
    provider_sequence_scope: str | None
    provider_sequence_is_contiguous: bool
    provider_event_time: datetime | None
    provider_send_time: datetime | None
    server_arrival_time: datetime
    normalized_event_time: datetime
    timestamp_source: TimestampSource
    price: Decimal | None
    last_quantity: Decimal | None
    cumulative_volume: Decimal | None
    bid_price: Decimal | None
    ask_price: Decimal | None
    currency: str
    raw_payload_hash: str
    deduplication_key: str
    quality_codes: tuple[EventQualityCode, ...]
    supersedes_event_id: str | None = None

    def to_contract(self) -> dict[str, object]:
        """Serialize without binary floats or timezone ambiguity."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "instrument_id": self.instrument_id,
            "event_type": self.event_type.value,
            "provider": self.provider,
            "provider_instrument_id": self.provider_instrument_id,
            "connection_epoch": self.connection_epoch,
            "provider_sequence": self.provider_sequence,
            "provider_sequence_scope": self.provider_sequence_scope,
            "provider_sequence_is_contiguous": self.provider_sequence_is_contiguous,
            "provider_event_time": _datetime_text(self.provider_event_time),
            "provider_send_time": _datetime_text(self.provider_send_time),
            "server_arrival_time": _datetime_text(self.server_arrival_time),
            "normalized_event_time": _datetime_text(self.normalized_event_time),
            "timestamp_source": self.timestamp_source.value,
            "price": _decimal_text(self.price),
            "last_quantity": _decimal_text(self.last_quantity),
            "cumulative_volume": _decimal_text(self.cumulative_volume),
            "bid_price": _decimal_text(self.bid_price),
            "ask_price": _decimal_text(self.ask_price),
            "currency": self.currency,
            "raw_payload_hash": self.raw_payload_hash,
            "deduplication_key": self.deduplication_key,
            "quality_codes": [item.value for item in self.quality_codes],
            "supersedes_event_id": self.supersedes_event_id,
        }


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _datetime_text(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None
