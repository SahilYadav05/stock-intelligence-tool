"""Normalization from provider observations into canonical market events."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.domain.enums import EventQualityCode, TimestampSource
from nifty_terminal.domain.instruments import InstrumentRegistry
from nifty_terminal.domain.market_event import CanonicalMarketEvent, RawMarketEvent


class NormalizationError(ValueError):
    """Raised when an event cannot be represented safely in canonical form."""


class MarketEventNormalizer:
    def __init__(self, registry: InstrumentRegistry) -> None:
        self._registry = registry

    def normalize(self, raw: RawMarketEvent) -> CanonicalMarketEvent:
        try:
            instrument = self._registry.resolve(
                raw.provider,
                raw.provider_instrument_id,
            )
        except KeyError as error:
            raise NormalizationError(str(error)) from error

        arrival_time = _as_utc(raw.server_arrival_time, "server_arrival_time")
        provider_event_time = _optional_utc(raw.provider_event_time, "provider_event_time")
        provider_send_time = _optional_utc(raw.provider_send_time, "provider_send_time")

        quality_codes: list[EventQualityCode] = []
        if provider_event_time is None:
            normalized_event_time = arrival_time
            timestamp_source = TimestampSource.ARRIVAL
            quality_codes.append(EventQualityCode.ARRIVAL_TIME_FALLBACK)
        else:
            normalized_event_time = provider_event_time
            timestamp_source = raw.timestamp_source
            if timestamp_source is TimestampSource.PROVIDER:
                quality_codes.append(EventQualityCode.PROVIDER_TIMESTAMP)

        if raw.provider_sequence is None:
            quality_codes.append(EventQualityCode.NO_PROVIDER_SEQUENCE)

        raw_payload_hash = _payload_hash(raw.raw_payload)
        deduplication_key = _deduplication_key(
            raw=raw,
            instrument_id=instrument.instrument_id,
            normalized_event_time=normalized_event_time,
            raw_payload_hash=raw_payload_hash,
        )
        event_id = str(uuid5(NAMESPACE_URL, deduplication_key))

        return CanonicalMarketEvent(
            schema_version=1,
            event_id=event_id,
            instrument_id=instrument.instrument_id,
            event_type=raw.event_type,
            provider=raw.provider.casefold(),
            provider_instrument_id=raw.provider_instrument_id,
            connection_epoch=raw.connection_epoch,
            provider_sequence=raw.provider_sequence,
            provider_sequence_scope=raw.provider_sequence_scope,
            provider_sequence_is_contiguous=raw.provider_sequence_is_contiguous,
            provider_event_time=provider_event_time,
            provider_send_time=provider_send_time,
            server_arrival_time=arrival_time,
            normalized_event_time=normalized_event_time,
            timestamp_source=timestamp_source,
            price=raw.price,
            last_quantity=raw.last_quantity,
            cumulative_volume=raw.cumulative_volume,
            bid_price=raw.bid_price,
            ask_price=raw.ask_price,
            currency=instrument.currency,
            raw_payload_hash=raw_payload_hash,
            deduplication_key=deduplication_key,
            quality_codes=tuple(sorted(set(quality_codes), key=lambda item: item.value)),
            supersedes_event_id=raw.supersedes_event_id,
        )


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise NormalizationError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_utc(value: datetime | None, field_name: str) -> datetime | None:
    return _as_utc(value, field_name) if value is not None else None


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deduplication_key(
    *,
    raw: RawMarketEvent,
    instrument_id: str,
    normalized_event_time: datetime,
    raw_payload_hash: str,
) -> str:
    if raw.provider_sequence is not None:
        identity = {
            "provider": raw.provider.casefold(),
            "instrument_id": instrument_id,
            "provider_sequence_scope": raw.provider_sequence_scope,
            "provider_sequence": raw.provider_sequence,
            "event_type": raw.event_type.value,
        }
    else:
        identity = {
            "provider": raw.provider.casefold(),
            "instrument_id": instrument_id,
            "event_type": raw.event_type.value,
            "event_time": normalized_event_time.isoformat(),
            "price": _decimal_text(raw.price),
            "last_quantity": _decimal_text(raw.last_quantity),
            "cumulative_volume": _decimal_text(raw.cumulative_volume),
            "bid_price": _decimal_text(raw.bid_price),
            "ask_price": _decimal_text(raw.ask_price),
            "raw_payload_hash": raw_payload_hash,
        }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported raw payload value: {type(value).__name__}")
