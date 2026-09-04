"""Deterministic offline provider used for development and fault-free tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

from nifty_terminal.domain.enums import ConnectionState, MarketEventType, TimestampSource
from nifty_terminal.domain.market_event import RawMarketEvent
from nifty_terminal.providers.base import ProviderAdapter, ProviderHealth


class ReplayProviderAdapter(ProviderAdapter):
    """Replays explicitly labelled test fixtures; it never represents live data."""

    def __init__(self, events: Iterable[RawMarketEvent]) -> None:
        self._events = tuple(events)
        self._state = ConnectionState.DISCONNECTED
        self._last_event_time: datetime | None = None

    @property
    def provider_name(self) -> str:
        return "replay"

    @property
    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider_name,
            connection_state=self._state,
            observed_at=datetime.now(timezone.utc),
            last_event_time=self._last_event_time,
            detail="Synthetic test replay; not a live market connection.",
        )

    async def connect(self) -> None:
        self._state = ConnectionState.RECOVERING
        await asyncio.sleep(0)
        self._state = ConnectionState.LIVE

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED
        await asyncio.sleep(0)

    async def stream(self) -> AsyncIterator[RawMarketEvent]:
        if self._state is not ConnectionState.LIVE:
            raise RuntimeError("Replay adapter must be connected before streaming")

        for event in self._events:
            self._last_event_time = event.provider_event_time
            await asyncio.sleep(0)
            yield event

    @classmethod
    def from_jsonl(cls, path: Path) -> "ReplayProviderAdapter":
        events: list[RawMarketEvent] = []
        with path.open("r", encoding="utf-8") as fixture:
            for line_number, line in enumerate(fixture, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    events.append(_raw_event_from_fixture(payload))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid replay fixture at {path}:{line_number}: {error}"
                    ) from error
        return cls(events)


def _raw_event_from_fixture(payload: dict[str, object]) -> RawMarketEvent:
    return RawMarketEvent(
        provider=str(payload["provider"]),
        provider_instrument_id=str(payload["provider_instrument_id"]),
        event_type=MarketEventType(str(payload["event_type"])),
        server_arrival_time=_parse_utc(str(payload["server_arrival_time"])),
        connection_epoch=str(payload["connection_epoch"]),
        raw_payload=payload.get("raw_payload", {}),
        provider_event_time=_optional_datetime(payload.get("provider_event_time")),
        provider_send_time=_optional_datetime(payload.get("provider_send_time")),
        timestamp_source=TimestampSource(str(payload.get("timestamp_source", "PROVIDER"))),
        provider_sequence=_optional_int(payload.get("provider_sequence")),
        provider_sequence_scope=_optional_string(payload.get("provider_sequence_scope")),
        provider_sequence_is_contiguous=bool(
            payload.get("provider_sequence_is_contiguous", False)
        ),
        price=_optional_decimal(payload.get("price")),
        last_quantity=_optional_decimal(payload.get("last_quantity")),
        cumulative_volume=_optional_decimal(payload.get("cumulative_volume")),
        bid_price=_optional_decimal(payload.get("bid_price")),
        ask_price=_optional_decimal(payload.get("ask_price")),
        supersedes_event_id=_optional_string(payload.get("supersedes_event_id")),
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Replay timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _optional_datetime(value: object) -> datetime | None:
    return _parse_utc(str(value)) if value is not None else None


def _optional_decimal(value: object) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
