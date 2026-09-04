"""Authoritative minute-bar finalization and deterministic aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.domain.candle import (
    Candle,
    CandleSource,
    CandleStatus,
    FinalizedMinuteBarInput,
    Timeframe,
)
from nifty_terminal.domain.instruments import InstrumentRegistry


@dataclass(frozen=True, slots=True)
class CandleEngineResult:
    minute_candle: Candle
    finalized_candles: tuple[Candle, ...]


class CandleEngine:
    """Builds 5m/15m/1h candles only from complete finalized 1m bars."""

    AGGREGATE_TIMEFRAMES = (Timeframe.M5, Timeframe.M15, Timeframe.H1)

    def __init__(
        self,
        *,
        calendar: NseSessionCalendar,
        registry: InstrumentRegistry,
        store: InMemoryCandleStore,
    ) -> None:
        self._calendar = calendar
        self._registry = registry
        self._store = store

    def ingest_finalized_minute(self, bar: FinalizedMinuteBarInput) -> CandleEngineResult:
        self._validate_minute(bar)
        previous = self._store.latest(bar.instrument_id, Timeframe.M1, bar.opens_at)
        if previous is not None and bar.provider_revision <= previous.source_revision:
            raise ValueError("Provider correction revision must increase monotonically")

        revision = 1 if previous is None else previous.revision + 1
        minute = Candle(
            schema_version=1,
            candle_id=_candle_id(
                bar.instrument_id,
                Timeframe.M1,
                bar.opens_at,
                revision,
                bar.provider_bar_id,
                bar.provider_revision,
            ),
            instrument_id=bar.instrument_id,
            timeframe=Timeframe.M1,
            opens_at=bar.opens_at.astimezone(timezone.utc),
            closes_at=bar.closes_at.astimezone(timezone.utc),
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            status=CandleStatus.FINALIZED,
            revision=revision,
            source=CandleSource.AUTHORITATIVE_MINUTE,
            provider=bar.provider.casefold(),
            source_revision=bar.provider_revision,
            finalized_at=bar.finalized_at.astimezone(timezone.utc),
            component_candle_ids=(bar.provider_bar_id,),
            source_watermark=bar.source_watermark,
            supersedes_candle_id=previous.candle_id if previous else None,
        )
        self._store.append(minute)

        finalized = tuple(
            candle
            for timeframe in self.AGGREGATE_TIMEFRAMES
            if (candle := self._rebuild_bucket(minute, timeframe)) is not None
        )
        return CandleEngineResult(minute_candle=minute, finalized_candles=finalized)

    def _validate_minute(self, bar: FinalizedMinuteBarInput) -> None:
        instrument = self._registry.get(bar.instrument_id)
        for field_name in ("opens_at", "closes_at", "finalized_at"):
            value = getattr(bar, field_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        if bar.closes_at - bar.opens_at != timedelta(minutes=1):
            raise ValueError("Authoritative minute input must span exactly one minute")
        bucket = self._calendar.bucket_for(bar.opens_at, Timeframe.M1)
        if bucket.opens_at != bar.opens_at.astimezone(timezone.utc):
            raise ValueError("Minute input is not aligned to the NSE session")
        if bar.closes_at.astimezone(timezone.utc) != bucket.closes_at:
            raise ValueError("Minute close does not match the NSE session bucket")
        if min(bar.open, bar.high, bar.low, bar.close) <= Decimal("0"):
            raise ValueError("OHLC values must be positive")
        if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            raise ValueError("OHLC values violate candle invariants")
        if bar.low > bar.high:
            raise ValueError("Candle low cannot exceed candle high")
        if not instrument.volume_supported and bar.volume is not None:
            raise ValueError("Volume must remain null for NIFTY 50 spot")
        if bar.provider_revision < 1:
            raise ValueError("Provider revision must be positive")

    def _rebuild_bucket(self, minute: Candle, timeframe: Timeframe) -> Candle | None:
        bucket = self._calendar.bucket_for(minute.opens_at, timeframe)
        if bucket.is_partial:
            return None

        components: list[Candle] = []
        for opens_at in self._calendar.expected_minute_opens(bucket):
            component = self._store.latest(minute.instrument_id, Timeframe.M1, opens_at)
            if component is None or component.status is not CandleStatus.FINALIZED:
                return None
            components.append(component)

        previous = self._store.latest(minute.instrument_id, timeframe, bucket.opens_at)
        component_ids = tuple(item.candle_id for item in components)
        if previous is not None and previous.component_candle_ids == component_ids:
            return None

        revision = 1 if previous is None else previous.revision + 1
        provider_names = sorted({item.provider for item in components})
        provider = provider_names[0] if len(provider_names) == 1 else "canonical-mixed"
        source_watermark = components[-1].source_watermark
        candle = Candle(
            schema_version=1,
            candle_id=_candle_id(
                minute.instrument_id,
                timeframe,
                bucket.opens_at,
                revision,
                source_watermark,
                max(item.source_revision for item in components),
            ),
            instrument_id=minute.instrument_id,
            timeframe=timeframe,
            opens_at=bucket.opens_at,
            closes_at=bucket.closes_at,
            open=components[0].open,
            high=max(item.high for item in components),
            low=min(item.low for item in components),
            close=components[-1].close,
            volume=_aggregate_volume(components),
            status=CandleStatus.FINALIZED,
            revision=revision,
            source=CandleSource.AGGREGATED,
            provider=provider,
            source_revision=max(item.source_revision for item in components),
            finalized_at=max(item.finalized_at for item in components if item.finalized_at),
            component_candle_ids=component_ids,
            source_watermark=source_watermark,
            supersedes_candle_id=previous.candle_id if previous else None,
        )
        self._store.append(candle)
        return candle


def _aggregate_volume(components: list[Candle]) -> Decimal | None:
    if any(item.volume is None for item in components):
        return None
    return sum((item.volume for item in components if item.volume is not None), Decimal("0"))


def _candle_id(
    instrument_id: str,
    timeframe: Timeframe,
    opens_at: object,
    revision: int,
    source_identity: str,
    source_revision: int,
) -> str:
    identity = (
        f"candle:{instrument_id}:{timeframe.value}:{opens_at}:"
        f"{revision}:{source_identity}:{source_revision}"
    )
    return str(uuid5(NAMESPACE_URL, identity))
