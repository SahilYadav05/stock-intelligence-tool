"""Provisional event-driven candles for chart display only."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe
from nifty_terminal.domain.enums import MarketEventType
from nifty_terminal.domain.market_event import CanonicalMarketEvent


class DevelopingCandleEngine:
    """Updates a visual candle but has no path that can finalize it."""

    def __init__(self, calendar: NseSessionCalendar) -> None:
        self._calendar = calendar
        self._current: dict[tuple[str, Timeframe], Candle] = {}

    def apply(self, event: CanonicalMarketEvent, timeframe: Timeframe) -> Candle | None:
        if event.event_type not in {MarketEventType.INDEX_VALUE, MarketEventType.TRADE}:
            return None
        if event.price is None:
            return None

        bucket = self._calendar.bucket_for(event.normalized_event_time, timeframe)
        key = (event.instrument_id, timeframe)
        existing = self._current.get(key)
        price = Decimal(event.price)

        if existing is not None and event.normalized_event_time < existing.opens_at:
            raise ValueError("Out-of-order event cannot mutate a later developing candle")

        if existing is None or existing.opens_at != bucket.opens_at:
            component_ids = (event.event_id,)
            revision = 1
            candle = Candle(
                schema_version=1,
                candle_id=_developing_id(event.instrument_id, timeframe, bucket.opens_at, revision),
                instrument_id=event.instrument_id,
                timeframe=timeframe,
                opens_at=bucket.opens_at,
                closes_at=bucket.closes_at,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=None,
                status=CandleStatus.DEVELOPING,
                revision=revision,
                source=CandleSource.PROVISIONAL_EVENTS,
                provider=event.provider,
                source_revision=revision,
                finalized_at=None,
                component_candle_ids=component_ids,
                source_watermark=event.event_id,
            )
        else:
            revision = existing.revision + 1
            candle = replace(
                existing,
                candle_id=_developing_id(
                    event.instrument_id, timeframe, bucket.opens_at, revision
                ),
                high=max(existing.high, price),
                low=min(existing.low, price),
                close=price,
                revision=revision,
                source_revision=revision,
                component_candle_ids=existing.component_candle_ids + (event.event_id,),
                source_watermark=event.event_id,
                supersedes_candle_id=existing.candle_id,
            )
        self._current[key] = candle
        return candle

    def current(self, instrument_id: str, timeframe: Timeframe) -> Candle | None:
        return self._current.get((instrument_id, timeframe))


def _developing_id(
    instrument_id: str,
    timeframe: Timeframe,
    opens_at: object,
    revision: int,
) -> str:
    return str(uuid5(NAMESPACE_URL, f"developing:{instrument_id}:{timeframe}:{opens_at}:{revision}"))
