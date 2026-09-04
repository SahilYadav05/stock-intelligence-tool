"""Append-only candle storage boundary with immutable revisions."""

from __future__ import annotations

from datetime import datetime

from nifty_terminal.domain.candle import Candle, Timeframe


CandleKey = tuple[str, Timeframe, datetime]


class InMemoryCandleStore:
    """Test/MVP store; a persistent adapter can replace it behind this boundary."""

    def __init__(self) -> None:
        self._revisions: dict[CandleKey, list[Candle]] = {}
        self._ids: set[str] = set()

    def append(self, candle: Candle) -> None:
        if candle.candle_id in self._ids:
            raise ValueError(f"Candle already stored: {candle.candle_id}")
        key = (candle.instrument_id, candle.timeframe, candle.opens_at)
        revisions = self._revisions.setdefault(key, [])
        expected_revision = len(revisions) + 1
        if candle.revision != expected_revision:
            raise ValueError(
                f"Expected candle revision {expected_revision}, received {candle.revision}"
            )
        if revisions and candle.supersedes_candle_id != revisions[-1].candle_id:
            raise ValueError("A correction must reference the immediately prior candle revision")
        if not revisions and candle.supersedes_candle_id is not None:
            raise ValueError("An initial candle cannot supersede another candle")
        revisions.append(candle)
        self._ids.add(candle.candle_id)

    def latest(
        self,
        instrument_id: str,
        timeframe: Timeframe,
        opens_at: datetime,
    ) -> Candle | None:
        revisions = self._revisions.get((instrument_id, timeframe, opens_at), [])
        return revisions[-1] if revisions else None

    def revisions(
        self,
        instrument_id: str,
        timeframe: Timeframe,
        opens_at: datetime,
    ) -> tuple[Candle, ...]:
        return tuple(self._revisions.get((instrument_id, timeframe, opens_at), ()))

    def latest_series(
        self,
        instrument_id: str,
        timeframe: Timeframe,
        *,
        closes_at_or_before: datetime | None = None,
    ) -> tuple[Candle, ...]:
        candles = [
            versions[-1]
            for (stored_instrument, stored_timeframe, _), versions in self._revisions.items()
            if stored_instrument == instrument_id and stored_timeframe is timeframe
        ]
        if closes_at_or_before is not None:
            candles = [item for item in candles if item.closes_at <= closes_at_or_before]
        return tuple(sorted(candles, key=lambda item: item.opens_at))
