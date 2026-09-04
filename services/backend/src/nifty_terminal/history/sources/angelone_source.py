"""Rate-controlled long-range Angel One historical acquisition."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import hashlib
import json

from nifty_terminal.domain.candle import FinalizedMinuteBarInput
from nifty_terminal.calendar.nse import IST
from nifty_terminal.history.models import HistoricalBatch, HistoricalRequest
from nifty_terminal.history.sources.base import HistoricalDataSource
from nifty_terminal.providers.base import FinalizedMinuteProvider


ProgressCallback = Callable[[int, int, int], None]
Clock = Callable[[], datetime]


class AcquiredHistoricalDataSource(HistoricalDataSource):
    """Expose one already-acquired immutable batch to the existing sync pipeline."""

    def __init__(self, batch: HistoricalBatch) -> None:
        self._batch = batch

    def fetch(self, request: HistoricalRequest) -> HistoricalBatch:
        if request != self._batch.request:
            raise ValueError("Acquired batch request does not match import request")
        return self._batch


class AngelOneHistoricalAcquirer:
    """Download finalized 1m bars in bounded requests without guessing corrections."""

    def __init__(
        self,
        *,
        provider: FinalizedMinuteProvider,
        provider_name: str = "angelone",
        chunk_days: int = 7,
        request_delay_milliseconds: int = 400,
        clock: Clock | None = None,
    ) -> None:
        if not 1 <= chunk_days <= 30:
            raise ValueError("chunk_days must be between 1 and 30")
        if not 0 <= request_delay_milliseconds <= 10_000:
            raise ValueError("request_delay_milliseconds must be between 0 and 10000")
        if not provider_name.strip():
            raise ValueError("provider_name is required")
        self._provider = provider
        self._provider_name = provider_name.casefold().strip()
        self._chunk_days = chunk_days
        self._request_delay = request_delay_milliseconds / 1_000
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def acquire(
        self,
        request: HistoricalRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> HistoricalBatch:
        boundaries = _chunk_boundaries(
            request.starts_at,
            request.ends_at,
            self._chunk_days,
        )
        rows_by_open: dict[datetime, FinalizedMinuteBarInput] = {}
        for index, (start, end) in enumerate(boundaries, start=1):
            rows = await self._provider.fetch_finalized_minutes(
                from_time=start,
                to_time=end,
            )
            for row in rows:
                _validate_row(
                    row,
                    request,
                    self._provider_name,
                    chunk_start=start,
                    chunk_end=end,
                )
                key = row.opens_at.astimezone(timezone.utc)
                existing = rows_by_open.get(key)
                if existing is not None and existing.provider_bar_id != row.provider_bar_id:
                    raise ValueError(
                        "Angel One returned conflicting finalized rows for one minute; "
                        "acquisition stopped instead of choosing silently"
                    )
                rows_by_open[key] = row
            if progress is not None:
                progress(index, len(boundaries), len(rows_by_open))
            if index < len(boundaries) and self._request_delay:
                await asyncio.sleep(self._request_delay)

        ordered = tuple(rows_by_open[key] for key in sorted(rows_by_open))
        acquired_at = _aware_utc(self._clock(), "clock")
        local_start = request.starts_at.astimezone(IST).date()
        local_end = request.ends_at.astimezone(IST).date() - timedelta(days=1)
        return HistoricalBatch(
            provider=self._provider_name,
            source_label=(
                f"angelone-api:{local_start.isoformat()}:"
                f"{local_end.isoformat()}:1m"
            ),
            source_sha256=historical_rows_sha256(ordered),
            acquired_at=acquired_at,
            request=request,
            rows=ordered,
        )


def _chunk_boundaries(
    starts_at: datetime,
    ends_at: datetime,
    chunk_days: int,
) -> tuple[tuple[datetime, datetime], ...]:
    start = _aware_utc(starts_at, "starts_at")
    end = _aware_utc(ends_at, "ends_at")
    if start >= end:
        raise ValueError("starts_at must precede ends_at")
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return tuple(chunks)


def _validate_row(
    row: FinalizedMinuteBarInput,
    request: HistoricalRequest,
    provider_name: str,
    *,
    chunk_start: datetime,
    chunk_end: datetime,
) -> None:
    if row.provider.casefold() != provider_name:
        raise ValueError("Historical acquisition mixed provider identities")
    if row.instrument_id != request.instrument_id:
        raise ValueError("Historical acquisition mixed instruments")
    if not (request.starts_at <= row.opens_at < request.ends_at):
        raise ValueError("Historical provider returned a row outside the requested interval")
    if not (chunk_start <= row.opens_at and row.closes_at <= chunk_end):
        raise ValueError("Historical provider returned a row outside its acquisition chunk")
    if row.volume is not None:
        raise ValueError("NIFTY 50 spot volume must remain null")


def historical_rows_sha256(rows: tuple[FinalizedMinuteBarInput, ...]) -> str:
    """Hash the exact ordered canonical research rows without provider secrets."""
    payload = [
        {
            "provider_bar_id": row.provider_bar_id,
            "provider": row.provider,
            "instrument_id": row.instrument_id,
            "opens_at": row.opens_at.astimezone(timezone.utc).isoformat(),
            "closes_at": row.closes_at.astimezone(timezone.utc).isoformat(),
            "open": format(row.open, "f"),
            "high": format(row.high, "f"),
            "low": format(row.low, "f"),
            "close": format(row.close, "f"),
            "volume": None,
            "provider_revision": row.provider_revision,
            "source_watermark": row.source_watermark,
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)
