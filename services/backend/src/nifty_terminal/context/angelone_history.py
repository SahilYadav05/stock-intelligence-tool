"""Fail-closed Angel One acquisition for non-NIFTY research context."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import time as clock
from typing import Callable

from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.context.bundle import ContextBar, ContextInstrument
from nifty_terminal.domain.candle import Timeframe


Progress = Callable[[str, int, int, int], None]


def acquire_instrument(
    *,
    smart_client: object,
    calendar: NseSessionCalendar,
    instrument_id: str,
    exchange: str,
    token: str,
    asset_kind: str,
    from_date: date,
    through_date: date,
    chunk_days: int = 7,
    request_delay_ms: int = 400,
    progress: Progress | None = None,
) -> ContextInstrument:
    if not token.isascii() or not token.isdigit():
        raise ValueError(f"{instrument_id} token must contain ASCII digits")
    start = datetime.combine(from_date, time.min, IST).astimezone(timezone.utc)
    end = datetime.combine(through_date + timedelta(days=1), time.min, IST).astimezone(timezone.utc)
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    rows: dict[datetime, ContextBar] = {}
    for number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        request = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": "ONE_MINUTE",
            "fromdate": chunk_start.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
            "todate": chunk_end.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
        }
        response = smart_client.getCandleData(request)
        if not isinstance(response, dict) or response.get("status") is not True:
            code = response.get("errorcode", "UNKNOWN") if isinstance(response, dict) else "INVALID"
            raise RuntimeError(f"Angel One context request failed safely [{code}]")
        for raw in response.get("data") or ():
            bar = _bar(raw, asset_kind=asset_kind)
            if not (start <= bar.opens_at < end):
                raise ValueError(f"{instrument_id} returned an out-of-request timestamp")
            prior = rows.get(bar.opens_at)
            if prior is not None and prior != bar:
                raise ValueError(f"{instrument_id} returned conflicting duplicate minutes")
            rows[bar.opens_at] = bar
        if progress:
            progress(instrument_id, number, len(chunks), len(rows))
        if number < len(chunks) and request_delay_ms:
            clock.sleep(request_delay_ms / 1000)

    retained = []
    excluded = 0
    for opens_at in sorted(rows):
        bar = rows[opens_at]
        try:
            bucket = calendar.bucket_for(opens_at, Timeframe.M1)
            aligned = bucket.opens_at == opens_at
        except ValueError:
            aligned = False
        if aligned:
            retained.append(bar)
        else:
            excluded += 1
    expected = _expected_minutes(calendar, from_date, through_date)
    return ContextInstrument(
        instrument_id=instrument_id,
        provider="angelone",
        exchange=exchange,
        token=token,
        asset_kind=asset_kind,
        bars=tuple(retained),
        expected_minutes=expected,
        excluded_out_of_session=excluded,
    )


def _bar(raw: object, *, asset_kind: str) -> ContextBar:
    if not isinstance(raw, (list, tuple)) or len(raw) < 5:
        raise ValueError("Angel One context candle must contain timestamp and OHLC")
    opens_at = datetime.fromisoformat(str(raw[0]).replace("Z", "+00:00"))
    if opens_at.tzinfo is None:
        raise ValueError("Angel One context timestamp omitted its timezone")
    values = tuple(Decimal(str(raw[index])) for index in range(1, 5))
    opening, high, low, close = values
    if min(values) <= 0 or low > min(opening, close) or high < max(opening, close):
        raise ValueError("Angel One context candle violates OHLC invariants")
    volume = None
    if asset_kind not in {"INDEX", "VOLATILITY_INDEX"} and len(raw) > 5 and raw[5] is not None:
        volume = Decimal(str(raw[5]))
        if volume < 0:
            raise ValueError("Context volume cannot be negative")
    return ContextBar(opens_at.astimezone(timezone.utc), opening, high, low, close, volume)


def _expected_minutes(calendar: NseSessionCalendar, start: date, end: date) -> int:
    total = 0
    current = start
    while current <= end:
        session = calendar.session_for_date(current)
        if session is not None:
            total += int((session.closes_at - session.opens_at).total_seconds() // 60)
        current += timedelta(days=1)
    return total
