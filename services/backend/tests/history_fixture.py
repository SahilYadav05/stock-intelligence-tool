from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe


SESSION_OPEN = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)
CSV_COLUMNS = (
    "provider_bar_id",
    "provider_revision",
    "opens_at",
    "closes_at",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "finalized_at",
    "source_watermark",
)


def write_history_csv(
    path: Path,
    indexes: list[int],
    *,
    volume: str = "",
    correction_after: bool = False,
) -> None:
    rows = [_csv_row(index, volume=volume) for index in indexes]
    if correction_after:
        correction = _csv_row(indexes[0], volume=volume, revision=2, adjustment=Decimal("5"))
        rows.append(correction)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def finalized_candles(
    count: int,
    *,
    timeframe: Timeframe = Timeframe.M5,
    skip_index: int | None = None,
) -> tuple[Candle, ...]:
    candles: list[Candle] = []
    for index in range(count):
        if index == skip_index:
            continue
        opens_at = SESSION_OPEN + timedelta(minutes=index * timeframe.minutes)
        price = Decimal("25000") + Decimal(index * 3) + Decimal(index % 4)
        candles.append(
            Candle(
                schema_version=1,
                candle_id=f"00000000-0000-5000-8000-{index:012d}",
                instrument_id="NIFTY50_SPOT",
                timeframe=timeframe,
                opens_at=opens_at,
                closes_at=opens_at + timedelta(minutes=timeframe.minutes),
                open=price,
                high=price + Decimal("4"),
                low=price - Decimal("2"),
                close=price + Decimal("2"),
                volume=None,
                status=CandleStatus.FINALIZED,
                revision=1,
                source=CandleSource.AGGREGATED,
                provider="replay",
                source_revision=1,
                finalized_at=opens_at + timedelta(minutes=timeframe.minutes, seconds=1),
                component_candle_ids=(f"component-{index}",),
                source_watermark=f"watermark-{index}",
            )
        )
    return tuple(candles)


def _csv_row(
    index: int,
    *,
    volume: str,
    revision: int = 1,
    adjustment: Decimal = Decimal("0"),
) -> dict[str, str]:
    opens_at = SESSION_OPEN + timedelta(minutes=index)
    price = Decimal("25000") + Decimal(index) + adjustment
    return {
        "provider_bar_id": f"provider-bar-{index}-r{revision}",
        "provider_revision": str(revision),
        "opens_at": _time(opens_at),
        "closes_at": _time(opens_at + timedelta(minutes=1)),
        "open": str(price),
        "high": str(price + 2),
        "low": str(price - 1),
        "close": str(price + 1),
        "volume": volume,
        "finalized_at": _time(opens_at + timedelta(minutes=1, seconds=1)),
        "source_watermark": f"watermark-{index}-r{revision}",
    }


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
