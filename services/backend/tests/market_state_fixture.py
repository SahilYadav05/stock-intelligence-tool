from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.candle import Candle, FinalizedMinuteBarInput, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.snapshots.builder import MarketStateSnapshotBuilder
from nifty_terminal.snapshots.models import DataMode
from nifty_terminal.snapshots.store import InMemorySnapshotStore


SESSION_OPEN = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)


def build_market_state_view() -> MarketStateView:
    candle_store = InMemoryCandleStore()
    engine = CandleEngine(
        calendar=NseSessionCalendar(),
        registry=build_mvp_instrument_registry(),
        store=candle_store,
    )
    latest_five: Candle | None = None
    for index in range(60):
        opens_at = SESSION_OPEN + timedelta(minutes=index)
        price = Decimal("25000") + Decimal(index)
        result = engine.ingest_finalized_minute(
            FinalizedMinuteBarInput(
                provider_bar_id=f"transport-bar-{index}",
                provider="replay",
                instrument_id="NIFTY50_SPOT",
                opens_at=opens_at,
                closes_at=opens_at + timedelta(minutes=1),
                open=price,
                high=price + 2,
                low=price - 1,
                close=price + 1,
                volume=None,
                provider_revision=1,
                finalized_at=opens_at + timedelta(minutes=1, seconds=1),
                source_watermark=f"transport-watermark-{index}",
            )
        )
        for candle in result.finalized_candles:
            if candle.timeframe is Timeframe.M5:
                latest_five = candle
    assert latest_five is not None

    snapshot = MarketStateSnapshotBuilder(
        candle_store=candle_store,
        snapshot_store=InMemorySnapshotStore(),
    ).build(
        primary_candle=latest_five,
        created_at=latest_five.closes_at + timedelta(seconds=2),
        data_mode=DataMode.LIVE,
        data_status=ConnectionState.LIVE,
    )
    finalized = tuple(
        candle
        for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1)
        for candle in candle_store.latest_series("NIFTY50_SPOT", timeframe)
    )
    return MarketStateView(
        schema_version=1,
        snapshot=snapshot,
        finalized_candles=finalized,
        developing_candle=None,
        published_at=latest_five.closes_at + timedelta(seconds=3),
    )
