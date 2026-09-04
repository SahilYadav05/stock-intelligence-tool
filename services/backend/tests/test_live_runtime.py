from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import IsolatedAsyncioTestCase

from nifty_terminal.delivery.service import MarketStateDeliveryService
from nifty_terminal.domain.candle import FinalizedMinuteBarInput
from nifty_terminal.domain.enums import ConnectionState, MarketEventType, TimestampSource
from nifty_terminal.domain.market_event import RawMarketEvent
from nifty_terminal.providers.base import ProviderAdapter, ProviderHealth
from nifty_terminal.runtime.live_market import LiveMarketRuntime, LiveRuntimeConfig


NOW = datetime(2026, 8, 24, 4, 46, 1, tzinfo=timezone.utc)  # 10:16:01 IST
CAS_NOW = datetime(2026, 8, 24, 9, 52, 0, tzinfo=timezone.utc)  # 15:22:00 IST
SESSION_OPEN = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)
STREAM_END = object()


class FakeLiveMinuteProvider(ProviderAdapter):
    def __init__(self, bars: tuple[FinalizedMinuteBarInput, ...]) -> None:
        self._bars = bars
        self._queue: asyncio.Queue[RawMarketEvent | object] = asyncio.Queue()
        self._state = ConnectionState.DISCONNECTED
        self.disconnect_calls = 0

    @property
    def provider_name(self) -> str:
        return "angelone"

    @property
    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider_name,
            connection_state=self._state,
            observed_at=NOW,
        )

    async def connect(self) -> None:
        self._state = ConnectionState.LIVE

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._state = ConnectionState.DISCONNECTED
        await self._queue.put(STREAM_END)

    async def emit(self, event: RawMarketEvent) -> None:
        await self._queue.put(event)

    async def stream(self) -> AsyncIterator[RawMarketEvent]:
        while True:
            item = await self._queue.get()
            if item is STREAM_END:
                return
            assert isinstance(item, RawMarketEvent)
            yield item

    async def fetch_finalized_minutes(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[FinalizedMinuteBarInput, ...]:
        return tuple(
            bar
            for bar in self._bars
            if from_time <= bar.opens_at and bar.closes_at <= to_time
        )


class LiveMarketRuntimeTests(IsolatedAsyncioTestCase):
    async def test_cas_observations_never_mutate_standard_candles_or_signals(self) -> None:
        provider = FakeLiveMinuteProvider(_cas_history())
        delivery = MarketStateDeliveryService()
        runtime = LiveMarketRuntime(
            adapter=provider,
            delivery=delivery,
            config=LiveRuntimeConfig(history_poll_seconds=60),
            clock=lambda: CAS_NOW,
        )

        await runtime.start()
        try:
            await runtime.wait_until_ready(timeout=2)
            view = delivery.read_model.get("NIFTY50_SPOT")
            self.assertIsNotNone(view)
            self.assertEqual(runtime.health.data_status, ConnectionState.MARKET_CLOSED)
            self.assertEqual(
                runtime.health.reason,
                "NSE_CLOSING_AUCTION_ACTIVE_STANDARD_SIGNAL_DISABLED",
            )
            self.assertEqual(runtime.health.historical_auction_observations, 2)
            self.assertEqual(
                view.snapshot.decision_time,  # type: ignore[union-attr]
                datetime(2026, 8, 24, 9, 45, tzinfo=timezone.utc),
            )
            self.assertIn(
                "CLOSING_AUCTION_STANDARD_SIGNAL_DISABLED",
                view.snapshot.blockers,  # type: ignore[union-attr]
            )
            await provider.emit(
                _tick(
                    CAS_NOW - timedelta(milliseconds=100),
                    Decimal("24350"),
                    sequence=99,
                    arrival=CAS_NOW,
                )
            )
            await _wait_until(lambda: runtime.health.live_auction_observations == 1)
            latest = delivery.read_model.get("NIFTY50_SPOT")
            self.assertIsNone(latest.developing_candle)  # type: ignore[union-attr]
        finally:
            await runtime.stop()

    async def test_tick_updates_visual_candle_but_never_model_inputs(self) -> None:
        provider = FakeLiveMinuteProvider(_history())
        delivery = MarketStateDeliveryService()
        runtime = LiveMarketRuntime(
            adapter=provider,
            delivery=delivery,
            config=LiveRuntimeConfig(
                history_lookback_days=2,
                history_recovery_minutes=15,
                history_poll_seconds=60,
                minute_finalization_delay_seconds=5,
                tick_fresh_seconds=3,
                tick_stale_seconds=15,
                chart_publish_interval_milliseconds=100,
            ),
            clock=lambda: NOW,
        )

        await runtime.start()
        try:
            await runtime.wait_until_ready(timeout=2)
            initial = delivery.read_model.get("NIFTY50_SPOT")
            self.assertIsNotNone(initial)
            self.assertEqual(initial.snapshot.data_status, ConnectionState.RECOVERING)  # type: ignore[union-attr]
            self.assertIsNone(initial.developing_candle)  # type: ignore[union-attr]
            history = runtime.chart_history(decision_time=initial.snapshot.decision_time)  # type: ignore[union-attr]
            self.assertEqual(sum(item.timeframe.value == "5m" for item in history), 12)
            self.assertEqual(sum(item.timeframe.value == "15m" for item in history), 4)
            self.assertEqual(sum(item.timeframe.value == "1h" for item in history), 1)
            self.assertTrue(
                all(item.closes_at <= initial.snapshot.decision_time for item in history)  # type: ignore[union-attr]
            )

            event_time = NOW - timedelta(milliseconds=100)
            await provider.emit(_tick(event_time, Decimal("25042.25"), sequence=1))
            view = await _wait_for_developing(delivery)

            self.assertEqual(view.snapshot.data_status, ConnectionState.LIVE)
            self.assertFalse(view.snapshot.live_inference_eligible)
            self.assertIn("LIVE_SIGNAL_KILL_SWITCH_ACTIVE", view.snapshot.blockers)
            self.assertIsNotNone(view.developing_candle)
            developing = view.developing_candle
            assert developing is not None
            self.assertEqual(developing.close, Decimal("25042.25"))
            self.assertIsNone(developing.volume)
            self.assertNotIn(
                developing.candle_id,
                view.snapshot.model_input_candle_ids,
            )
            self.assertEqual(view.snapshot.data_as_of, event_time)
            self.assertEqual(view.snapshot.decision_time, datetime(2026, 8, 24, 4, 45, tzinfo=timezone.utc))
            self.assertTrue(
                all(candle.status.value == "FINALIZED" for candle in view.finalized_candles)
            )
            self.assertTrue(
                all(candle.volume is None for candle in view.finalized_candles)
            )
            self.assertEqual(runtime.health.canonical_events_stored, 1)
        finally:
            await runtime.stop()

        self.assertGreaterEqual(provider.disconnect_calls, 1)
        self.assertEqual(runtime.health.data_status, ConnectionState.DISCONNECTED)

    async def test_duplicate_tick_is_counted_and_not_reapplied(self) -> None:
        provider = FakeLiveMinuteProvider(_history())
        delivery = MarketStateDeliveryService()
        runtime = LiveMarketRuntime(
            adapter=provider,
            delivery=delivery,
            config=LiveRuntimeConfig(history_poll_seconds=60),
            clock=lambda: NOW,
        )
        duplicate = _tick(NOW - timedelta(milliseconds=100), Decimal("25042.25"), sequence=8)

        await runtime.start()
        try:
            await runtime.wait_until_ready(timeout=2)
            await provider.emit(duplicate)
            await _wait_for_developing(delivery)
            await provider.emit(duplicate)
            await _wait_until(lambda: runtime.health.duplicate_events == 1)
        finally:
            await runtime.stop()

        self.assertEqual(runtime.health.raw_events_received, 2)
        self.assertEqual(runtime.health.canonical_events_stored, 1)


def _history() -> tuple[FinalizedMinuteBarInput, ...]:
    rows: list[FinalizedMinuteBarInput] = []
    for index in range(60):
        opens_at = SESSION_OPEN + timedelta(minutes=index)
        price = Decimal("25000") + Decimal(index) / Decimal("10")
        rows.append(
            FinalizedMinuteBarInput(
                provider_bar_id=f"angelone:{opens_at.isoformat()}:{price}",
                provider="angelone",
                instrument_id="NIFTY50_SPOT",
                opens_at=opens_at,
                closes_at=opens_at + timedelta(minutes=1),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal("0.25"),
                volume=None,
                provider_revision=1,
                finalized_at=NOW,
                source_watermark=f"history-{index}",
            )
        )
    return tuple(rows)


def _cas_history() -> tuple[FinalizedMinuteBarInput, ...]:
    rows: list[FinalizedMinuteBarInput] = []
    for index in range(360):
        opens_at = SESSION_OPEN + timedelta(minutes=index)
        price = Decimal("24300") + Decimal(index) / Decimal("10")
        rows.append(
            FinalizedMinuteBarInput(
                provider_bar_id=f"continuous:{opens_at.isoformat()}:{price}",
                provider="angelone",
                instrument_id="NIFTY50_SPOT",
                opens_at=opens_at,
                closes_at=opens_at + timedelta(minutes=1),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal("0.25"),
                volume=None,
                provider_revision=1,
                finalized_at=CAS_NOW,
                source_watermark=f"continuous-{index}",
            )
        )
    for minute in (0, 5):
        opens_at = datetime(2026, 8, 24, 9, 45, tzinfo=timezone.utc) + timedelta(
            minutes=minute
        )
        rows.append(
            FinalizedMinuteBarInput(
                provider_bar_id=f"auction:{opens_at.isoformat()}",
                provider="angelone",
                instrument_id="NIFTY50_SPOT",
                opens_at=opens_at,
                closes_at=opens_at + timedelta(minutes=1),
                open=Decimal("24340"),
                high=Decimal("24340"),
                low=Decimal("24340"),
                close=Decimal("24340"),
                volume=None,
                provider_revision=1,
                finalized_at=CAS_NOW,
                source_watermark=f"auction-{minute}",
            )
        )
    return tuple(rows)


def _tick(
    event_time: datetime,
    price: Decimal,
    *,
    sequence: int,
    arrival: datetime = NOW,
) -> RawMarketEvent:
    return RawMarketEvent(
        provider="angelone",
        provider_instrument_id="NSE:99926000",
        event_type=MarketEventType.INDEX_VALUE,
        server_arrival_time=arrival,
        connection_epoch="test-connection",
        raw_payload={"sequence": sequence, "price": format(price, "f")},
        provider_event_time=event_time,
        timestamp_source=TimestampSource.EXCHANGE,
        provider_sequence=sequence,
        provider_sequence_scope="test-connection:NSE:99926000",
        price=price,
    )


async def _wait_for_developing(
    delivery: MarketStateDeliveryService,
):
    value = None
    for _ in range(100):
        value = delivery.read_model.get("NIFTY50_SPOT")
        if value is not None and value.developing_candle is not None:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("Developing candle was not published")


async def _wait_until(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for runtime state")
