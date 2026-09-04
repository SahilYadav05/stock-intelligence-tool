from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from unittest import TestCase

from fastapi.testclient import TestClient

from market_state_fixture import build_market_state_view
from nifty_terminal.api.app import create_app
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.delivery.read_model import InMemoryMarketStateReadModel
from nifty_terminal.delivery.service import MarketStateDeliveryService
from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.price_action.engine import PriceActionEngine
from nifty_terminal.price_action.models import PriceActionBias, SetupState
from nifty_terminal.features.research_v4 import (
    PRICE_ACTION_FEATURE_NAMES,
    PRICE_ACTION_FEATURE_SET_HASH,
    build_price_action_research_matrix,
)
from nifty_terminal.settings import Settings
from nifty_terminal.snapshots.models import DataMode, MarketStateSnapshot
from test_model_v2_research import _sample
from test_trade_aligned_research import _candles, _dataset


def _settings() -> Settings:
    return Settings(
        app_name="price-action-test",
        environment="test",
        log_level="WARNING",
        market_data_mode="replay",
        market_data_provider=None,
    )


def _candle(
    *,
    timeframe: Timeframe,
    index: int,
    count: int,
    decision_time: datetime,
    bullish: bool,
) -> Candle:
    minutes = timeframe.minutes
    closes_at = decision_time - timedelta(minutes=minutes * (count - index - 1))
    opens_at = closes_at - timedelta(minutes=minutes)
    wave = (Decimal(0), Decimal(5), Decimal(9), Decimal(4), Decimal(-1), Decimal(-6), Decimal(-9), Decimal(-4))[index % 8]
    direction = Decimal(1) if bullish else Decimal(-1)
    close = Decimal("24000") + direction * (Decimal(index) * Decimal("2.2") + wave)
    open_price = close - direction * Decimal("1.2")
    high = max(open_price, close) + Decimal("2")
    low = min(open_price, close) - Decimal("2")
    if index == count - 1:
        close += direction * Decimal("18")
        high = max(high, close + Decimal("2"))
        low = min(low, open_price - Decimal("2"))
    return Candle(
        schema_version=1,
        candle_id=f"{timeframe.value}-{index}",
        instrument_id="NIFTY50_SPOT",
        timeframe=timeframe,
        opens_at=opens_at,
        closes_at=closes_at,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=None,
        status=CandleStatus.FINALIZED,
        revision=1,
        source=CandleSource.AGGREGATED,
        provider="test",
        source_revision=1,
        finalized_at=closes_at + timedelta(seconds=1),
        component_candle_ids=(),
        source_watermark=f"watermark-{timeframe.value}-{index}",
    )


def _view(*, bullish: bool = True, live: bool = True) -> MarketStateView:
    decision_time = datetime(2026, 8, 24, 10, 0, tzinfo=timezone.utc)
    counts = {Timeframe.M5: 80, Timeframe.M15: 55, Timeframe.H1: 55}
    candles = tuple(
        _candle(
            timeframe=timeframe,
            index=index,
            count=count,
            decision_time=decision_time,
            bullish=bullish,
        )
        for timeframe, count in counts.items()
        for index in range(count)
    )
    ids = tuple(item.candle_id for item in candles)
    checksum = hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()
    primary = next(
        item
        for item in reversed(candles)
        if item.timeframe is Timeframe.M5
    )
    context_15m = next(item for item in reversed(candles) if item.timeframe is Timeframe.M15)
    context_1h = next(item for item in reversed(candles) if item.timeframe is Timeframe.H1)
    snapshot = MarketStateSnapshot(
        schema_version=1,
        snapshot_id="price-action-snapshot",
        instrument_id="NIFTY50_SPOT",
        decision_time=decision_time,
        created_at=decision_time + timedelta(seconds=1),
        data_as_of=decision_time,
        data_mode=DataMode.LIVE,
        data_status=ConnectionState.LIVE if live else ConnectionState.MARKET_CLOSED,
        primary_timeframe=Timeframe.M5,
        primary_candle_id=primary.candle_id,
        context_15m_candle_id=context_15m.candle_id,
        context_1h_candle_id=context_1h.candle_id,
        recent_primary_candle_ids=tuple(
            item.candle_id for item in candles if item.timeframe is Timeframe.M5
        ),
        developing_candle_id=None,
        model_input_candle_ids=ids,
        source_watermark="price-action-watermark",
        candle_revision_checksum=checksum,
        live_inference_eligible=live,
        blockers=() if live else ("DATA_STATUS_MARKET_CLOSED",),
    )
    return MarketStateView(
        schema_version=1,
        snapshot=snapshot,
        finalized_candles=candles,
        developing_candle=None,
        published_at=decision_time + timedelta(seconds=2),
    )


class PriceActionEngineTests(TestCase):
    def test_bullish_structure_produces_ordered_conditional_risk_plan(self) -> None:
        analysis = PriceActionEngine().analyze(_view(bullish=True))

        self.assertEqual(analysis.bias, PriceActionBias.BULLISH)
        self.assertGreaterEqual(analysis.confluence_score, 25)
        self.assertIn(analysis.setup, {SetupState.BUY_TRIGGER, SetupState.BULLISH_WATCH})
        self.assertIsNotNone(analysis.trade_plan)
        assert analysis.trade_plan is not None
        plan = analysis.trade_plan
        self.assertEqual(plan.direction, "BUY")
        self.assertLess(plan.stop, plan.trigger)
        self.assertLess(plan.trigger, plan.target1)
        self.assertLess(plan.target1, plan.target2)
        self.assertLess(plan.target2, plan.target3)
        self.assertEqual(plan.target1_reward_risk, 1.25)

    def test_bearish_structure_produces_symmetric_sell_plan(self) -> None:
        analysis = PriceActionEngine().analyze(_view(bullish=False))

        self.assertEqual(analysis.bias, PriceActionBias.BEARISH)
        self.assertIsNotNone(analysis.trade_plan)
        assert analysis.trade_plan is not None
        plan = analysis.trade_plan
        self.assertEqual(plan.direction, "SELL")
        self.assertGreater(plan.stop, plan.trigger)
        self.assertGreater(plan.trigger, plan.target1)
        self.assertGreater(plan.target1, plan.target2)
        self.assertGreater(plan.target2, plan.target3)

    def test_closed_market_never_emits_a_live_trigger(self) -> None:
        analysis = PriceActionEngine().analyze(_view(live=False))

        self.assertNotIn(analysis.setup, {SetupState.BUY_TRIGGER, SetupState.SELL_TRIGGER})
        self.assertIn("DATA_NOT_LIVE_MARKET_CLOSED", analysis.blockers)

    def test_api_returns_snapshot_bound_research_contract(self) -> None:
        market = _view()
        store = InMemoryMarketStateReadModel()
        store.put(market)
        app = create_app(
            settings=_settings(),
            delivery=MarketStateDeliveryService(read_model=store),
        )
        with TestClient(app) as client:
            response = client.get(
                f"/api/v1/price-action/NIFTY50_SPOT?snapshot_id={market.snapshot.snapshot_id}"
            )

        payload = response.json()
        self.assertEqual(payload["sync_state"], "SYNCED")
        self.assertEqual(payload["analysis"]["snapshot_id"], market.snapshot.snapshot_id)
        self.assertTrue(payload["analysis"]["research_only"])
        self.assertFalse(payload["analysis"]["official_signal"])
        self.assertIsNone(payload["analysis"]["calibrated_probability"])

    def test_short_fixture_fails_closed_when_history_is_insufficient(self) -> None:
        market = build_market_state_view()
        analysis = PriceActionEngine().analyze(market)

        self.assertEqual(analysis.bias, PriceActionBias.UNAVAILABLE)
        self.assertEqual(analysis.setup, SetupState.UNAVAILABLE)
        self.assertTrue(any(item.startswith("INSUFFICIENT_5M_HISTORY") for item in analysis.blockers))

    def test_research_features_are_future_invariant_and_versioned(self) -> None:
        candles = _candles(Timeframe.M5, 90, minutes=5)
        sample = replace(
            _sample(),
            primary_candle_id=candles[60].candle_id,
            decision_time=candles[60].closes_at,
            label_window_end=candles[60].closes_at + timedelta(minutes=60),
        )
        dataset = _dataset(sample)

        original = build_price_action_research_matrix(dataset, candles)
        changed = list(candles)
        changed[80] = replace(
            changed[80],
            high=changed[80].high + Decimal("500"),
            low=changed[80].low - Decimal("500"),
            close=changed[80].close + Decimal("250"),
        )
        future_changed = build_price_action_research_matrix(dataset, tuple(changed))

        self.assertEqual(original.rows, future_changed.rows)
        self.assertTrue(set(PRICE_ACTION_FEATURE_NAMES).issubset(original.feature_names))
        self.assertEqual(len(PRICE_ACTION_FEATURE_SET_HASH), 64)
