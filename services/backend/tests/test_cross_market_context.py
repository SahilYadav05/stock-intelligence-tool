from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from nifty_terminal.context.bundle import (
    ContextBar,
    ContextBundle,
    ContextInstrument,
    bundle_sha256,
    read_bundle,
    write_bundle,
)
from nifty_terminal.context.features import (
    CONTEXT_FEATURE_SET_HASH,
    CONTEXT_FEATURE_VERSION,
    build_context_feature_matrix,
)

from test_trade_aligned_research import _candles, _dataset
from test_model_v2_research import _sample
from nifty_terminal.domain.candle import Timeframe


class CrossMarketContextTests(TestCase):
    def test_step18c_contract_can_never_release_a_model_or_signal(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with (root / "contracts" / "cross-market-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            import json
            schema = json.load(file)
        for name in (
            "model_artifact_created",
            "approved_for_live_inference",
            "precise_probability_display_allowed",
            "official_signal_available",
            "automatic_trading_enabled",
        ):
            self.assertFalse(schema["properties"][name]["const"])

    def test_bundle_is_content_addressed_and_round_trips(self) -> None:
        bundle = _bundle()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "context.json.gz"
            digest = write_bundle(path, bundle)
            restored = read_bundle(path)
        self.assertEqual(digest, bundle_sha256(restored))
        self.assertEqual(len(digest), 64)
        self.assertEqual(restored.instruments[0].bars, bundle.instruments[0].bars)

    def test_context_features_are_exact_finalized_and_future_invariant(self) -> None:
        candles = _candles(Timeframe.M5, 90, minutes=5)
        sample = replace(
            _sample(),
            primary_candle_id=candles[60].candle_id,
            decision_time=candles[60].closes_at,
            label_window_end=candles[60].closes_at + timedelta(minutes=60),
        )
        dataset = _dataset(sample)
        bundle = _bundle(start=candles[0].opens_at, count=450)
        original = build_context_feature_matrix(
            dataset=dataset, primary_candles=candles, bundle=bundle
        )
        instruments = list(bundle.instruments)
        future_bars = list(instruments[0].bars)
        future_bars[400] = replace(future_bars[400], close=Decimal("999999"), high=Decimal("999999"))
        instruments[0] = replace(instruments[0], bars=tuple(future_bars))
        changed = build_context_feature_matrix(
            dataset=dataset,
            primary_candles=candles,
            bundle=replace(bundle, instruments=tuple(instruments)),
        )
        self.assertEqual(original.matrix.rows, changed.matrix.rows)
        self.assertEqual(original.dataset.eligible_samples, 1)
        self.assertIn("context_market__banknifty_spot__return_1", original.matrix.feature_names)
        self.assertEqual(CONTEXT_FEATURE_VERSION, "canonical_cross_market_features.v1")
        self.assertEqual(len(CONTEXT_FEATURE_SET_HASH), 64)

    def test_missing_exact_context_candle_is_not_imputed(self) -> None:
        candles = _candles(Timeframe.M5, 90, minutes=5)
        sample = replace(
            _sample(),
            primary_candle_id=candles[60].candle_id,
            decision_time=candles[60].closes_at,
            label_window_end=candles[60].closes_at + timedelta(minutes=60),
        )
        dataset = _dataset(sample)
        bundle = _bundle(start=candles[0].opens_at, count=450)
        instruments = list(bundle.instruments)
        missing_open = sample.decision_time - timedelta(minutes=1)
        instruments[1] = replace(
            instruments[1],
            bars=tuple(item for item in instruments[1].bars if item.opens_at != missing_open),
        )
        result = build_context_feature_matrix(
            dataset=dataset,
            primary_candles=candles,
            bundle=replace(bundle, instruments=tuple(instruments)),
        )
        self.assertEqual(result.dataset.eligible_samples, 0)
        self.assertEqual(result.diagnostics["excluded_samples"]["CONTEXT_5M_EXACT_CLOSE_MISSING"], 1)
        self.assertFalse(result.diagnostics["missing_values_imputed"])


def _bundle(*, start=None, count: int = 300) -> ContextBundle:
    if start is None:
        start = _candles(Timeframe.M5, 1, minutes=5)[0].opens_at
    instruments = []
    for name, base, kind in (
        ("BANKNIFTY_SPOT", Decimal("52000"), "INDEX"),
        ("INDIA_VIX_SPOT", Decimal("14"), "VOLATILITY_INDEX"),
    ):
        bars = []
        price = base
        for index in range(count):
            change = Decimal(str((index % 9 - 4) * 0.01))
            close = price + change
            bars.append(ContextBar(
                opens_at=start + timedelta(minutes=index),
                open=price,
                high=max(price, close) + Decimal("0.02"),
                low=min(price, close) - Decimal("0.02"),
                close=close,
                volume=None,
            ))
            price = close
        instruments.append(ContextInstrument(
            instrument_id=name,
            provider="test",
            exchange="NSE",
            token="1" if name.startswith("BANK") else "2",
            asset_kind=kind,
            bars=tuple(bars),
            expected_minutes=count,
            excluded_out_of_session=0,
        ))
    return ContextBundle(
        schema_version=1,
        provider="test",
        requested_from="2026-08-24",
        requested_through="2026-08-24",
        acquired_at="2026-08-25T00:00:00Z",
        instruments=tuple(instruments),
        source_notes=(),
    )
