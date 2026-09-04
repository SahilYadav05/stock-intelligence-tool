from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest import IsolatedAsyncioTestCase, TestCase

from market_state_fixture import build_market_state_view
from nifty_terminal.delivery.hub import MarketStateHub
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.delivery.read_model import InMemoryMarketStateReadModel


class MarketStateViewTests(TestCase):
    def test_contract_contains_one_atomic_synced_snapshot(self) -> None:
        view = build_market_state_view()
        contract = view.to_contract()

        self.assertEqual(contract["sync_state"], "SYNCED")
        self.assertEqual(contract["snapshot"]["primary_timeframe"], "5m")  # type: ignore[index]
        self.assertGreater(len(contract["finalized_candles"]), 0)  # type: ignore[arg-type]

    def test_missing_snapshot_revision_fails_before_publication(self) -> None:
        view = build_market_state_view()
        with self.assertRaisesRegex(ValueError, "missing"):
            MarketStateView(
                schema_version=1,
                snapshot=view.snapshot,
                finalized_candles=view.finalized_candles[1:],
                developing_candle=None,
                published_at=view.published_at,
            )

    def test_invalid_snapshot_checksum_fails_before_publication(self) -> None:
        view = build_market_state_view()
        invalid_snapshot = replace(view.snapshot, candle_revision_checksum="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            replace(view, snapshot=invalid_snapshot)

    def test_read_model_rejects_time_regression(self) -> None:
        view = build_market_state_view()
        store = InMemoryMarketStateReadModel()
        store.put(view)
        with self.assertRaisesRegex(ValueError, "older"):
            store.put(replace(view, published_at=view.published_at - timedelta(seconds=1)))


class MarketStateHubTests(IsolatedAsyncioTestCase):
    async def test_sequences_are_monotonic_and_queue_is_bounded(self) -> None:
        view = build_market_state_view()
        hub = MarketStateHub(subscriber_queue_size=1)
        async with hub.subscribe("NIFTY50_SPOT") as queue:
            first = await hub.publish(view)
            second = await hub.publish(view)
            delivered = await queue.get()

        self.assertEqual(first.sequence, 1)
        self.assertEqual(second.sequence, 2)
        self.assertEqual(delivered.sequence, 2)
