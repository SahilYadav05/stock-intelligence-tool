from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from unittest import TestCase

from dashboard_fixture import build_analysis_view
from market_state_fixture import build_market_state_view
from nifty_terminal.dashboard.models import AnalysisView
from nifty_terminal.dashboard.read_model import InMemoryAnalysisReadModel


class DashboardAnalysisTests(TestCase):
    def test_analysis_contract_references_exact_chart_snapshot_and_revision(self) -> None:
        market = build_market_state_view()
        analysis = build_analysis_view(market)
        contract = analysis.to_contract()

        self.assertEqual(contract["snapshot_id"], market.snapshot.snapshot_id)
        self.assertEqual(
            contract["candle_revision_checksum"],
            market.snapshot.candle_revision_checksum,
        )
        self.assertEqual(contract["signal"]["direction"], "WAIT")  # type: ignore[index]

    def test_analysis_rejects_signal_from_different_snapshot(self) -> None:
        market = build_market_state_view()
        valid = build_analysis_view(market)
        mismatched_signal = replace(valid.signal, snapshot_id="different-snapshot")

        with self.assertRaisesRegex(ValueError, "same snapshot"):
            replace(valid, signal=mismatched_signal)

    def test_read_model_rejects_time_regression(self) -> None:
        analysis = build_analysis_view(build_market_state_view())
        store = InMemoryAnalysisReadModel()
        store.put(analysis)

        with self.assertRaisesRegex(ValueError, "older"):
            store.put(replace(analysis, generated_at=analysis.generated_at - timedelta(seconds=1)))
