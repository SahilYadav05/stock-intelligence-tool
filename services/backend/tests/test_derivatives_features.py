from __future__ import annotations

from unittest import TestCase

from nifty_terminal.derivatives.features import (
    DERIVATIVES_FEATURE_NAMES,
    build_derivatives_feature_rows,
    evaluate_derivatives_readiness,
)


class DerivativesFeatureTests(TestCase):
    def test_features_use_only_current_and_prior_point_in_time_snapshots(self) -> None:
        rows = build_derivatives_feature_rows(
            (
                _snapshot("a", "2026-09-04T03:45:00Z", spot=25000, future=25050, oi=1000),
                _snapshot("b", "2026-09-04T03:50:00Z", spot=25025, future=25080, oi=1010),
            )
        )

        self.assertFalse(rows[0].complete_core)
        self.assertTrue(rows[1].complete_core)
        values = dict(zip(DERIVATIVES_FEATURE_NAMES, rows[1].feature_values))
        self.assertAlmostEqual(values["derivatives__spot_return_1"], 0.001)
        self.assertAlmostEqual(
            values["derivatives__open_interest_change_pct"], 0.01
        )
        self.assertEqual(values["derivatives__volume_delta_log1p"], 0.0)

    def test_contract_roll_resets_change_features(self) -> None:
        first = _snapshot("a", "2026-09-04T03:45:00Z", spot=25000, future=25050, oi=1000)
        second = _snapshot("b", "2026-09-04T03:50:00Z", spot=25025, future=25080, oi=1010)
        second["futures_symbol"] = "NIFTY29OCT26FUT"

        rows = build_derivatives_feature_rows((first, second))

        self.assertFalse(rows[1].complete_core)
        values = dict(zip(DERIVATIVES_FEATURE_NAMES, rows[1].feature_values))
        self.assertIsNone(values["derivatives__open_interest_change_pct"])

    def test_readiness_is_a_hard_support_and_completeness_gate(self) -> None:
        snapshots = (
            _snapshot("a", "2026-09-04T03:45:00Z", spot=25000, future=25050, oi=1000),
            _snapshot("b", "2026-09-04T03:50:00Z", spot=25025, future=25080, oi=1010),
        )

        blocked = evaluate_derivatives_readiness(snapshots)
        ready = evaluate_derivatives_readiness(
            snapshots,
            minimum_complete_rows=1,
            minimum_sessions=1,
            minimum_calendar_span_days=1,
            minimum_options_complete_share=0.5,
        )

        self.assertFalse(blocked.ready_for_model_research)
        self.assertTrue(ready.ready_for_model_research)
        self.assertTrue(ready.to_contract()["research_only"])


def _snapshot(snapshot_id: str, observed_at: str, *, spot: float, future: float, oi: float):
    return {
        "snapshot_id": snapshot_id,
        "observed_at": observed_at,
        "spot_ltp": spot,
        "futures_symbol": "NIFTY24SEP26FUT",
        "futures_ltp": future,
        "futures_volume": 1000,
        "futures_open_interest": oi,
        "futures_book_imbalance": 0.2,
        "futures_basis_bps": (future - spot) / spot * 10_000,
        "provider_put_call_ratio": 1.1,
        "atm_implied_volatility": 13.0,
        "delta25_put_call_iv_skew": 2.0,
        "option_volume_put_call_ratio": 1.2,
    }
