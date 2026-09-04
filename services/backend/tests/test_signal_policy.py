from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase

from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.signals.lifecycle import assess_signal
from nifty_terminal.signals.models import (
    SignalContext,
    SignalDirection,
    SignalEventType,
    SignalLifecycleStatus,
)
from nifty_terminal.signals.policy import SignalPolicy


class SignalPolicyTests(TestCase):
    def test_buy_requires_every_hard_gate_and_builds_atr_levels(self) -> None:
        decision = SignalPolicy().evaluate(_context((0.10, 0.15, 0.75)))

        self.assertEqual(decision.direction, SignalDirection.BUY)
        self.assertEqual(decision.lifecycle_status, SignalLifecycleStatus.ACTIVE)
        self.assertEqual(decision.risk_levels.entry_low, Decimal("24990.00"))
        self.assertEqual(decision.risk_levels.entry_high, Decimal("25010.00"))
        self.assertEqual(decision.risk_levels.stop, Decimal("24925.00"))
        self.assertEqual(decision.risk_levels.target1, Decimal("25100.00"))
        self.assertGreaterEqual(decision.risk_levels.target1_reward_risk, 1.25)
        self.assertIsNotNone(decision.probabilities)

    def test_sell_uses_symmetric_underlying_levels(self) -> None:
        decision = SignalPolicy().evaluate(_context((0.76, 0.14, 0.10)))

        self.assertEqual(decision.direction, SignalDirection.SELL)
        self.assertEqual(decision.risk_levels.stop, Decimal("25075.00"))
        self.assertEqual(decision.risk_levels.target1, Decimal("24900.00"))
        self.assertFalse(decision.to_contract()["automatic_execution"])

    def test_stale_data_forces_wait_and_hides_probability(self) -> None:
        decision = SignalPolicy().evaluate(
            _context((0.10, 0.15, 0.75), data_status=ConnectionState.STALE)
        )

        self.assertEqual(decision.direction, SignalDirection.WAIT)
        self.assertIn("DATA_NOT_LIVE_STALE", decision.blockers)
        self.assertIsNone(decision.risk_levels)
        self.assertIsNone(decision.probabilities)

    def test_failed_calibration_forces_wait_and_withholds_precise_probability(self) -> None:
        decision = SignalPolicy().evaluate(
            _context((0.10, 0.15, 0.75), calibration_release_passed=False)
        )

        self.assertEqual(decision.direction, SignalDirection.WAIT)
        self.assertIn("CALIBRATION_RELEASE_GATE_NOT_PASSED", decision.blockers)
        self.assertIsNone(decision.probabilities)

    def test_active_opposite_signal_cannot_flip_directly(self) -> None:
        decision = SignalPolicy().evaluate(
            _context(
                (0.05, 0.08, 0.87),
                current_active_direction=SignalDirection.SELL,
            )
        )

        self.assertEqual(decision.direction, SignalDirection.WAIT)
        self.assertIn("CONFLICTING_ACTIVE_SIGNAL_MUST_INVALIDATE_FIRST", decision.blockers)

    def test_low_margin_is_wait_even_when_direction_is_largest_class(self) -> None:
        decision = SignalPolicy().evaluate(_context((0.18, 0.23, 0.59)))

        self.assertEqual(decision.direction, SignalDirection.WAIT)
        self.assertIn("PROBABILITY_THRESHOLD_NOT_MET", decision.blockers)

    def test_lifecycle_assessment_is_separate_and_does_not_mutate_signal(self) -> None:
        signal = SignalPolicy().evaluate(_context((0.10, 0.15, 0.75)))
        event = assess_signal(
            signal,
            observed_at=signal.decision_time + timedelta(minutes=5),
            high=Decimal("25110"),
            low=Decimal("24980"),
        )

        self.assertEqual(event.event_type, SignalEventType.TARGET_HIT)
        self.assertEqual(event.status, SignalLifecycleStatus.TARGET_HIT)
        self.assertEqual(signal.lifecycle_status, SignalLifecycleStatus.ACTIVE)

    def test_ambiguous_same_candle_stop_and_target_invalidates(self) -> None:
        signal = SignalPolicy().evaluate(_context((0.10, 0.15, 0.75)))
        event = assess_signal(
            signal,
            observed_at=signal.decision_time + timedelta(minutes=5),
            high=Decimal("25110"),
            low=Decimal("24900"),
        )

        self.assertEqual(event.event_type, SignalEventType.INVALIDATED)
        self.assertEqual(event.reason, "AMBIGUOUS_INTRABAR_STOP_AND_TARGET_ORDER")


def _context(
    probabilities: tuple[float, float, float],
    **overrides,
) -> SignalContext:
    values = {
        "prediction_id": "prediction-1",
        "calibration_id": "calibration-1",
        "snapshot_id": "snapshot-1",
        "instrument_id": "NIFTY50_SPOT",
        "decision_time": datetime(2026, 8, 24, 5, 5, tzinfo=timezone.utc),
        "data_as_of": datetime(2026, 8, 24, 5, 5, tzinfo=timezone.utc),
        "input_revision_checksum": "a" * 64,
        "calibrated_probabilities": tuple(
            zip(("DOWN", "NEITHER", "UP"), probabilities, strict=True)
        ),
        "reference_close": Decimal("25000"),
        "atr": Decimal("100"),
        "data_status": ConnectionState.LIVE,
        "snapshot_synced": True,
        "finalized_primary": True,
        "finalized_15m_context": True,
        "finalized_1h_context": True,
        "feature_ready": True,
        "calibration_release_passed": True,
        "probability_bin_supported": True,
        "event_risk_clear": True,
        "current_active_direction": None,
    }
    values.update(overrides)
    return SignalContext(**values)
