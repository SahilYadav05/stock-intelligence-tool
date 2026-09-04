from __future__ import annotations

from datetime import timedelta

from nifty_terminal.dashboard.models import AnalysisView, ContextStatus, NewsStatus
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.signals.definitions import RISK_POLICY_VERSION, SIGNAL_POLICY_VERSION
from nifty_terminal.signals.models import (
    SignalDecision,
    SignalDirection,
    SignalLifecycleStatus,
)


def build_analysis_view(market: MarketStateView) -> AnalysisView:
    snapshot = market.snapshot
    signal = SignalDecision(
        schema_version=1,
        signal_id="signal-dashboard-fixture",
        prediction_id="prediction-dashboard-fixture",
        calibration_id="calibration-dashboard-fixture",
        snapshot_id=snapshot.snapshot_id,
        instrument_id=snapshot.instrument_id,
        decision_time=snapshot.decision_time,
        created_at=snapshot.decision_time,
        expires_at=snapshot.decision_time + timedelta(minutes=60),
        direction=SignalDirection.WAIT,
        lifecycle_status=SignalLifecycleStatus.NO_SIGNAL,
        probabilities=None,
        expected_atr=None,
        risk_levels=None,
        blockers=("PROBABILITY_THRESHOLD_NOT_MET",),
        signal_policy_version=SIGNAL_POLICY_VERSION,
        risk_policy_version=RISK_POLICY_VERSION,
        input_revision_checksum=snapshot.candle_revision_checksum,
    )
    return AnalysisView(
        schema_version=1,
        analysis_id="analysis-dashboard-fixture",
        snapshot_id=snapshot.snapshot_id,
        candle_revision_checksum=snapshot.candle_revision_checksum,
        instrument_id=snapshot.instrument_id,
        decision_time=snapshot.decision_time,
        generated_at=snapshot.decision_time + timedelta(seconds=2),
        data_as_of=snapshot.data_as_of,
        signal=signal,
        model_version="research-model-fixture",
        calibration_version="multiclass_temperature.v1",
        feature_version="price_features.v1",
        market_context_status=ContextStatus.UNAVAILABLE,
        regime=None,
        trend=None,
        momentum=None,
        volatility=None,
        support_levels=(),
        resistance_levels=(),
        reasons=(),
        contradictory_evidence=(),
        news_status=NewsStatus.UNAVAILABLE,
        news_items=(),
        historical_analog_count=None,
        historical_analog_summary=None,
        historical_signals=(),
    )
