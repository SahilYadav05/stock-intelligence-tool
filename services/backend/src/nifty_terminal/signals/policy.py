"""Deterministic BUY/SELL/WAIT policy with explicit hard gates."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.signals.definitions import RISK_POLICY_VERSION, SIGNAL_POLICY_VERSION
from nifty_terminal.signals.models import (
    PolicyConfig,
    RiskLevels,
    SignalContext,
    SignalDecision,
    SignalDirection,
    SignalLifecycleStatus,
)


class SignalPolicy:
    def evaluate(
        self,
        context: SignalContext,
        config: PolicyConfig | None = None,
    ) -> SignalDecision:
        resolved = config or PolicyConfig()
        blockers = _hard_gate_blockers(context)
        probabilities = dict(context.calibrated_probabilities)
        if set(probabilities) != {"DOWN", "NEITHER", "UP"}:
            raise ValueError("Signal policy requires DOWN, NEITHER, and UP probabilities")
        if abs(sum(probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("Calibrated probabilities must sum to one")

        direction = SignalDirection.WAIT
        expected_atr: float | None = None
        risk_levels: RiskLevels | None = None
        if not blockers:
            candidate = (
                SignalDirection.BUY
                if probabilities["UP"] >= probabilities["DOWN"]
                else SignalDirection.SELL
            )
            directional = probabilities["UP"] if candidate is SignalDirection.BUY else probabilities["DOWN"]
            opposing = probabilities["DOWN"] if candidate is SignalDirection.BUY else probabilities["UP"]
            margin = directional - max(opposing, probabilities["NEITHER"])
            expected_atr = directional * 1.0 - opposing * float(resolved.stop_atr)
            if directional < resolved.activation_probability:
                blockers.append("PROBABILITY_THRESHOLD_NOT_MET")
            if margin < resolved.minimum_directional_margin:
                blockers.append("CLASS_MARGIN_THRESHOLD_NOT_MET")
            if probabilities["NEITHER"] > resolved.maximum_neither_probability:
                blockers.append("NEITHER_PROBABILITY_TOO_HIGH")
            if expected_atr < resolved.minimum_expected_atr:
                blockers.append("EXPECTED_VALUE_GATE_FAILED")
            if context.current_active_direction not in (None, candidate):
                if directional < resolved.reversal_probability or margin < resolved.reversal_margin:
                    blockers.append("HYSTERESIS_REVERSAL_THRESHOLD_NOT_MET")
                blockers.append("CONFLICTING_ACTIVE_SIGNAL_MUST_INVALIDATE_FIRST")
            if not blockers:
                assert context.reference_close is not None and context.atr is not None
                risk_levels = _risk_levels(candidate, context.reference_close, context.atr, resolved)
                if risk_levels.target1_reward_risk < resolved.minimum_target1_reward_risk:
                    blockers.append("RISK_REWARD_GATE_FAILED")
                else:
                    direction = candidate

        if blockers:
            direction = SignalDirection.WAIT
            expected_atr = None
            risk_levels = None
        expires_at = context.decision_time + timedelta(minutes=5 * resolved.expiry_bars)
        identity = json.dumps(
            {
                "prediction_id": context.prediction_id,
                "calibration_id": context.calibration_id,
                "snapshot_id": context.snapshot_id,
                "signal_policy_version": SIGNAL_POLICY_VERSION,
                "risk_policy_version": RISK_POLICY_VERSION,
                "direction": direction.value,
                "blockers": blockers,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        signal_id = str(uuid5(NAMESPACE_URL, f"signal:{hashlib.sha256(identity.encode()).hexdigest()}"))
        return SignalDecision(
            schema_version=1,
            signal_id=signal_id,
            prediction_id=context.prediction_id,
            calibration_id=context.calibration_id,
            snapshot_id=context.snapshot_id,
            instrument_id=context.instrument_id,
            decision_time=context.decision_time,
            created_at=context.decision_time,
            expires_at=expires_at,
            direction=direction,
            lifecycle_status=(
                SignalLifecycleStatus.ACTIVE
                if direction is not SignalDirection.WAIT
                else SignalLifecycleStatus.NO_SIGNAL
            ),
            probabilities=(
                context.calibrated_probabilities
                if context.calibration_release_passed
                and context.probability_bin_supported
                and context.data_status is ConnectionState.LIVE
                and context.snapshot_synced
                else None
            ),
            expected_atr=expected_atr,
            risk_levels=risk_levels,
            blockers=tuple(blockers),
            signal_policy_version=SIGNAL_POLICY_VERSION,
            risk_policy_version=RISK_POLICY_VERSION,
            input_revision_checksum=context.input_revision_checksum,
        )


def _hard_gate_blockers(context: SignalContext) -> list[str]:
    blockers = []
    if context.data_status is not ConnectionState.LIVE:
        blockers.append(f"DATA_NOT_LIVE_{context.data_status.value}")
    if not context.snapshot_synced:
        blockers.append("SNAPSHOT_REVISION_MISMATCH")
    if not context.finalized_primary:
        blockers.append("PRIMARY_CANDLE_NOT_FINALIZED")
    if not context.finalized_15m_context:
        blockers.append("FINALIZED_15M_CONTEXT_MISSING")
    if not context.finalized_1h_context:
        blockers.append("FINALIZED_1H_CONTEXT_MISSING")
    if not context.feature_ready:
        blockers.append("FEATURE_SNAPSHOT_NOT_READY")
    if not context.calibration_release_passed:
        blockers.append("CALIBRATION_RELEASE_GATE_NOT_PASSED")
    if not context.probability_bin_supported:
        blockers.append("PROBABILITY_BIN_NOT_SUPPORTED")
    if not context.event_risk_clear:
        blockers.append("EVENT_RISK_GATE_ACTIVE")
    if context.reference_close is None:
        blockers.append("REFERENCE_PRICE_UNAVAILABLE")
    if context.atr is None or context.atr <= 0:
        blockers.append("ATR_UNAVAILABLE")
    return blockers


def _risk_levels(
    direction: SignalDirection,
    close: Decimal,
    atr: Decimal,
    config: PolicyConfig,
) -> RiskLevels:
    sign = Decimal("1") if direction is SignalDirection.BUY else Decimal("-1")
    entry_low = close - config.entry_half_width_atr * atr
    entry_high = close + config.entry_half_width_atr * atr
    stop = close - sign * config.stop_atr * atr
    target1 = close + sign * config.target1_atr * atr
    target2 = close + sign * config.target2_atr * atr
    target3 = close + sign * config.target3_atr * atr
    return RiskLevels(
        entry_low=min(entry_low, entry_high),
        entry_high=max(entry_low, entry_high),
        stop=stop,
        invalidation=stop,
        target1=target1,
        target2=target2,
        target3=target3,
        target1_reward_risk=float(config.target1_atr / config.stop_atr),
    )
