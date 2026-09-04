"""Chronological policy replay over separately calibrated OOS predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from nifty_terminal.calibration.models import CalibrationReport
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.signals.models import PolicyConfig, SignalContext, SignalDecision, SignalDirection
from nifty_terminal.signals.policy import SignalPolicy


@dataclass(frozen=True, slots=True)
class SignalReplayInput:
    prediction_id: str
    snapshot_id: str
    instrument_id: str
    decision_time: datetime
    input_revision_checksum: str
    reference_close: Decimal
    atr: Decimal


@dataclass(frozen=True, slots=True)
class Step7ResearchReport:
    schema_version: int
    created_at: datetime
    calibration: CalibrationReport
    policy_config: PolicyConfig
    decisions: tuple[SignalDecision, ...]

    def to_contract(self) -> dict[str, object]:
        support = {direction.value: 0 for direction in SignalDirection}
        for decision in self.decisions:
            support[decision.direction.value] += 1
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "calibration": self.calibration.to_contract(),
            "policy": {
                "config": self.policy_config.to_contract(),
                "decision_count": len(self.decisions),
                "direction_support": support,
                "wait_first": True,
                "automatic_execution": False,
            },
            "signals": [item.to_contract() for item in self.decisions],
        }


def replay_signal_policy(
    *,
    calibration: CalibrationReport,
    inputs: tuple[SignalReplayInput, ...],
    config: PolicyConfig | None = None,
) -> tuple[SignalDecision, ...]:
    resolved = config or PolicyConfig()
    source = {item.prediction_id: item for item in inputs}
    supported_bins = set(calibration.supported_confidence_bins)
    active_direction: SignalDirection | None = None
    active_expires_at: datetime | None = None
    decisions = []
    for prediction in calibration.predictions:
        item = source.get(prediction.source_prediction_id)
        if item is None:
            raise ValueError(f"Missing signal replay input for {prediction.source_prediction_id}")
        if active_expires_at is not None and item.decision_time >= active_expires_at:
            active_direction, active_expires_at = None, None
        probability_bin = _confidence_bin(max(dict(prediction.calibrated_probabilities).values()))
        decision = SignalPolicy().evaluate(
            SignalContext(
                prediction_id=item.prediction_id,
                calibration_id=calibration.calibration_id,
                snapshot_id=item.snapshot_id,
                instrument_id=item.instrument_id,
                decision_time=item.decision_time,
                data_as_of=item.decision_time,
                input_revision_checksum=item.input_revision_checksum,
                calibrated_probabilities=prediction.calibrated_probabilities,
                reference_close=item.reference_close,
                atr=item.atr,
                data_status=ConnectionState.LIVE,
                snapshot_synced=True,
                finalized_primary=True,
                finalized_15m_context=True,
                finalized_1h_context=True,
                feature_ready=True,
                calibration_release_passed=calibration.release_gate_passed,
                probability_bin_supported=probability_bin in supported_bins,
                event_risk_clear=True,
                current_active_direction=active_direction,
            ),
            resolved,
        )
        decisions.append(decision)
        if decision.direction is not SignalDirection.WAIT:
            active_direction, active_expires_at = decision.direction, decision.expires_at
    return tuple(decisions)


def _confidence_bin(confidence: float) -> str:
    index = min(int(confidence * 10), 9)
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"
