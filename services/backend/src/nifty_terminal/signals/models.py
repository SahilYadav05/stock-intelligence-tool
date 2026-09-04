"""Immutable signal policy inputs, decisions, risk levels, and lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from nifty_terminal.domain.enums import ConnectionState


class SignalDirection(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class SignalLifecycleStatus(StrEnum):
    NO_SIGNAL = "NO_SIGNAL"
    SETUP_DETECTED = "SETUP_DETECTED"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    ACTIVE = "ACTIVE"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class SignalEventType(StrEnum):
    CREATED = "CREATED"
    ACTIVATED = "ACTIVATED"
    MAINTAINED = "MAINTAINED"
    UPGRADED = "UPGRADED"
    DOWNGRADED = "DOWNGRADED"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    activation_probability: float = 0.60
    minimum_directional_margin: float = 0.15
    maximum_neither_probability: float = 0.45
    minimum_expected_atr: float = 0.15
    entry_half_width_atr: Decimal = Decimal("0.10")
    stop_atr: Decimal = Decimal("0.75")
    target1_atr: Decimal = Decimal("1.00")
    target2_atr: Decimal = Decimal("1.50")
    target3_atr: Decimal = Decimal("2.00")
    minimum_target1_reward_risk: float = 1.25
    reversal_probability: float = 0.72
    reversal_margin: float = 0.25
    expiry_bars: int = 12

    def __post_init__(self) -> None:
        probability_values = (
            self.activation_probability,
            self.minimum_directional_margin,
            self.maximum_neither_probability,
            self.minimum_expected_atr,
            self.reversal_probability,
            self.reversal_margin,
        )
        if any(not 0.0 <= value <= 1.0 for value in probability_values):
            raise ValueError("Policy probabilities and margins must be in [0, 1]")
        if self.reversal_probability < self.activation_probability:
            raise ValueError("Reversal threshold must not be weaker than activation")
        if min(self.entry_half_width_atr, self.stop_atr, self.target1_atr) <= 0:
            raise ValueError("Risk distances must be positive")
        if not self.target1_atr < self.target2_atr < self.target3_atr:
            raise ValueError("Targets must be strictly ordered")
        if self.expiry_bars < 1:
            raise ValueError("Signal expiry must be positive")

    def to_contract(self) -> dict[str, object]:
        return {
            "activation_probability": self.activation_probability,
            "minimum_directional_margin": self.minimum_directional_margin,
            "maximum_neither_probability": self.maximum_neither_probability,
            "minimum_expected_atr": self.minimum_expected_atr,
            "entry_half_width_atr": format(self.entry_half_width_atr, "f"),
            "stop_atr": format(self.stop_atr, "f"),
            "target1_atr": format(self.target1_atr, "f"),
            "target2_atr": format(self.target2_atr, "f"),
            "target3_atr": format(self.target3_atr, "f"),
            "minimum_target1_reward_risk": self.minimum_target1_reward_risk,
            "reversal_probability": self.reversal_probability,
            "reversal_margin": self.reversal_margin,
            "expiry_bars": self.expiry_bars,
            "status": "RESEARCH_DEFAULTS_REQUIRE_POLICY_VALIDATION",
        }


@dataclass(frozen=True, slots=True)
class SignalContext:
    prediction_id: str
    calibration_id: str
    snapshot_id: str
    instrument_id: str
    decision_time: datetime
    data_as_of: datetime
    input_revision_checksum: str
    calibrated_probabilities: tuple[tuple[str, float], ...]
    reference_close: Decimal | None
    atr: Decimal | None
    data_status: ConnectionState
    snapshot_synced: bool
    finalized_primary: bool
    finalized_15m_context: bool
    finalized_1h_context: bool
    feature_ready: bool
    calibration_release_passed: bool
    probability_bin_supported: bool
    event_risk_clear: bool
    current_active_direction: SignalDirection | None = None


@dataclass(frozen=True, slots=True)
class RiskLevels:
    entry_low: Decimal
    entry_high: Decimal
    stop: Decimal
    invalidation: Decimal
    target1: Decimal
    target2: Decimal
    target3: Decimal
    target1_reward_risk: float

    def to_contract(self) -> dict[str, object]:
        return {
            "entry_low": _decimal(self.entry_low),
            "entry_high": _decimal(self.entry_high),
            "stop": _decimal(self.stop),
            "invalidation": _decimal(self.invalidation),
            "target1": _decimal(self.target1),
            "target2": _decimal(self.target2),
            "target3": _decimal(self.target3),
            "target1_reward_risk": self.target1_reward_risk,
        }


@dataclass(frozen=True, slots=True)
class SignalDecision:
    schema_version: int
    signal_id: str
    prediction_id: str
    calibration_id: str
    snapshot_id: str
    instrument_id: str
    decision_time: datetime
    created_at: datetime
    expires_at: datetime
    direction: SignalDirection
    lifecycle_status: SignalLifecycleStatus
    probabilities: tuple[tuple[str, float], ...] | None
    expected_atr: float | None
    risk_levels: RiskLevels | None
    blockers: tuple[str, ...]
    signal_policy_version: str
    risk_policy_version: str
    input_revision_checksum: str

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signal_id": self.signal_id,
            "prediction_id": self.prediction_id,
            "calibration_id": self.calibration_id,
            "snapshot_id": self.snapshot_id,
            "instrument_id": self.instrument_id,
            "decision_time": _time(self.decision_time),
            "created_at": _time(self.created_at),
            "expires_at": _time(self.expires_at),
            "direction": self.direction.value,
            "lifecycle_status": self.lifecycle_status.value,
            "probabilities": dict(self.probabilities) if self.probabilities else None,
            "expected_atr": self.expected_atr,
            "risk_levels": self.risk_levels.to_contract() if self.risk_levels else None,
            "blockers": list(self.blockers),
            "signal_policy_version": self.signal_policy_version,
            "risk_policy_version": self.risk_policy_version,
            "input_revision_checksum": self.input_revision_checksum,
            "automatic_execution": False,
        }


@dataclass(frozen=True, slots=True)
class SignalLifecycleEvent:
    event_id: str
    signal_id: str
    event_type: SignalEventType
    status: SignalLifecycleStatus
    occurred_at: datetime
    observed_price: Decimal | None
    reason: str

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event_id": self.event_id,
            "signal_id": self.signal_id,
            "event_type": self.event_type.value,
            "status": self.status.value,
            "occurred_at": _time(self.occurred_at),
            "observed_price": _decimal(self.observed_price),
            "reason": self.reason,
        }


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _decimal(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None
