"""Immutable contracts for research-only price-action decision support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class PriceActionBias(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNAVAILABLE = "UNAVAILABLE"


class SetupState(StrEnum):
    BUY_TRIGGER = "BUY_TRIGGER"
    SELL_TRIGGER = "SELL_TRIGGER"
    BULLISH_WATCH = "BULLISH_WATCH"
    BEARISH_WATCH = "BEARISH_WATCH"
    WAIT = "WAIT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PriceActionLevel:
    price: Decimal
    kind: str
    strength: int
    touches: int

    def to_contract(self) -> dict[str, object]:
        return {
            "price": _decimal(self.price),
            "kind": self.kind,
            "strength": self.strength,
            "touches": self.touches,
        }


@dataclass(frozen=True, slots=True)
class ConditionalTradePlan:
    direction: str
    trigger: Decimal
    entry_low: Decimal
    entry_high: Decimal
    stop: Decimal
    invalidation: Decimal
    target1: Decimal
    target2: Decimal
    target3: Decimal
    risk_points: Decimal
    target1_reward_risk: float
    target2_reward_risk: float
    target3_reward_risk: float
    expiry_bars: int
    blockers: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "trigger": _decimal(self.trigger),
            "entry_low": _decimal(self.entry_low),
            "entry_high": _decimal(self.entry_high),
            "stop": _decimal(self.stop),
            "invalidation": _decimal(self.invalidation),
            "target1": _decimal(self.target1),
            "target2": _decimal(self.target2),
            "target3": _decimal(self.target3),
            "risk_points": _decimal(self.risk_points),
            "target1_reward_risk": self.target1_reward_risk,
            "target2_reward_risk": self.target2_reward_risk,
            "target3_reward_risk": self.target3_reward_risk,
            "expiry_bars": self.expiry_bars,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class PriceActionAnalysis:
    snapshot_id: str
    candle_revision_checksum: str
    instrument_id: str
    decision_time: datetime
    generated_at: datetime
    version: str
    bias: PriceActionBias
    setup: SetupState
    confluence_score: int
    evidence_grade: str
    structure_5m: str
    trend_5m: str
    trend_15m: str
    trend_1h: str
    volatility_regime: str
    patterns: tuple[str, ...]
    support_levels: tuple[PriceActionLevel, ...]
    resistance_levels: tuple[PriceActionLevel, ...]
    reasons: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    trade_plan: ConditionalTradePlan | None
    blockers: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "snapshot_id": self.snapshot_id,
            "candle_revision_checksum": self.candle_revision_checksum,
            "instrument_id": self.instrument_id,
            "decision_time": _time(self.decision_time),
            "generated_at": _time(self.generated_at),
            "version": self.version,
            "bias": self.bias.value,
            "setup": self.setup.value,
            "confluence_score": self.confluence_score,
            "evidence_grade": self.evidence_grade,
            "structure_5m": self.structure_5m,
            "trend_5m": self.trend_5m,
            "trend_15m": self.trend_15m,
            "trend_1h": self.trend_1h,
            "volatility_regime": self.volatility_regime,
            "patterns": list(self.patterns),
            "support_levels": [item.to_contract() for item in self.support_levels],
            "resistance_levels": [item.to_contract() for item in self.resistance_levels],
            "reasons": list(self.reasons),
            "contradictory_evidence": list(self.contradictory_evidence),
            "trade_plan": self.trade_plan.to_contract() if self.trade_plan else None,
            "blockers": list(self.blockers),
            "research_only": True,
            "official_signal": False,
            "calibrated_probability": None,
            "automatic_execution": False,
        }


def _decimal(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
