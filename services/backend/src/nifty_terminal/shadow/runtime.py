"""Automatic finalized-candle shadow inference with immutable forward outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.candle import Candle, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.features.snapshot import SnapshotFeatureAssembler
from nifty_terminal.price_action.engine import PriceActionEngine
from nifty_terminal.price_action.replay import (
    SCALE_STATIC_POLICY,
    replay_price_action_contract,
)
from nifty_terminal.shadow.artifacts import LoadedShadowArtifacts, load_shadow_artifacts
from nifty_terminal.shadow.ledger import SQLiteShadowLedger
from nifty_terminal.snapshots.models import DataMode


ALLOWED_SNAPSHOT_BLOCKERS = frozenset({"LIVE_SIGNAL_KILL_SWITCH_ACTIVE"})


@dataclass(frozen=True, slots=True)
class ShadowRuntimeStatus:
    enabled: bool
    runtime_mode: str
    healthy: bool
    reason: str
    manifest_sha256: str
    prediction_count: int
    assessment_count: int
    pending_assessment_count: int
    latest_prediction_decision_time: str | None
    last_error_type: str | None

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "runtime_mode": self.runtime_mode,
            "healthy": self.healthy,
            "reason": self.reason,
            "manifest_sha256": self.manifest_sha256,
            "prediction_count": self.prediction_count,
            "assessment_count": self.assessment_count,
            "pending_assessment_count": self.pending_assessment_count,
            "latest_prediction_decision_time": self.latest_prediction_decision_time,
            "last_error_type": self.last_error_type,
            "shadow_only": True,
            "precise_probability_display_allowed": False,
            "official_signal_available": False,
            "automatic_trading_enabled": False,
        }


class ShadowRuntime:
    def __init__(
        self,
        *,
        artifacts: LoadedShadowArtifacts,
        ledger: SQLiteShadowLedger,
    ) -> None:
        self._artifacts = artifacts
        self._ledger = ledger
        self._features = SnapshotFeatureAssembler(NseSessionCalendar())
        self._price_action = PriceActionEngine()
        self._lock = RLock()
        self._healthy = True
        self._reason = "WAITING_FOR_FINALIZED_LIVE_SNAPSHOT"
        self._last_error_type: str | None = None

    @property
    def status(self) -> ShadowRuntimeStatus:
        counts = self._ledger.status()
        return ShadowRuntimeStatus(
            enabled=True,
            runtime_mode=str(self._artifacts.policy["runtime_mode"]),
            healthy=self._healthy,
            reason=self._reason,
            manifest_sha256=self._artifacts.manifest_sha256,
            prediction_count=int(counts["prediction_count"]),
            assessment_count=int(counts["assessment_count"]),
            pending_assessment_count=int(counts["pending_assessment_count"]),
            latest_prediction_decision_time=counts["latest_prediction_decision_time"],
            last_error_type=self._last_error_type,
        )

    def process(
        self,
        *,
        view: MarketStateView,
        minute_candles: tuple[Candle, ...],
        observed_at: datetime,
    ) -> None:
        with self._lock:
            try:
                self._assess_due(minute_candles=minute_candles, observed_at=observed_at)
                self._predict(view=view, observed_at=observed_at)
                self._healthy = True
                self._last_error_type = None
            except Exception as error:
                self._healthy = False
                self._reason = f"SHADOW_RUNTIME_{type(error).__name__.upper()}"
                self._last_error_type = type(error).__name__

    def _predict(self, *, view: MarketStateView, observed_at: datetime) -> None:
        snapshot = view.snapshot
        blockers = [item for item in snapshot.blockers if item not in ALLOWED_SNAPSHOT_BLOCKERS]
        if snapshot.data_mode is not DataMode.LIVE:
            blockers.append("SHADOW_REQUIRES_LIVE_DATA_MODE")
        if snapshot.data_status is not ConnectionState.LIVE:
            blockers.append(f"SHADOW_DATA_STATUS_{snapshot.data_status.value}")
        if blockers:
            self._reason = blockers[0]
            return
        features = self._features.assemble(view)
        if not features.is_ready:
            self._reason = features.blockers[0] if features.blockers else "FEATURES_NOT_READY"
            return
        candle_by_id = {item.candle_id: item for item in view.finalized_candles}
        primary = candle_by_id[snapshot.primary_candle_id]
        atr_value = dict(features.values).get("primary_5m__atr_14")
        if atr_value is None:
            self._reason = "SHADOW_ATR_UNAVAILABLE"
            return
        atr = Decimal(str(atr_value))
        raw, calibrated = self._artifacts.predict(features)
        direction = self._artifacts.shadow_direction(raw, calibrated)
        price_action = self._price_action.analyze(view, generated_at=observed_at)
        prediction_id = str(
            uuid5(
                NAMESPACE_URL,
                "shadow-prediction:"
                f"{snapshot.snapshot_id}:{snapshot.candle_revision_checksum}:"
                f"{self._artifacts.model['sha256']}:{self._artifacts.policy['sha256']}",
            )
        )
        payload = {
            "schema_version": 1,
            "prediction_id": prediction_id,
            "snapshot_id": snapshot.snapshot_id,
            "instrument_id": snapshot.instrument_id,
            "decision_time": _time(snapshot.decision_time),
            "data_as_of": _time(snapshot.data_as_of),
            "generated_at": _time(observed_at),
            "outcome_due_at": _time(snapshot.decision_time + timedelta(minutes=60)),
            "input_revision_checksum": snapshot.candle_revision_checksum,
            "feature_snapshot_id": features.feature_snapshot_id,
            "model_artifact_sha256": self._artifacts.model["sha256"],
            "policy_artifact_sha256": self._artifacts.policy["sha256"],
            "reference_close": format(primary.close, "f"),
            "atr_at_decision": format(atr, "f"),
            "up_barrier": format(primary.close + Decimal("1.5") * atr, "f"),
            "down_barrier": format(primary.close - Decimal("1.5") * atr, "f"),
            "raw_probabilities": _probabilities(raw),
            "calibrated_probabilities": _probabilities(calibrated),
            "shadow_candidate_direction": direction,
            "price_action_analysis": price_action.to_contract(),
            "official_signal": None,
            "precise_probability_display_allowed": False,
            "shadow_only": True,
            "automatic_trading_enabled": False,
        }
        inserted = self._ledger.append_prediction(payload)
        self._reason = (
            "SHADOW_PREDICTION_RECORDED" if inserted else "SHADOW_PREDICTION_ALREADY_RECORDED"
        )

    def _assess_due(
        self,
        *,
        minute_candles: tuple[Candle, ...],
        observed_at: datetime,
    ) -> None:
        by_open = {item.opens_at: item for item in minute_candles if item.timeframe is Timeframe.M1}
        for prediction in self._ledger.pending(due_at_or_before=observed_at):
            decision_time = datetime.fromisoformat(str(prediction["decision_time"]).replace("Z", "+00:00"))
            window = tuple(by_open.get(decision_time + timedelta(minutes=index)) for index in range(60))
            if any(item is None for item in window):
                outcome, first_touch_at, quality = "UNASSESSABLE", None, "MISSING_MINUTE_WINDOW"
                price_action_path = None
            else:
                complete_window = tuple(item for item in window if item is not None)
                outcome, first_touch_at = _first_touch_outcome(
                    complete_window,
                    up=Decimal(str(prediction["up_barrier"])),
                    down=Decimal(str(prediction["down_barrier"])),
                )
                quality = "ASSESSED"
                price_action_path = _assess_price_action_path(prediction, complete_window)
            assessment_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"shadow-assessment:{prediction['prediction_id']}:1p5-atr-60m-v1",
                )
            )
            self._ledger.append_assessment(
                {
                    "schema_version": 1,
                    "assessment_id": assessment_id,
                    "prediction_id": prediction["prediction_id"],
                    "snapshot_id": prediction["snapshot_id"],
                    "decision_time": prediction["decision_time"],
                    "assessed_at": _time(observed_at),
                    "actual_outcome": outcome,
                    "first_touch_at": _time(first_touch_at) if first_touch_at else None,
                    "assessment_quality": quality,
                    "price_action_path": price_action_path,
                    "label_definition": "SYMMETRIC_1.5_ATR_FIRST_TOUCH_60M",
                    "original_prediction_immutable": True,
                }
            )


def build_shadow_runtime(*, manifest_path: Path, ledger_path: Path) -> ShadowRuntime:
    return ShadowRuntime(
        artifacts=load_shadow_artifacts(manifest_path),
        ledger=SQLiteShadowLedger(ledger_path),
    )


def _first_touch_outcome(
    candles: tuple[Candle, ...], *, up: Decimal, down: Decimal
) -> tuple[str, datetime | None]:
    for candle in candles:
        up_hit = candle.high >= up
        down_hit = candle.low <= down
        if up_hit and down_hit:
            return "AMBIGUOUS", candle.closes_at
        if up_hit:
            return "UP", candle.closes_at
        if down_hit:
            return "DOWN", candle.closes_at
    return "NEITHER", None


def _probabilities(values) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(("DOWN", "NEITHER", "UP"))}


def _assess_price_action_path(
    prediction: dict[str, object],
    candles: tuple[Candle, ...],
) -> dict[str, object] | None:
    """Conservatively replay a frozen conditional trigger over the next hour."""
    analysis = prediction.get("price_action_analysis")
    if not isinstance(analysis, dict):
        return None
    plan = analysis.get("trade_plan")
    if not isinstance(plan, dict) or plan.get("direction") not in {"BUY", "SELL"}:
        return None
    result = replay_price_action_contract(
        plan=plan,
        minute_candles=candles,
        policy=SCALE_STATIC_POLICY,
    )
    return {
        "status": result.status,
        "direction": result.direction,
        "entered_at": _time(result.entered_at) if result.entered_at else None,
        "entry_price": (
            format(result.entry_price, "f") if result.entry_price is not None else None
        ),
        "maximum_target_reached": result.maximum_target_reached,
        "stop_hit": result.stop_hit,
        "stop_first_within_bar": True,
        "execution_policy": SCALE_STATIC_POLICY.name,
        "net_points_after_slippage": format(result.net_points, "f"),
        "r_multiple_after_slippage": format(result.r_multiple, "f"),
    }


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
