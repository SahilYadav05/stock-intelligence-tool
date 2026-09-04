"""Append-only lifecycle assessments; original signals are never mutated."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.signals.models import (
    SignalDecision,
    SignalDirection,
    SignalEventType,
    SignalLifecycleEvent,
    SignalLifecycleStatus,
)


def assess_signal(
    signal: SignalDecision,
    *,
    observed_at: datetime,
    high: Decimal,
    low: Decimal,
) -> SignalLifecycleEvent:
    if signal.direction is SignalDirection.WAIT or signal.risk_levels is None:
        return _event(
            signal,
            SignalEventType.MAINTAINED,
            SignalLifecycleStatus.NO_SIGNAL,
            observed_at,
            None,
            "WAIT_HAS_NO_ACTIVE_RISK_LEVELS",
        )
    levels = signal.risk_levels
    if observed_at >= signal.expires_at:
        return _event(
            signal,
            SignalEventType.EXPIRED,
            SignalLifecycleStatus.EXPIRED,
            observed_at,
            None,
            "SIGNAL_HORIZON_ELAPSED",
        )
    if signal.direction is SignalDirection.BUY:
        stop_hit, target_hit = low <= levels.stop, high >= levels.target1
        stop_price, target_price = levels.stop, levels.target1
    else:
        stop_hit, target_hit = high >= levels.stop, low <= levels.target1
        stop_price, target_price = levels.stop, levels.target1
    if stop_hit and target_hit:
        return _event(
            signal,
            SignalEventType.INVALIDATED,
            SignalLifecycleStatus.INVALIDATED,
            observed_at,
            None,
            "AMBIGUOUS_INTRABAR_STOP_AND_TARGET_ORDER",
        )
    if stop_hit:
        return _event(
            signal,
            SignalEventType.STOP_HIT,
            SignalLifecycleStatus.STOP_HIT,
            observed_at,
            stop_price,
            "STOP_LEVEL_TOUCHED",
        )
    if target_hit:
        return _event(
            signal,
            SignalEventType.TARGET_HIT,
            SignalLifecycleStatus.TARGET_HIT,
            observed_at,
            target_price,
            "TARGET_1_TOUCHED",
        )
    return _event(
        signal,
        SignalEventType.MAINTAINED,
        SignalLifecycleStatus.ACTIVE,
        observed_at,
        None,
        "ACTIVE_LEVELS_NOT_TOUCHED",
    )


def _event(
    signal: SignalDecision,
    event_type: SignalEventType,
    status: SignalLifecycleStatus,
    occurred_at: datetime,
    price: Decimal | None,
    reason: str,
) -> SignalLifecycleEvent:
    event_identity = (
        f"signal-event:{signal.signal_id}:{occurred_at.isoformat()}:{event_type.value}:{reason}"
    )
    event_id = str(uuid5(NAMESPACE_URL, event_identity))
    return SignalLifecycleEvent(event_id, signal.signal_id, event_type, status, occurred_at, price, reason)
