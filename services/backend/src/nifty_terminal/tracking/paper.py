"""Conservative paper-only lifecycle rules based on finalized candles."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.dashboard.models import AnalysisView
from nifty_terminal.signals.models import SignalDirection, SignalLifecycleStatus
from nifty_terminal.tracking.models import (
    PaperTrade,
    PaperTradeEvent,
    PaperTradeEventType,
    PaperTradeStatus,
)


def create_paper_trade(analysis: AnalysisView) -> PaperTrade | None:
    signal = analysis.signal
    if (
        signal.direction is SignalDirection.WAIT
        or signal.lifecycle_status is not SignalLifecycleStatus.ACTIVE
        or signal.risk_levels is None
    ):
        return None
    levels = signal.risk_levels
    identity = f"paper-trade:{signal.signal_id}:{signal.input_revision_checksum}"
    return PaperTrade(
        paper_trade_id=str(uuid5(NAMESPACE_URL, identity)),
        signal_id=signal.signal_id,
        prediction_id=signal.prediction_id,
        snapshot_id=signal.snapshot_id,
        instrument_id=signal.instrument_id,
        created_at=signal.created_at,
        expires_at=signal.expires_at,
        direction=signal.direction,
        entry_low=levels.entry_low,
        entry_high=levels.entry_high,
        stop=levels.stop,
        target1=levels.target1,
        target2=levels.target2,
        target3=levels.target3,
        model_version=analysis.model_version,
        calibration_version=analysis.calibration_version,
        signal_policy_version=signal.signal_policy_version,
        input_revision_checksum=signal.input_revision_checksum,
    )


def assess_paper_trade(
    trade: PaperTrade,
    *,
    observed_at: datetime,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    opened_price: Decimal | None,
) -> PaperTradeEvent | None:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    if low > high or close < low or close > high:
        raise ValueError("paper candle observation is invalid")
    if opened_price is None:
        if observed_at >= trade.expires_at:
            return _event(
                trade,
                PaperTradeEventType.EXPIRED,
                PaperTradeStatus.EXPIRED,
                observed_at,
                None,
                None,
                "ENTRY_NOT_TOUCHED_BEFORE_EXPIRY",
            )
        entry_touched = high >= trade.entry_low and low <= trade.entry_high
        if not entry_touched:
            return None
        stop_touched, target_touched = _terminal_touches(trade, high=high, low=low)
        if stop_touched or target_touched:
            return _event(
                trade,
                PaperTradeEventType.INVALIDATED,
                PaperTradeStatus.INVALIDATED,
                observed_at,
                None,
                None,
                "AMBIGUOUS_ENTRY_AND_EXIT_ORDER_IN_SAME_CANDLE",
            )
        conservative_entry = (
            trade.entry_high if trade.direction is SignalDirection.BUY else trade.entry_low
        )
        return _event(
            trade,
            PaperTradeEventType.OPENED,
            PaperTradeStatus.OPEN,
            observed_at,
            conservative_entry,
            None,
            "ENTRY_ZONE_TOUCHED_CONSERVATIVE_FILL",
        )

    stop_touched, target_touched = _terminal_touches(trade, high=high, low=low)
    if stop_touched and target_touched:
        return _event(
            trade,
            PaperTradeEventType.INVALIDATED,
            PaperTradeStatus.INVALIDATED,
            observed_at,
            None,
            None,
            "AMBIGUOUS_INTRABAR_STOP_AND_TARGET_ORDER",
        )
    if stop_touched:
        return _terminal_event(
            trade,
            PaperTradeEventType.STOP_HIT,
            PaperTradeStatus.STOP_HIT,
            observed_at,
            trade.stop,
            opened_price,
            "STOP_LEVEL_TOUCHED",
        )
    if target_touched:
        return _terminal_event(
            trade,
            PaperTradeEventType.TARGET_1_HIT,
            PaperTradeStatus.TARGET_1_HIT,
            observed_at,
            trade.target1,
            opened_price,
            "TARGET_1_LEVEL_TOUCHED",
        )
    if observed_at >= trade.expires_at:
        return _terminal_event(
            trade,
            PaperTradeEventType.EXPIRED,
            PaperTradeStatus.EXPIRED,
            observed_at,
            close,
            opened_price,
            "SIGNAL_HORIZON_ELAPSED_AT_FINALIZED_CLOSE",
        )
    return None


def _terminal_touches(trade: PaperTrade, *, high: Decimal, low: Decimal) -> tuple[bool, bool]:
    if trade.direction is SignalDirection.BUY:
        return low <= trade.stop, high >= trade.target1
    return high >= trade.stop, low <= trade.target1


def _terminal_event(
    trade: PaperTrade,
    event_type: PaperTradeEventType,
    status: PaperTradeStatus,
    occurred_at: datetime,
    exit_price: Decimal,
    opened_price: Decimal,
    reason: str,
) -> PaperTradeEvent:
    pnl = (
        exit_price - opened_price
        if trade.direction is SignalDirection.BUY
        else opened_price - exit_price
    )
    return _event(trade, event_type, status, occurred_at, exit_price, pnl, reason)


def _event(
    trade: PaperTrade,
    event_type: PaperTradeEventType,
    status: PaperTradeStatus,
    occurred_at: datetime,
    price: Decimal | None,
    pnl: Decimal | None,
    reason: str,
) -> PaperTradeEvent:
    identity = (
        f"paper-event:{trade.paper_trade_id}:{occurred_at.isoformat()}:"
        f"{event_type.value}:{reason}"
    )
    return PaperTradeEvent(
        event_id=str(uuid5(NAMESPACE_URL, identity)),
        paper_trade_id=trade.paper_trade_id,
        signal_id=trade.signal_id,
        event_type=event_type,
        status=status,
        occurred_at=occurred_at,
        observed_price=price,
        pnl_points=pnl,
        reason=reason,
    )
