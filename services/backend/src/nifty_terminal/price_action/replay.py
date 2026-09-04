"""Conservative replay of the exact conditional price-action trade levels.

The live price-action engine owns trigger, entry-zone, stop and T1/T2/T3
levels.  This module owns only the predeclared execution policy applied to
those immutable levels.  It is shared by historical research and shadow
assessment so displayed plans and measured plans cannot silently diverge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Sequence

from nifty_terminal.domain.candle import Candle
from nifty_terminal.price_action.models import ConditionalTradePlan


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class PriceActionExecutionPolicy:
    name: str
    target_allocations: tuple[Decimal, Decimal, Decimal]
    protect_after_target1: bool
    protect_after_target2: bool
    one_way_slippage_points: Decimal = Decimal("0.50")

    def __post_init__(self) -> None:
        if len(self.target_allocations) != 3:
            raise ValueError("Exactly three target allocations are required")
        if any(item < ZERO for item in self.target_allocations):
            raise ValueError("Target allocations cannot be negative")
        if sum(self.target_allocations, ZERO) != Decimal("1"):
            raise ValueError("Target allocations must sum to one")
        if self.one_way_slippage_points < ZERO:
            raise ValueError("Slippage cannot be negative")

    def to_contract(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target_allocations": [float(item) for item in self.target_allocations],
            "protect_after_target1": self.protect_after_target1,
            "protect_after_target2": self.protect_after_target2,
            "one_way_slippage_points": float(self.one_way_slippage_points),
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
            "gap_beyond_entry_zone": "NO_CHASE_NO_FILL",
            "protective_stop_changes_apply": "NEXT_MINUTE_ONLY",
        }


FULL_TARGET1_POLICY = PriceActionExecutionPolicy(
    name="FULL_TARGET1",
    target_allocations=(Decimal("1"), ZERO, ZERO),
    protect_after_target1=False,
    protect_after_target2=False,
)
SCALE_STATIC_POLICY = PriceActionExecutionPolicy(
    name="SCALE_50_30_20_STATIC_STOP",
    target_allocations=(Decimal("0.50"), Decimal("0.30"), Decimal("0.20")),
    protect_after_target1=False,
    protect_after_target2=False,
)
SCALE_PROTECTED_POLICY = PriceActionExecutionPolicy(
    name="SCALE_50_30_20_PROTECTED",
    target_allocations=(Decimal("0.50"), Decimal("0.30"), Decimal("0.20")),
    protect_after_target1=True,
    protect_after_target2=True,
)
EXECUTION_POLICY_CANDIDATES = (
    FULL_TARGET1_POLICY,
    SCALE_STATIC_POLICY,
    SCALE_PROTECTED_POLICY,
)


@dataclass(frozen=True, slots=True)
class PriceActionPathResult:
    status: str
    direction: str
    entered_at: datetime | None
    exited_at: datetime | None
    entry_price: Decimal | None
    exit_price: Decimal | None
    maximum_target_reached: int
    stop_hit: bool
    net_points: Decimal
    r_multiple: Decimal
    realized_allocations: tuple[Decimal, Decimal, Decimal]
    remaining_allocation: Decimal

    @property
    def entered(self) -> bool:
        return self.entered_at is not None

    @property
    def profitable(self) -> bool:
        return self.net_points > ZERO

    def to_contract(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("entered_at", "exited_at"):
            value = payload[key]
            payload[key] = value.isoformat().replace("+00:00", "Z") if value else None
        for key in (
            "entry_price",
            "exit_price",
            "net_points",
            "r_multiple",
            "remaining_allocation",
        ):
            value = payload[key]
            payload[key] = format(value, "f") if value is not None else None
        payload["realized_allocations"] = [
            float(item) for item in self.realized_allocations
        ]
        payload["same_minute_resolution"] = "STOP_FIRST_CONSERVATIVE"
        return payload


def replay_price_action_plan(
    *,
    plan: ConditionalTradePlan,
    minute_candles: Sequence[Candle],
    policy: PriceActionExecutionPolicy,
) -> PriceActionPathResult:
    """Replay a frozen plan without inferring unavailable intraminute order."""
    if plan.direction not in {"BUY", "SELL"}:
        raise ValueError("Price-action plan direction must be BUY or SELL")
    ordered = tuple(sorted(minute_candles, key=lambda item: item.opens_at))
    if not ordered:
        raise ValueError("At least one minute candle is required")
    sign = Decimal("1") if plan.direction == "BUY" else Decimal("-1")
    targets = (plan.target1, plan.target2, plan.target3)
    allocations = policy.target_allocations
    entry_price: Decimal | None = None
    entered_at: datetime | None = None
    active_stop = plan.stop
    remaining = Decimal("1")
    realized = [ZERO, ZERO, ZERO]
    proceeds = ZERO
    maximum_target = 0
    last_exit: Decimal | None = None
    last_exit_at: datetime | None = None

    for candle in ordered:
        if entered_at is None:
            crossed = (
                candle.high >= plan.trigger
                if plan.direction == "BUY"
                else candle.low <= plan.trigger
            )
            if not crossed:
                continue
            gap_outside_zone = (
                candle.open > plan.entry_high
                if plan.direction == "BUY"
                else candle.open < plan.entry_low
            )
            if gap_outside_zone:
                return _empty_result("MISSED_GAP_BEYOND_ENTRY_ZONE", plan.direction)
            raw_entry = (
                max(candle.open, plan.trigger)
                if plan.direction == "BUY"
                else min(candle.open, plan.trigger)
            )
            entry_price = raw_entry + sign * policy.one_way_slippage_points
            entered_at = candle.opens_at

        # The adverse path is always resolved first when one candle contains
        # both stop and target prices.  A stop moved after a target becomes
        # active only for the following minute.
        stop_hit = (
            candle.low <= active_stop
            if plan.direction == "BUY"
            else candle.high >= active_stop
        )
        if stop_hit:
            raw_exit = active_stop
            effective_exit = raw_exit - sign * policy.one_way_slippage_points
            proceeds += remaining * sign * (effective_exit - entry_price)
            last_exit = raw_exit
            last_exit_at = candle.closes_at
            return _result(
                status="STOPPED" if maximum_target == 0 else "STOPPED_AFTER_TARGET",
                plan=plan,
                entered_at=entered_at,
                exited_at=last_exit_at,
                entry_price=entry_price,
                exit_price=last_exit,
                maximum_target=maximum_target,
                stop_hit=True,
                net_points=proceeds,
                realized=realized,
                remaining=ZERO,
            )

        reached_before = maximum_target
        for target_index, target in enumerate(targets):
            if target_index < maximum_target or allocations[target_index] <= ZERO:
                continue
            touched = (
                candle.high >= target
                if plan.direction == "BUY"
                else candle.low <= target
            )
            if not touched:
                break
            allocation = min(allocations[target_index], remaining)
            if allocation > ZERO:
                effective_exit = target - sign * policy.one_way_slippage_points
                proceeds += allocation * sign * (effective_exit - entry_price)
                remaining -= allocation
                realized[target_index] += allocation
                last_exit = target
                last_exit_at = candle.closes_at
            maximum_target = target_index + 1

        if remaining <= ZERO:
            return _result(
                status=f"TARGET{maximum_target}_REACHED",
                plan=plan,
                entered_at=entered_at,
                exited_at=last_exit_at,
                entry_price=entry_price,
                exit_price=last_exit,
                maximum_target=maximum_target,
                stop_hit=False,
                net_points=proceeds,
                realized=realized,
                remaining=ZERO,
            )
        if maximum_target > reached_before:
            if maximum_target >= 2 and policy.protect_after_target2:
                active_stop = plan.target1
            elif maximum_target >= 1 and policy.protect_after_target1:
                active_stop = plan.trigger

    if entered_at is None or entry_price is None:
        return _empty_result("NOT_TRIGGERED", plan.direction)
    raw_exit = ordered[-1].close
    effective_exit = raw_exit - sign * policy.one_way_slippage_points
    proceeds += remaining * sign * (effective_exit - entry_price)
    return _result(
        status="EXPIRED",
        plan=plan,
        entered_at=entered_at,
        exited_at=ordered[-1].closes_at,
        entry_price=entry_price,
        exit_price=raw_exit,
        maximum_target=maximum_target,
        stop_hit=False,
        net_points=proceeds,
        realized=realized,
        remaining=ZERO,
    )


def replay_price_action_contract(
    *,
    plan: dict[str, object],
    minute_candles: Sequence[Candle],
    policy: PriceActionExecutionPolicy,
) -> PriceActionPathResult:
    """Replay a serialized plan emitted by ``ConditionalTradePlan.to_contract``."""
    direction = str(plan.get("direction", ""))
    trigger = Decimal(str(plan["trigger"]))
    stop = Decimal(str(plan["stop"]))
    risk = abs(trigger - stop)
    if risk <= ZERO:
        raise ValueError("Serialized price-action plan has non-positive risk")
    contract = ConditionalTradePlan(
        direction=direction,
        trigger=trigger,
        entry_low=Decimal(str(plan.get("entry_low", trigger))),
        entry_high=Decimal(str(plan.get("entry_high", trigger))),
        stop=stop,
        invalidation=Decimal(str(plan.get("invalidation", stop))),
        target1=Decimal(str(plan["target1"])),
        target2=Decimal(str(plan["target2"])),
        target3=Decimal(str(plan["target3"])),
        risk_points=Decimal(str(plan.get("risk_points", risk))),
        target1_reward_risk=float(plan.get("target1_reward_risk", 1.25)),
        target2_reward_risk=float(plan.get("target2_reward_risk", 2.0)),
        target3_reward_risk=float(plan.get("target3_reward_risk", 3.0)),
        expiry_bars=int(plan.get("expiry_bars", 12)),
        blockers=tuple(str(item) for item in plan.get("blockers", ())),
    )
    return replay_price_action_plan(
        plan=contract, minute_candles=minute_candles, policy=policy
    )


def _empty_result(status: str, direction: str) -> PriceActionPathResult:
    return PriceActionPathResult(
        status=status,
        direction=direction,
        entered_at=None,
        exited_at=None,
        entry_price=None,
        exit_price=None,
        maximum_target_reached=0,
        stop_hit=False,
        net_points=ZERO,
        r_multiple=ZERO,
        realized_allocations=(ZERO, ZERO, ZERO),
        remaining_allocation=Decimal("1"),
    )


def _result(
    *,
    status: str,
    plan: ConditionalTradePlan,
    entered_at: datetime,
    exited_at: datetime | None,
    entry_price: Decimal,
    exit_price: Decimal | None,
    maximum_target: int,
    stop_hit: bool,
    net_points: Decimal,
    realized: list[Decimal],
    remaining: Decimal,
) -> PriceActionPathResult:
    return PriceActionPathResult(
        status=status,
        direction=plan.direction,
        entered_at=entered_at,
        exited_at=exited_at,
        entry_price=entry_price,
        exit_price=exit_price,
        maximum_target_reached=maximum_target,
        stop_hit=stop_hit,
        net_points=net_points,
        r_multiple=net_points / plan.risk_points,
        realized_allocations=tuple(realized),  # type: ignore[arg-type]
        remaining_allocation=remaining,
    )
