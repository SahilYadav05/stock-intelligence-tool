"""Deterministic, point-in-time price-action and market-structure engine.

This module intentionally produces research-only conditional setups.  It never
converts an unapproved model into an official signal and never uses developing
candles or future-confirmed pivots.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from nifty_terminal.calendar.nse import IST
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.candle import Candle, CandleStatus, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.price_action.models import (
    ConditionalTradePlan,
    PriceActionAnalysis,
    PriceActionBias,
    PriceActionLevel,
    SetupState,
)


PRICE_ACTION_VERSION = "causal_market_structure.v1"
MINIMUM_PRIMARY_CANDLES = 50
MINIMUM_CONTEXT_CANDLES = 20
PIVOT_WINDOW = 2
ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class _Pivot:
    index: int
    price: Decimal
    kind: str


@dataclass(frozen=True, slots=True)
class _Evidence:
    label: str
    score: int
    trigger: bool = False


class PriceActionEngine:
    """Builds a reproducible setup from the exact candles named by a snapshot."""

    def analyze(
        self,
        view: MarketStateView,
        *,
        generated_at: datetime | None = None,
    ) -> PriceActionAnalysis:
        generated = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        series = {
            timeframe: tuple(
                sorted(
                    (
                        item
                        for item in view.finalized_candles
                        if item.timeframe is timeframe
                    ),
                    key=lambda item: item.opens_at,
                )
            )
            for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1)
        }
        blockers = _input_blockers(view, series)
        primary = series[Timeframe.M5]
        if len(primary) < 15:
            return _unavailable(view, generated, blockers)

        atr = _atr(primary, 14)
        if atr is None or atr <= ZERO:
            return _unavailable(view, generated, (*blockers, "ATR_UNAVAILABLE"))

        current = primary[-1]
        pivots = _confirmed_pivots(primary)
        pivot_highs = tuple(item for item in pivots if item.kind == "HIGH")
        pivot_lows = tuple(item for item in pivots if item.kind == "LOW")
        structure = _structure(pivot_highs, pivot_lows)
        trends = {
            timeframe: _trend(series[timeframe])
            for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1)
        }
        evidence = list(
            _evidence(
                primary=primary,
                pivot_highs=pivot_highs,
                pivot_lows=pivot_lows,
                structure=structure,
                trends=trends,
                atr=atr,
            )
        )
        raw_score = sum(item.score for item in evidence)
        score = max(-100, min(100, raw_score))
        patterns = _patterns(primary, pivot_highs, pivot_lows, atr)
        supports, resistances = _levels(primary, pivot_highs, pivot_lows, atr)
        trigger_evidence = any(item.trigger and item.score > 0 for item in evidence)
        sell_trigger_evidence = any(item.trigger and item.score < 0 for item in evidence)

        bias = (
            PriceActionBias.BULLISH
            if score >= 25
            else PriceActionBias.BEARISH
            if score <= -25
            else PriceActionBias.NEUTRAL
        )
        setup = _setup_state(
            score=score,
            bullish_trigger=trigger_evidence,
            bearish_trigger=sell_trigger_evidence,
            live=view.snapshot.data_status is ConnectionState.LIVE,
            blockers=blockers,
        )
        plan = _trade_plan(
            setup=setup,
            bias=bias,
            current=current,
            atr=atr,
            pivot_highs=pivot_highs,
            pivot_lows=pivot_lows,
            supports=supports,
            resistances=resistances,
        )
        if plan is not None and plan.blockers:
            blockers = (*blockers, *(f"RISK_PLAN:{item}" for item in plan.blockers))
            if setup is SetupState.BUY_TRIGGER:
                setup = SetupState.BULLISH_WATCH
            elif setup is SetupState.SELL_TRIGGER:
                setup = SetupState.BEARISH_WATCH
        reasons = tuple(item.label for item in evidence if item.score and _supports(item.score, bias))
        contradictory = tuple(
            item.label for item in evidence if item.score and not _supports(item.score, bias)
        )
        grade = (
            "STRONG"
            if abs(score) >= 70
            else "MODERATE"
            if abs(score) >= 45
            else "WEAK"
            if abs(score) >= 25
            else "NO_EDGE"
        )
        return PriceActionAnalysis(
            snapshot_id=view.snapshot.snapshot_id,
            candle_revision_checksum=view.snapshot.candle_revision_checksum,
            instrument_id=view.snapshot.instrument_id,
            decision_time=view.snapshot.decision_time,
            generated_at=generated,
            version=PRICE_ACTION_VERSION,
            bias=bias,
            setup=setup,
            confluence_score=score,
            evidence_grade=grade,
            structure_5m=structure,
            trend_5m=trends[Timeframe.M5],
            trend_15m=trends[Timeframe.M15],
            trend_1h=trends[Timeframe.H1],
            volatility_regime=_volatility_regime(primary, atr),
            patterns=patterns,
            support_levels=supports,
            resistance_levels=resistances,
            reasons=reasons,
            contradictory_evidence=contradictory,
            trade_plan=plan,
            blockers=tuple(sorted(set(blockers))),
        )


def _input_blockers(
    view: MarketStateView,
    series: dict[Timeframe, tuple[Candle, ...]],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if view.snapshot.data_status is not ConnectionState.LIVE:
        blockers.append(f"DATA_NOT_LIVE_{view.snapshot.data_status.value}")
    if len(series[Timeframe.M5]) < MINIMUM_PRIMARY_CANDLES:
        blockers.append(
            f"INSUFFICIENT_5M_HISTORY:{len(series[Timeframe.M5])}/{MINIMUM_PRIMARY_CANDLES}"
        )
    for timeframe in (Timeframe.M15, Timeframe.H1):
        if len(series[timeframe]) < MINIMUM_CONTEXT_CANDLES:
            blockers.append(
                f"INSUFFICIENT_{timeframe.value.upper()}_HISTORY:"
                f"{len(series[timeframe])}/{MINIMUM_CONTEXT_CANDLES}"
            )
    if any(
        item.status is not CandleStatus.FINALIZED
        for candles in series.values()
        for item in candles
    ):
        blockers.append("DEVELOPING_CANDLE_IN_ANALYSIS_INPUT")
    return tuple(blockers)


def _unavailable(
    view: MarketStateView,
    generated_at: datetime,
    blockers: tuple[str, ...],
) -> PriceActionAnalysis:
    return PriceActionAnalysis(
        snapshot_id=view.snapshot.snapshot_id,
        candle_revision_checksum=view.snapshot.candle_revision_checksum,
        instrument_id=view.snapshot.instrument_id,
        decision_time=view.snapshot.decision_time,
        generated_at=generated_at,
        version=PRICE_ACTION_VERSION,
        bias=PriceActionBias.UNAVAILABLE,
        setup=SetupState.UNAVAILABLE,
        confluence_score=0,
        evidence_grade="UNAVAILABLE",
        structure_5m="UNAVAILABLE",
        trend_5m="UNAVAILABLE",
        trend_15m="UNAVAILABLE",
        trend_1h="UNAVAILABLE",
        volatility_regime="UNAVAILABLE",
        patterns=(),
        support_levels=(),
        resistance_levels=(),
        reasons=(),
        contradictory_evidence=(),
        trade_plan=None,
        blockers=tuple(sorted(set(blockers))),
    )


def _atr(candles: tuple[Candle, ...], period: int) -> Decimal | None:
    if len(candles) < period:
        return None
    ranges: list[Decimal] = []
    for index, candle in enumerate(candles):
        if index == 0:
            ranges.append(candle.high - candle.low)
        else:
            previous = candles[index - 1].close
            ranges.append(max(candle.high - candle.low, abs(candle.high - previous), abs(candle.low - previous)))
    value = sum(ranges[:period], ZERO) / Decimal(period)
    for item in ranges[period:]:
        value = (value * Decimal(period - 1) + item) / Decimal(period)
    return value


def _ema(candles: tuple[Candle, ...], period: int) -> Decimal | None:
    if len(candles) < period:
        return None
    closes = [item.close for item in candles]
    value = sum(closes[:period], ZERO) / Decimal(period)
    alpha = Decimal(2) / Decimal(period + 1)
    for close in closes[period:]:
        value = alpha * close + (Decimal(1) - alpha) * value
    return value


def _trend(candles: tuple[Candle, ...]) -> str:
    if len(candles) < MINIMUM_CONTEXT_CANDLES:
        return "UNAVAILABLE"
    ema20 = _ema(candles, 20)
    ema50 = _ema(candles, 50)
    close = candles[-1].close
    if ema20 is None:
        return "UNAVAILABLE"
    if ema50 is None:
        return "BULLISH" if close > ema20 else "BEARISH" if close < ema20 else "MIXED"
    if close > ema20 > ema50:
        return "BULLISH"
    if close < ema20 < ema50:
        return "BEARISH"
    return "MIXED"


def _confirmed_pivots(candles: tuple[Candle, ...]) -> tuple[_Pivot, ...]:
    result: list[_Pivot] = []
    width = PIVOT_WINDOW
    for index in range(width, len(candles) - width):
        left = candles[index - width : index]
        right = candles[index + 1 : index + width + 1]
        candle = candles[index]
        if all(candle.high > item.high for item in (*left, *right)):
            result.append(_Pivot(index=index, price=candle.high, kind="HIGH"))
        if all(candle.low < item.low for item in (*left, *right)):
            result.append(_Pivot(index=index, price=candle.low, kind="LOW"))
    return tuple(result)


def _structure(highs: tuple[_Pivot, ...], lows: tuple[_Pivot, ...]) -> str:
    if len(highs) < 2 or len(lows) < 2:
        return "UNCONFIRMED"
    higher_high = highs[-1].price > highs[-2].price
    higher_low = lows[-1].price > lows[-2].price
    if higher_high and higher_low:
        return "HIGHER_HIGHS_HIGHER_LOWS"
    if not higher_high and not higher_low:
        return "LOWER_HIGHS_LOWER_LOWS"
    return "RANGE_OR_TRANSITION"


def _evidence(
    *,
    primary: tuple[Candle, ...],
    pivot_highs: tuple[_Pivot, ...],
    pivot_lows: tuple[_Pivot, ...],
    structure: str,
    trends: dict[Timeframe, str],
    atr: Decimal,
) -> tuple[_Evidence, ...]:
    result: list[_Evidence] = []
    weights = {Timeframe.M5: 12, Timeframe.M15: 16, Timeframe.H1: 20}
    labels = {Timeframe.M5: "5m", Timeframe.M15: "15m", Timeframe.H1: "1h"}
    for timeframe, weight in weights.items():
        trend = trends[timeframe]
        if trend == "BULLISH":
            result.append(_Evidence(f"{labels[timeframe]} close and EMA structure are bullish", weight))
        elif trend == "BEARISH":
            result.append(_Evidence(f"{labels[timeframe]} close and EMA structure are bearish", -weight))
    if structure == "HIGHER_HIGHS_HIGHER_LOWS":
        result.append(_Evidence("Confirmed 5m swings form higher highs and higher lows", 18))
    elif structure == "LOWER_HIGHS_LOWER_LOWS":
        result.append(_Evidence("Confirmed 5m swings form lower highs and lower lows", -18))

    current = primary[-1]
    buffer = atr * Decimal("0.05")
    if pivot_highs and current.close > pivot_highs[-1].price + buffer:
        result.append(_Evidence("5m close confirmed a break above the latest swing high", 22, True))
    if pivot_lows and current.close < pivot_lows[-1].price - buffer:
        result.append(_Evidence("5m close confirmed a break below the latest swing low", -22, True))
    if pivot_lows and current.low < pivot_lows[-1].price and current.close > pivot_lows[-1].price:
        result.append(_Evidence("Price swept a confirmed swing low and closed back above it", 14, True))
    if pivot_highs and current.high > pivot_highs[-1].price and current.close < pivot_highs[-1].price:
        result.append(_Evidence("Price swept a confirmed swing high and closed back below it", -14, True))

    opening = _opening_range(primary)
    if opening is not None:
        opening_high, opening_low = opening
        if current.close > opening_high + buffer:
            result.append(_Evidence("Price is accepted above the first 30-minute opening range", 12, True))
        elif current.close < opening_low - buffer:
            result.append(_Evidence("Price is accepted below the first 30-minute opening range", -12, True))

    previous_session = _previous_session_levels(primary)
    if previous_session is not None:
        previous_high, previous_low, _ = previous_session
        if current.close > previous_high + buffer:
            result.append(_Evidence("Price closed above the previous session high", 12, True))
        elif current.close < previous_low - buffer:
            result.append(_Evidence("Price closed below the previous session low", -12, True))

    body = current.close - current.open
    total_range = max(current.high - current.low, Decimal("0.01"))
    if abs(body) / total_range >= Decimal("0.65") and total_range >= atr * Decimal("1.1"):
        result.append(
            _Evidence(
                "Latest candle shows directional range expansion",
                8 if body > 0 else -8,
                True,
            )
        )
    pattern_scores = {
        "BULLISH_ENGULFING": 8,
        "HAMMER_REJECTION": 6,
        "BEARISH_ENGULFING": -8,
        "SHOOTING_STAR_REJECTION": -6,
    }
    for pattern in _patterns(primary, pivot_highs, pivot_lows, atr):
        if pattern in pattern_scores:
            result.append(_Evidence(pattern.replace("_", " ").title(), pattern_scores[pattern], True))
    return tuple(result)


def _patterns(
    candles: tuple[Candle, ...],
    pivot_highs: tuple[_Pivot, ...],
    pivot_lows: tuple[_Pivot, ...],
    atr: Decimal,
) -> tuple[str, ...]:
    if len(candles) < 2:
        return ()
    current, previous = candles[-1], candles[-2]
    body = current.close - current.open
    previous_body = previous.close - previous.open
    body_abs = abs(body)
    candle_range = max(current.high - current.low, Decimal("0.01"))
    upper = current.high - max(current.open, current.close)
    lower = min(current.open, current.close) - current.low
    near_support = bool(pivot_lows and abs(current.low - pivot_lows[-1].price) <= atr * Decimal("0.35"))
    near_resistance = bool(pivot_highs and abs(current.high - pivot_highs[-1].price) <= atr * Decimal("0.35"))
    patterns: list[str] = []
    if (
        body > 0
        and previous_body < 0
        and current.open <= previous.close
        and current.close >= previous.open
    ):
        patterns.append("BULLISH_ENGULFING")
    if (
        body < 0
        and previous_body > 0
        and current.open >= previous.close
        and current.close <= previous.open
    ):
        patterns.append("BEARISH_ENGULFING")
    if body_abs / candle_range <= Decimal("0.15"):
        patterns.append("DOJI_INDECISION")
    if lower >= max(body_abs * Decimal(2), candle_range * Decimal("0.45")) and upper <= candle_range * Decimal("0.2") and near_support:
        patterns.append("HAMMER_REJECTION")
    if upper >= max(body_abs * Decimal(2), candle_range * Decimal("0.45")) and lower <= candle_range * Decimal("0.2") and near_resistance:
        patterns.append("SHOOTING_STAR_REJECTION")
    if current.high < previous.high and current.low > previous.low:
        patterns.append("INSIDE_BAR_COMPRESSION")
    if current.high > previous.high and current.low < previous.low:
        patterns.append("OUTSIDE_BAR_EXPANSION")
    return tuple(patterns)


def _levels(
    primary: tuple[Candle, ...],
    highs: tuple[_Pivot, ...],
    lows: tuple[_Pivot, ...],
    atr: Decimal,
) -> tuple[tuple[PriceActionLevel, ...], tuple[PriceActionLevel, ...]]:
    current = primary[-1].close
    candidates: list[tuple[Decimal, str, int]] = []
    candidates.extend((item.price, "SWING_LOW", 2) for item in lows[-8:])
    candidates.extend((item.price, "SWING_HIGH", 2) for item in highs[-8:])
    opening = _opening_range(primary)
    if opening:
        candidates.extend(((opening[0], "OPENING_RANGE_HIGH", 3), (opening[1], "OPENING_RANGE_LOW", 3)))
    previous = _previous_session_levels(primary)
    if previous:
        candidates.extend(
            (
                (previous[0], "PREVIOUS_DAY_HIGH", 3),
                (previous[1], "PREVIOUS_DAY_LOW", 3),
                (previous[2], "PREVIOUS_DAY_CLOSE", 2),
            )
        )
    tolerance = max(atr * Decimal("0.18"), Decimal("0.01"))
    clusters: list[list[tuple[Decimal, str, int]]] = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        cluster = next(
            (
                item
                for item in clusters
                if abs(sum((value[0] for value in item), ZERO) / Decimal(len(item)) - candidate[0])
                <= tolerance
            ),
            None,
        )
        if cluster is None:
            clusters.append([candidate])
        else:
            cluster.append(candidate)
    levels = []
    for cluster in clusters:
        weighted_total = sum((price * Decimal(weight) for price, _, weight in cluster), ZERO)
        total_weight = sum(weight for _, _, weight in cluster)
        price = weighted_total / Decimal(total_weight)
        strongest = max(cluster, key=lambda item: item[2])
        levels.append(
            PriceActionLevel(
                price=price,
                kind=strongest[1],
                strength=min(5, total_weight),
                touches=len(cluster),
            )
        )
    supports = tuple(sorted((item for item in levels if item.price <= current), key=lambda item: item.price, reverse=True)[:3])
    resistances = tuple(sorted((item for item in levels if item.price > current), key=lambda item: item.price)[:3])
    return supports, resistances


def _opening_range(primary: tuple[Candle, ...]) -> tuple[Decimal, Decimal] | None:
    current_day = primary[-1].opens_at.astimezone(IST).date()
    session = tuple(item for item in primary if item.opens_at.astimezone(IST).date() == current_day)
    if len(session) < 6:
        return None
    first = session[:6]
    return max(item.high for item in first), min(item.low for item in first)


def _previous_session_levels(
    primary: tuple[Candle, ...],
) -> tuple[Decimal, Decimal, Decimal] | None:
    current_day = primary[-1].opens_at.astimezone(IST).date()
    earlier = tuple(item for item in primary if item.opens_at.astimezone(IST).date() < current_day)
    if not earlier:
        return None
    previous_day = earlier[-1].opens_at.astimezone(IST).date()
    session = tuple(item for item in earlier if item.opens_at.astimezone(IST).date() == previous_day)
    first_open = session[0].opens_at.astimezone(IST)
    last_close = session[-1].closes_at.astimezone(IST)
    if (first_open.hour, first_open.minute) != (9, 15) or (last_close.hour, last_close.minute) != (15, 30):
        return None
    return max(item.high for item in session), min(item.low for item in session), session[-1].close


def _setup_state(
    *,
    score: int,
    bullish_trigger: bool,
    bearish_trigger: bool,
    live: bool,
    blockers: tuple[str, ...],
) -> SetupState:
    if not live or blockers:
        if score >= 35:
            return SetupState.BULLISH_WATCH
        if score <= -35:
            return SetupState.BEARISH_WATCH
        return SetupState.WAIT
    if score >= 55 and bullish_trigger:
        return SetupState.BUY_TRIGGER
    if score <= -55 and bearish_trigger:
        return SetupState.SELL_TRIGGER
    if score >= 35:
        return SetupState.BULLISH_WATCH
    if score <= -35:
        return SetupState.BEARISH_WATCH
    return SetupState.WAIT


def _trade_plan(
    *,
    setup: SetupState,
    bias: PriceActionBias,
    current: Candle,
    atr: Decimal,
    pivot_highs: tuple[_Pivot, ...],
    pivot_lows: tuple[_Pivot, ...],
    supports: tuple[PriceActionLevel, ...],
    resistances: tuple[PriceActionLevel, ...],
) -> ConditionalTradePlan | None:
    if bias not in {PriceActionBias.BULLISH, PriceActionBias.BEARISH}:
        return None
    is_buy = bias is PriceActionBias.BULLISH
    direction = "BUY" if is_buy else "SELL"
    trigger_buffer = atr * Decimal("0.05")
    if setup in {SetupState.BUY_TRIGGER, SetupState.SELL_TRIGGER}:
        trigger = current.close
    elif is_buy:
        reference = pivot_highs[-1].price if pivot_highs else current.high
        trigger = max(current.close, reference + trigger_buffer)
    else:
        reference = pivot_lows[-1].price if pivot_lows else current.low
        trigger = min(current.close, reference - trigger_buffer)

    structure_levels = supports if is_buy else resistances
    structural = structure_levels[0].price if structure_levels else None
    minimum_risk = atr * Decimal("0.65")
    maximum_risk = atr * Decimal("1.50")
    if structural is None:
        risk = atr
        invalidation = trigger - risk if is_buy else trigger + risk
    else:
        invalidation = structural - trigger_buffer if is_buy else structural + trigger_buffer
        structural_risk = trigger - invalidation if is_buy else invalidation - trigger
        risk = max(minimum_risk, structural_risk)
        invalidation = trigger - risk if is_buy else trigger + risk
    blockers: list[str] = []
    if risk > maximum_risk:
        blockers.append("STRUCTURE_STOP_EXCEEDS_1.5_ATR")
    sign = Decimal(1) if is_buy else Decimal(-1)
    entry_half_width = atr * Decimal("0.05")
    stop = trigger - sign * risk
    target1 = trigger + sign * risk * Decimal("1.25")
    target2 = trigger + sign * risk * Decimal("2.00")
    target3 = trigger + sign * risk * Decimal("3.00")
    opposing = resistances[0].price if is_buy and resistances else supports[0].price if not is_buy and supports else None
    if opposing is not None:
        room = opposing - trigger if is_buy else trigger - opposing
        if room > ZERO and room < risk:
            blockers.append("NEAREST_OPPOSING_LEVEL_INSIDE_1R")
    return ConditionalTradePlan(
        direction=direction,
        trigger=trigger,
        entry_low=trigger - entry_half_width,
        entry_high=trigger + entry_half_width,
        stop=stop,
        invalidation=invalidation,
        target1=target1,
        target2=target2,
        target3=target3,
        risk_points=risk,
        target1_reward_risk=1.25,
        target2_reward_risk=2.0,
        target3_reward_risk=3.0,
        expiry_bars=12,
        blockers=tuple(blockers),
    )


def _volatility_regime(primary: tuple[Candle, ...], current_atr: Decimal) -> str:
    if len(primary) < 35:
        return "UNAVAILABLE"
    atr_history = tuple(
        value
        for index in range(14, len(primary))
        if (value := _atr(primary[: index + 1], 14)) is not None
    )[-20:]
    if not atr_history:
        return "UNAVAILABLE"
    median = sorted(atr_history)[len(atr_history) // 2]
    ratio = current_atr / median if median > ZERO else Decimal(1)
    return "EXPANDING" if ratio >= Decimal("1.25") else "COMPRESSED" if ratio <= Decimal("0.80") else "NORMAL"


def _supports(score: int, bias: PriceActionBias) -> bool:
    return (score > 0 and bias is PriceActionBias.BULLISH) or (
        score < 0 and bias is PriceActionBias.BEARISH
    )
